# bert_embed_line_mp.py
# -*- coding: utf-8 -*-
"""
Line-level multiprocessing BERT embeddings.
 • Sequential over devices   (one device fully processed before the next)
 • Parallel   over sentences (tasks distributed across all CPU cores)
Tested on Python 3.10, CPU-only, PyTorch 2.x.
"""

# ---------------------------------------------------------------------------
# 1.  SET PYTORCH THREAD COUNTS **BEFORE** ANY TORCH / TRANSFORMERS IMPORTS
# ---------------------------------------------------------------------------
import os
import torch

# Single MKL / OpenMP thread per worker; safe in the main process too.
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# 2.  NOW we can import everything else
# ---------------------------------------------------------------------------
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

import get_data  # <-- your local helper, unchanged


# ---------------------------------------------------------------------------
# 3.  PER-WORKER GLOBALS & INITIALISER
# ---------------------------------------------------------------------------
_WORKER_TOKENIZER = None   # loaded once per worker
_WORKER_MODEL = None


def _worker_init(model_name: str) -> None:
    """
    Runs ONCE in every worker process (called by ProcessPoolExecutor).
    Loads BERT and tokenizer; threads are already capped globally.
    """
    global _WORKER_TOKENIZER, _WORKER_MODEL

    # Lazily load; with 'fork' the parent already holds the weights in RAM,
    # so children share them via copy-on-write (very memory-efficient).
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    _WORKER_MODEL = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True, local_files_only=True)


def _encode_sentence(task: Tuple[int, str]) -> Tuple[int, str]:
    """
    Worker function: embeds ONE sentence.
    Returns (index, embedding_line) so the caller can re-insert in order.
    `task` is (index, sentence_text).
    """
    idx, sentence = task
    tok = _WORKER_TOKENIZER
    mdl = _WORKER_MODEL

    inputs = tok(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(mdl.device)

    with torch.no_grad():
        hidden = mdl(**inputs, output_hidden_states=True).hidden_states[-1]
        vec = hidden[0, 0, :].cpu().numpy()

    return idx, " ".join(map(str, vec))


# ---------------------------------------------------------------------------
# 4.  OPTIONAL FINE-TUNE (unchanged from your earlier version)
# ---------------------------------------------------------------------------
def fine_tune_model(
    base_model,
    tokenizer,
    sentences,
    vector_size: int,
    lr: float = 4e-5,
    epochs: int = 3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base_model.to(device)

    inputs = tokenizer(
        [s[0] for s in sentences],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=vector_size,
    )
    ds = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for ids, masks in tqdm(dl, desc=f"Fine-tune epoch {ep+1}/{epochs}"):
            ids, masks = ids.to(device), masks.to(device)
            loss = model(input_ids=ids, attention_mask=masks, labels=ids).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f" epoch {ep+1}: loss={total:.4f}")

    model.to("cpu").eval()


# ---------------------------------------------------------------------------
# 5.  MAIN  create_embeddings()  FUNCTION
# ---------------------------------------------------------------------------
def create_embeddings(
    file_path: str,
    device_list: List[str],
    save_dir: str,
    data_dir: str,
    group_option: bool,
    word_embedding_option,
    window_size: int,
    slide_length: int,
    vector_size: int = 768,
    fine_tune_percent: float = 0.9,
    num_workers: Optional[int] = None,
) -> Tuple[int, int, Optional[int]]:
    """
    Returns (total_seen_lines, total_unseen_lines, sentinel)
    sentinel = 0 if something was written, else None.
    """

    # ----- pick model ------------------------------------------------------
    model_dict = {
        128: "google/bert_uncased_L-2_H-128_A-2",
        256: "google/bert_uncased_L-4_H-256_A-4",
        512: "google/bert_uncased_L-8_H-512_A-8",
        768: "google-bert/bert-base-uncased",
    }
    if vector_size not in model_dict:
        raise ValueError(f"vector_size must be one of {list(model_dict)}")
    model_name = model_dict[vector_size]

    # ----- directory layout ------------------------------------------------
    top = os.path.join(
        save_dir,
        "Grouped" if group_option else "Ungrouped",
        f"{window_size}_{slide_length}",
    )
    os.makedirs(top, exist_ok=True)
    embed_dir = os.path.join(top, "bert_embeddings")
    os.makedirs(embed_dir, exist_ok=True)

    # ----- preload model once in parent (helps 'fork' share RAM) ----------
    parent_tokenizer = AutoTokenizer.from_pretrained(model_name)
    parent_model = AutoModelForMaskedLM.from_pretrained(
        model_name, output_hidden_states=True
    ).eval().to("cpu")
    torch.save(parent_model.state_dict(), os.path.join(top, "model.pth"))
    print("Model & tokenizer pre-loaded in parent.")

    # ----- fine-tune (optional, disabled by default) -----------------------
    fine_tune_option = False
    if fine_tune_option:
        all_seen = []
        for dev in device_list:
            seen, _ = get_data.get_data(data_dir, dev)
            all_seen.extend(seen[: int(len(seen) * fine_tune_percent)])
        fine_tune_model(parent_model, parent_tokenizer, all_seen, vector_size)

    # ----- choose pool settings -------------------------------------------
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(method)
    print(f"Using {num_workers} worker(s), start-method='{method}'")

    total_seen = 0
    total_unseen = 0

    # ----------------------------------------------------------------------
    # 6.  PROCESS EACH DEVICE SEQUENTIALLY  (core of your requirement)
    # ----------------------------------------------------------------------
    for dev in device_list:
        seen, unseen = get_data.get_data(data_dir, dev)
        if not seen and not unseen:
            print(f"⚠  No data for device {dev}")
            continue

        # Pre-allocate result lists to preserve order
        seen_out = ["" for _ in range(len(seen))]
        unseen_out = ["" for _ in range(len(unseen))]

        # Build tasks list for this device only
        tasks = [(i, s[0]) for i, s in enumerate(seen)] + [
            (i + len(seen), s[0]) for i, s in enumerate(unseen)
        ]

        # Map tasks to pool --------------------------------------------------
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(model_name,),
        ) as pool:
            futs = [pool.submit(_encode_sentence, t) for t in tasks]

            for fut in tqdm(
                as_completed(futs),
                total=len(futs),
                desc=f"Embedding {dev}",
            ):
                idx, emb = fut.result()
                if idx < len(seen):
                    seen_out[idx] = emb
                else:
                    unseen_out[idx - len(seen)] = emb

        # Write results -----------------------------------------------------
        dev_seen_file = os.path.join(embed_dir, f"{dev}_seen_bert_embeddings.txt")
        dev_unseen_file = os.path.join(embed_dir, f"{dev}_unseen_bert_embeddings.txt")
        with open(dev_seen_file, "w") as f:
            f.write("\n".join(seen_out))
        with open(dev_unseen_file, "w") as f:
            f.write("\n".join(unseen_out))

        total_seen += len(seen_out)
        total_unseen += len(unseen_out)
        print(f"✓  {dev}: wrote {len(seen_out)} seen, {len(unseen_out)} unseen")

    sentinel = 0 if (total_seen + total_unseen) else None
    return total_seen, total_unseen, sentinel

