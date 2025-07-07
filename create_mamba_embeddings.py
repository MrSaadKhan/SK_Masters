import os
import numpy as np
import torch
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import get_data
from tqdm import tqdm
from transformers.utils import logging

logging.set_verbosity_error()  # Suppress HF warnings

_global_model = None
_global_tokenizer = None
_global_max_length = None


def _init_model(model_name: str, max_length: int, hf_token: str | None = None):
    global _global_model, _global_tokenizer, _global_max_length
    if _global_model is not None:
        return

    _global_max_length = max_length
    tok_kwargs = {}
    if hf_token:
        tok_kwargs['use_auth_token'] = hf_token

    print("[Init] Loading tokenizer …")
    _global_tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    print("[Init] Loading config & model …")
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.output_hidden_states = True

    mdl_kwargs = {'config': cfg, 'torch_dtype': torch.float16}
    if hf_token:
        mdl_kwargs['use_auth_token'] = hf_token

    _global_model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs).to('cpu')
    _global_model.eval()
    print("[Init] Model ready ✔️")


def _embed_sentence(sentence: str) -> np.ndarray:
    inputs = _global_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=_global_max_length,
    ).to(_global_model.device)

    with torch.no_grad():
        outputs = _global_model(**inputs)

    last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden)
    mask = inputs["attention_mask"][0].unsqueeze(-1)  # (seq_len, 1)
    summed = (last_hidden * mask).sum(dim=0)  # (hidden,)
    counts = mask.sum(dim=0)  # (1,)
    return (summed / counts).cpu().numpy()  # (hidden,)


def _embed_worker(sentence: str) -> str:
    try:
        vec = _embed_sentence(sentence)
        return " ".join(map(str, vec))
    except Exception:
        return ""  # silently ignore errors for line ordering


def _write_embeddings_parallel(
    sentences: list[str],
    outfile: str,
    model_name: str,
    max_length: int,
    hf_token: str | None = None,
    chunksize: int = 1,
    desc: str = "",   # <-- added description for tqdm
):
    _init_model(model_name, max_length, hf_token)

    try:
        mp.set_start_method("fork", force=False)
    except RuntimeError:
        pass

    with mp.Pool(processes=55) as pool, open(outfile, "w") as sink:
        for idx, vec_str in enumerate(
            tqdm(pool.imap(_embed_worker, sentences, chunksize=chunksize), total=len(sentences), desc=desc)
        ):
            sink.write(vec_str + "\n")


def create_device_mamba_embeddings(
    model_name: str,
    device: str,
    save_dir: str,
    data_dir: str,
    vector_size: int = 512,
    hf_token: str | None = None,
    max_total_embeddings: int = 0,
) -> tuple[int, int]:
    seen_file = os.path.join(save_dir, f"{device}_seen_mamba_embeddings.txt")
    unseen_file = os.path.join(save_dir, f"{device}_unseen_mamba_embeddings.txt")

    if os.path.exists(seen_file) and os.path.exists(unseen_file):
        return 0, 0

    seen, unseen = get_data.get_data(data_dir, device)
    seen_texts = [s[0] for s in seen]
    unseen_texts = [s[0] for s in unseen]

    if max_total_embeddings:
        seen_cap = int(0.7 * max_total_embeddings)
        unseen_cap = max_total_embeddings - seen_cap
        seen_texts = seen_texts[: min(seen_cap, len(seen_texts))]
        unseen_texts = unseen_texts[: min(unseen_cap, len(unseen_texts))]

    print(f"[Device] → {device}")
    print(f"Embedding {device}: {len(seen_texts)} seen / {len(unseen_texts)} unseen")

    _write_embeddings_parallel(seen_texts, seen_file, model_name, vector_size, hf_token, desc=f"{device} (Seen)")
    _write_embeddings_parallel(unseen_texts, unseen_file, model_name, vector_size, hf_token, desc=f"{device} (Unseen)")

    print(f"[Device] ✔️ {device}: seen={len(seen_texts)}, unseen={len(unseen_texts)}")

    return len(seen_texts), len(unseen_texts)


def create_embeddings(
    file_path: str,
    device_list: list[str],
    save_dir: str,
    data_dir: str,
    group_option: int,
    word_embedding_option: int,
    window_size: int,
    slide_length: int,
    vector_size: int = 512,
    hf_token: str | None = None,
    max_total_embeddings: int = 0,
) -> tuple[int, int]:
    model_name = "state-spaces/mamba-130m-hf"
    grouping = "Grouped" if group_option else "Ungrouped"
    embeddings_dir = os.path.join(save_dir, grouping, f"{window_size}_{slide_length}", "mamba_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    total_seen = 0
    total_unseen = 0

    for device in device_list:
        s, u = create_device_mamba_embeddings(
            model_name,
            device,
            embeddings_dir,
            data_dir,
            vector_size,
            hf_token,
            max_total_embeddings,
        )
        total_seen += s
        total_unseen += u

    print(f"[Summary] total_seen={total_seen}, total_unseen={total_unseen}")
    return total_seen, total_unseen
