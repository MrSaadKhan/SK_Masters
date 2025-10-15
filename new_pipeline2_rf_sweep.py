#!/usr/bin/env python3
# new_pipeline2_rf_sweep_with_cache_stream_incremental_fixed.py
"""
Sweep RF_TRAIN_PCT from 2% to 70% (step 2%), reuse existing per-device embedding caches,
and only compute embeddings for devices that are missing or invalid.

FIX: Use numpy.lib.format.open_memmap to create true .npy files (with header) so
      np.load(..., mmap_mode='r') works reliably. Also robustly handle old/broken
      files that were created without headers (previous raw memmap writes).
"""
import os
import csv
import textwrap
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertTokenizer, BertModel
import torch
import sys
import matplotlib.pyplot as plt

# ---------------- Config ----------------
device_list = [
    "irobot_roomba.json",
    "line_clova_wave.json",
    "nature_remo.json",
    "qrio_hub.json",
    "xiaomi_mijia_led.json",
    "powerelectric_wi-fi_plug.json",
    "planex_smacam_outdoor.json"
]
INPUT_FOLDER = "preprocessed_data_group_merged/ungrouped"

# Sweep settings
START_PCT = 0.02
END_PCT = 0.70
STEP_PCT = 0.02
RF_VAL_PCT = 0.30   # fixed: reserve last 30% per device for validation

# BERT / RF config
BERT_MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RF_N_ESTIMATORS = 500
RF_N_JOBS = -1

# Embedding cache folder
EMB_DIR = "embeddings"

# Misc / performance
BATCH_TOKEN_MAX_LEN = 512
EMBED_BATCH_SIZE = 32   # how many texts tokenized as a batch (one forward per batch)
PRINT_SAMPLE_LINES = True

# ---------------- Helpers ----------------
def short(s, width=120):
    return textwrap.shorten(s.replace("\n", "\\n"), width=width, placeholder="…")

def count_nonempty_lines(path):
    cnt = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt

def load_nonempty_lines(file_path):
    """Return list of non-empty lines (preserve order)."""
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.rstrip("\n")
            if s.strip():
                lines.append(s)
    return lines

def get_bert_embedding_batch(sentences, tokenizer, model, max_length=512):
    """
    Compute embeddings for a list of sentences in one forward pass (batched).
    Returns numpy array shape (len(sentences), hidden_size).
    """
    model.eval()
    with torch.no_grad():
        enc = tokenizer(sentences, return_tensors='pt', truncation=True,
                        max_length=max_length, padding=True).to(DEVICE)
        outputs = model(**enc)
        cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls  # shape (B, H)

def interleave_round_robin(per_device_lists, devices_order, device_label_map):
    """
    Round-robin combine lists without truncation.
    Returns combined_items, combined_labels.
    """
    pointers = {d: 0 for d in devices_order}
    total_remaining = sum(len(per_device_lists[d]) for d in devices_order)
    combined = []
    labels = []
    while total_remaining > 0:
        for d in devices_order:
            idx = pointers[d]
            lst = per_device_lists[d]
            if idx < len(lst):
                combined.append(lst[idx])
                labels.append(device_label_map[d])
                pointers[d] += 1
                total_remaining -= 1
    return combined, labels

# ---------------- Prepare environment ----------------
os.makedirs(EMB_DIR, exist_ok=True)

print("Loading tokenizer and BERT (this may take a moment)...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()  # not fine-tuning here

device_label_map = {dev: idx for idx, dev in enumerate(device_list)}

# ---------------- Precompute / load lines and set up caches ----------------
per_device_lines = {}
per_device_total_lines = {}

for device in device_list:
    file_name = device.replace('.json', '.json.txt')
    file_path = os.path.join(INPUT_FOLDER, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    total_lines = count_nonempty_lines(file_path)
    if total_lines == 0:
        raise ValueError(f"No non-empty lines in {file_path}")
    per_device_total_lines[device] = total_lines

    # prepare lines cache file
    safe_name = device.replace('.json', '')
    lines_cache_path = os.path.join(EMB_DIR, f"{safe_name}.lines.txt")

    if os.path.exists(lines_cache_path):
        lines = load_nonempty_lines(lines_cache_path)
        if len(lines) != total_lines:
            print(f"  lines cache mismatch for {device}: cached={len(lines)} vs file={total_lines}. Recreating.")
            lines = load_nonempty_lines(file_path)
            with open(lines_cache_path, 'w', encoding='utf-8') as wf:
                for L in lines:
                    wf.write(L + "\n")
    else:
        lines = load_nonempty_lines(file_path)
        with open(lines_cache_path, 'w', encoding='utf-8') as wf:
            for L in lines:
                wf.write(L + "\n")

    per_device_lines[device] = lines

print("\nAll device line counts:")
for d in device_list:
    print(f"  {d}: total non-empty lines = {per_device_total_lines[d]}")
print("")

# ---------------- Streaming embedding computation & caching ----------------
from numpy.lib.format import open_memmap

def compute_and_cache_embeddings_stream(device, batch_size=EMBED_BATCH_SIZE):
    """
    Compute embeddings in batches and write each row immediately to disk.
    Use numpy.lib.format.open_memmap to create a valid .npy file (with header),
    so that np.load(..., mmap_mode='r') can later read it reliably.

    Creates:
      - embeddings/<device_basename>.emb.npy  (binary .npy file, shape=(N,H), dtype=float32)
      - embeddings/<device_basename>.emb.txt  (textual file: rows of floats)  (optional)
    Returns the numpy memmapped array loaded with np.load(..., mmap_mode='r').
    """
    safe_name = device.replace('.json', '')
    lines = per_device_lines[device]
    n = len(lines)
    emb_npy_path = os.path.join(EMB_DIR, f"{safe_name}.emb.npy")
    emb_txt_path = os.path.join(EMB_DIR, f"{safe_name}.emb.txt")

    # If a stale/incorrect .npy exists but isn't a valid .npy, remove it first
    if os.path.exists(emb_npy_path):
        try:
            # try load header-only to validate
            candidate = np.load(emb_npy_path, mmap_mode='r')
            if candidate.ndim == 2 and candidate.shape[0] == n and candidate.shape[1] == bert_model.config.hidden_size:
                print(f"  Existing .npy for {device} looks valid; reusing: {emb_npy_path}")
                return candidate
            else:
                print(f"  Existing .npy shape mismatch for {device}: {getattr(candidate, 'shape', None)} vs expected=({n},{bert_model.config.hidden_size}). Recomputing.")
                try:
                    os.remove(emb_npy_path)
                except OSError:
                    pass
                if os.path.exists(emb_txt_path):
                    try:
                        os.remove(emb_txt_path)
                    except OSError:
                        pass
        except ValueError as ve:
            # this happens for files that are not valid .npy (e.g. raw memmap dump)
            print(f"  Existing file {emb_npy_path} is not a valid .npy (ValueError). Will remove and recompute. Detail: {ve}")
            try:
                os.remove(emb_npy_path)
            except OSError:
                pass
            if os.path.exists(emb_txt_path):
                try:
                    os.remove(emb_txt_path)
                except OSError:
                    pass
        except Exception as e:
            print(f"  Error while validating existing .npy for {device}: {e}. Will recompute.")
            try:
                os.remove(emb_npy_path)
            except OSError:
                pass
            if os.path.exists(emb_txt_path):
                try:
                    os.remove(emb_txt_path)
                except OSError:
                    pass

    hidden_size = bert_model.config.hidden_size

    # Create a proper .npy memmap using numpy.lib.format.open_memmap
    print(f"Streaming-computing embeddings for {device} ({n} lines). Writing .npy -> {emb_npy_path} and text -> {emb_txt_path}")
    mm = open_memmap(emb_npy_path, dtype='float32', mode='w+', shape=(n, hidden_size))

    # open text file for incremental write
    txt_f = open(emb_txt_path, 'w', encoding='utf-8')

    # iterate in batches, compute batch outputs, write to memmap and txt
    bert_model.eval()
    with torch.no_grad():
        idx = 0
        for start in tqdm(range(0, n, batch_size), desc=f"Embedding (stream) {safe_name}"):
            batch_texts = lines[start:start+batch_size]
            batch_embs = get_bert_embedding_batch(batch_texts, tokenizer, bert_model, max_length=BATCH_TOKEN_MAX_LEN)
            B = batch_embs.shape[0]
            for b in range(B):
                row = batch_embs[b].astype(np.float32)
                mm[idx, :] = row  # write to memmap-backed .npy
                # write textual version (space separated)
                txt_f.write(" ".join(f"{float(v):.6e}" for v in row) + "\n")
                idx += 1
    txt_f.flush()
    txt_f.close()
    mm.flush()
    # load as read-only memmap via np.load (gives proper array-like with shape)
    arr = np.load(emb_npy_path, mmap_mode='r')
    print(f"  Finished {device}. .npy saved at {emb_npy_path} shape={arr.shape}, txt saved at {emb_txt_path}")
    return arr

def load_embeddings_memmap_for_device(device):
    """
    Returns a numpy array-like for the device embeddings.
    If not present or invalid, computes via streaming.

    This function now:
      - attempts to load with np.load(..., mmap_mode='r') which reads the .npy header correctly
      - handles older broken files by catching ValueError and recomputing
    """
    safe_name = device.replace('.json', '')
    emb_npy_path = os.path.join(EMB_DIR, f"{safe_name}.emb.npy")
    emb_txt_path = os.path.join(EMB_DIR, f"{safe_name}.emb.txt")
    total_lines = per_device_total_lines[device]
    hidden_size = bert_model.config.hidden_size

    if os.path.exists(emb_npy_path):
        try:
            arr = np.load(emb_npy_path, mmap_mode='r')
            # if arr is 1D but length divisible by hidden_size, attempt reshape
            if arr.ndim == 1:
                if arr.size % hidden_size == 0:
                    rows = arr.size // hidden_size
                    arr = arr.reshape((rows, hidden_size))
                else:
                    raise ValueError("1D .npy cannot be reshaped to (rows, hidden_size)")
            if arr.ndim == 2 and arr.shape[0] == total_lines and arr.shape[1] == hidden_size:
                return arr
            else:
                print(f"  .npy shape mismatch for {device}: found={getattr(arr, 'shape', None)}, expected=({total_lines},{hidden_size}). Recomputing.")
                try:
                    os.remove(emb_npy_path)
                except OSError:
                    pass
                if os.path.exists(emb_txt_path):
                    try:
                        os.remove(emb_txt_path)
                    except OSError:
                        pass
        except ValueError as ve:
            # invalid .npy (e.g. raw binary) -> remove and recompute
            print(f"  Existing file {emb_npy_path} is not a valid .npy (ValueError): {ve}. Recomputing.")
            try:
                os.remove(emb_npy_path)
            except OSError:
                pass
            if os.path.exists(emb_txt_path):
                try:
                    os.remove(emb_txt_path)
                except OSError:
                    pass
        except Exception as e:
            print(f"  Failed to load existing .npy for {device}: {e}. Recomputing.")
            try:
                os.remove(emb_npy_path)
            except OSError:
                pass
            if os.path.exists(emb_txt_path):
                try:
                    os.remove(emb_txt_path)
                except OSError:
                    pass

    # compute & return
    return compute_and_cache_embeddings_stream(device, batch_size=EMBED_BATCH_SIZE)

# ---------------- New: compute only missing embeddings up-front ----------------
def compute_missing_embeddings(devices):
    """
    For each device in devices: if embeddings .npy exists and rows match, skip;
    otherwise compute embeddings for that device and save .npy + .txt.
    """
    to_compute = []
    for device in devices:
        safe_name = device.replace('.json', '')
        emb_npy_path = os.path.join(EMB_DIR, f"{safe_name}.emb.npy")
        total_lines = per_device_total_lines[device]
        if os.path.exists(emb_npy_path):
            try:
                arr = np.load(emb_npy_path, mmap_mode='r')
                if arr.ndim == 1:
                    if arr.size % bert_model.config.hidden_size == 0:
                        arr = arr.reshape((-1, bert_model.config.hidden_size))
                if arr.ndim == 2 and arr.shape[0] == total_lines:
                    print(f"Skipping {device}: existing embedding .npy ok ({arr.shape[0]} rows).")
                    continue
                else:
                    print(f"Will recompute {device}: .npy rows {getattr(arr, 'shape', None)} != file rows {total_lines}.")
                    to_compute.append(device)
            except Exception:
                print(f"Will recompute {device}: unable to read .npy (or invalid).")
                to_compute.append(device)
        else:
            print(f"Will compute embeddings for {device} (no existing .npy).")
            to_compute.append(device)

    for device in to_compute:
        compute_and_cache_embeddings_stream(device, batch_size=EMBED_BATCH_SIZE)

# compute missing caches once (so reruns with added/changed device_list reuse existing caches)
compute_missing_embeddings(device_list)

# ---------------- Sweep loop (uses memmaps) ----------------
results = []  # list of dicts: {'rf_train_pct': pct, 'macro_f1': val, 'n_train': n, 'n_val': n}

rf_train_pcts = np.arange(START_PCT, END_PCT + 1e-9, STEP_PCT)
for pct in rf_train_pcts:
    RF_TRAIN_PCT = float(pct)
    print("------------------------------------------------------------")
    print(f"Running sweep step: RF_TRAIN_PCT = {RF_TRAIN_PCT*100:.1f}%")
    # Build per-device train and val index lists
    per_device_train_emb_idxs = {}
    per_device_val_emb_idxs = {}
    total_train = 0
    total_val = 0
    for device in device_list:
        total_lines = per_device_total_lines[device]

        # compute counts
        rf_val_count = int(total_lines * RF_VAL_PCT)
        rf_val_count = max(1, rf_val_count)

        rf_train_count = int(total_lines * RF_TRAIN_PCT)
        rf_train_count = max(1, rf_train_count)

        # ensure no overlap
        if rf_train_count + rf_val_count > total_lines:
            new_rf_train_count = max(1, total_lines - rf_val_count)
            print(f"  Warning device {device}: train+val ({rf_train_count}+{rf_val_count}) > total ({total_lines}). Reducing train to {new_rf_train_count}.")
            rf_train_count = new_rf_train_count

        train_idxs = list(range(0, rf_train_count))
        val_idxs = list(range(total_lines - rf_val_count, total_lines))

        per_device_train_emb_idxs[device] = train_idxs
        per_device_val_emb_idxs[device] = val_idxs
        total_train += len(train_idxs)
        total_val += len(val_idxs)

        # print ranges and samples
        train_range_1b = (1, len(train_idxs))
        val_range_1b = (total_lines - len(val_idxs) + 1, total_lines)
        if PRINT_SAMPLE_LINES and len(train_idxs) > 0 and len(val_idxs) > 0:
            lines = per_device_lines[device]
            s_first = short(lines[train_idxs[0]]) if train_idxs else "<none>"
            s_last = short(lines[train_idxs[-1]]) if train_idxs else "<none>"
            v_first = short(lines[val_idxs[0]]) if val_idxs else "<none>"
            v_last = short(lines[val_idxs[-1]]) if val_idxs else "<none>"
            print(f"  {device}: total={total_lines}, train lines 1-based={train_range_1b} (count={len(train_idxs)}), val lines 1-based={val_range_1b} (count={len(val_idxs)})")
            print(f"    train sample first/last: {s_first} / {s_last}")
            print(f"    val   sample first/last: {v_first} / {v_last}")
        else:
            print(f"  {device}: total={total_lines}, train_count={len(train_idxs)}, val_count={len(val_idxs)}")

    print(f"  Combined totals before interleave: n_train={total_train}, n_val={total_val}")

    # Build per-device embeddings memmaps (should exist now) and select rows using indices
    per_device_train_lists = {}
    per_device_val_lists = {}
    for device in device_list:
        emb_mm = load_embeddings_memmap_for_device(device)  # array-like shape (N, H)
        t_idxs = per_device_train_emb_idxs[device]
        v_idxs = per_device_val_emb_idxs[device]
        # Instead of slicing entire arrays at once, we create lists of 1D arrays referenced from memmap
        per_device_train_lists[device] = [emb_mm[i] for i in t_idxs] if len(t_idxs) > 0 else []
        per_device_val_lists[device]   = [emb_mm[i] for i in v_idxs] if len(v_idxs) > 0 else []

    # Interleave
    train_emb_list, train_labels = interleave_round_robin(per_device_train_lists, device_list, device_label_map)
    val_emb_list, val_labels     = interleave_round_robin(per_device_val_lists, device_list, device_label_map)

    # Convert lists to arrays for RF (this will allocate arrays of size n_train x H and n_val x H)
    if len(train_emb_list) == 0:
        print("  No training samples available for this pct; skipping.")
        continue
    X_train = np.vstack(train_emb_list).astype(np.float32)
    y_train = np.array(train_labels, dtype=int)
    X_val = np.vstack(val_emb_list).astype(np.float32) if len(val_emb_list) > 0 else np.zeros((0, X_train.shape[1]), dtype=np.float32)
    y_val = np.array(val_labels, dtype=int) if len(val_emb_list) > 0 else np.array([], dtype=int)

    print(f"  Combined totals after interleave: n_train={X_train.shape[0]}, n_val={X_val.shape[0]}")

    # Train RF
    print("  Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=RF_N_JOBS, random_state=42)
    rf.fit(X_train, y_train)

    # Eval
    if X_val.shape[0] == 0:
        print("  No validation samples available; skipping evaluation for this pct.")
        continue
    y_pred = rf.predict(X_val)
    f1m = f1_score(y_val, y_pred, average='macro')
    print(f"  -> Macro F1 at RF_TRAIN_PCT={RF_TRAIN_PCT*100:.1f}%: {f1m:.4f}")

    results.append({
        "rf_train_pct": RF_TRAIN_PCT,
        "macro_f1": float(f1m),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0])
    })

# ---------------- Save results and plot ----------------
if len(results) == 0:
    print("No results collected. Exiting.")
    sys.exit(0)

csv_path = "rf_train_pct_results.csv"
with open(csv_path, "w", newline='', encoding='utf-8') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=["rf_train_pct", "n_train", "n_val", "macro_f1"])
    writer.writeheader()
    for r in results:
        writer.writerow({"rf_train_pct": r["rf_train_pct"], "n_train": r["n_train"], "n_val": r["n_val"], "macro_f1": r["macro_f1"]})
print(f"\nSaved numeric results to: {csv_path}")

# prepare plot
pcts = [r["rf_train_pct"] * 100.0 for r in results]  # percent
f1s = [r["macro_f1"] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(pcts, f1s, marker='o')  # default matplotlib styling
plt.xlabel("RF_TRAIN_PCT (%)")
plt.ylabel("Macro F1 (validation)")
plt.title("RF training size (pct of file) vs Macro F1 (validation, last 30% per device)")
plt.grid(True)
plt.tight_layout()
plot_path = "rf_train_pct_vs_f1.png"
plt.savefig(plot_path)
print(f"Saved plot to: {plot_path}")

print("\nSummary results (RF_TRAIN_PCT%, n_train, n_val, macro_f1):")
for r in results:
    print(f"  {r['rf_train_pct']*100:5.1f}%  |  n_train={r['n_train']:6d}  n_val={r['n_val']:6d}  macro_f1={r['macro_f1']:.4f}")

print("\nDone. Files generated:")
print(f"  - {csv_path}")
print(f"  - {plot_path}")
print(f"  - embedding files (per device) in folder: {EMB_DIR}")
