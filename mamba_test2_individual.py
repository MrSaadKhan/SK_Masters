#!/usr/bin/env python3
"""
compute_mamba_token_stats.py
------------------------------------------------
Compute Mamba token counts for every line in every .txt file under a directory.
Saves per-line token counts, summary statistics, and CCDF plots (static and interactive).

No CLI arguments: configure paths via variables below.

Dependencies:
    pip install transformers tqdm numpy matplotlib plotly
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import plotly.express as px

# ───── CONFIGURATION ─────────────────────────────────────────────────────────
# Directory containing your .txt files:
DIR_WITH_TXTS = Path("/home/iotresearch/saad/masters/preprocessed_data/ungrouped")
# Output directory (will be created/overwritten):
OUT_DIR = Path.cwd() / "mamba_token_output"
# Number of parallel workers (None = CPU count):
WORKERS = 50
# Mamba tokenizer model
MODEL_NAME = "state-spaces/mamba-130m-hf"

# ───── WORKER INITIALIZATION ──────────────────────────────────────────────────
_TOKENIZER = None

def _init_worker(model_name: str):
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)

# ───── FILE PROCESSING ─────────────────────────────────────────────────────────
def _process_file(path: Path):
    """
    Tokenize each line in `path`, return a list of (filepath, lineno, token_count).
    """
    global _TOKENIZER
    results = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, start=1):
            text = line.rstrip("\n")
            count = len(_TOKENIZER.encode(text, add_special_tokens=False))
            results.append((str(path), lineno, count))
    return results

# ───── MAIN FUNCTION ──────────────────────────────────────────────────────────
def main():
    if not DIR_WITH_TXTS.is_dir():
        sys.exit(f"Directory not found: {DIR_WITH_TXTS}")

    txt_files = list(DIR_WITH_TXTS.rglob("*.txt"))
    if not txt_files:
        sys.exit(f"No .txt files found in {DIR_WITH_TXTS}")

    # Prepare output
    OUT_DIR.mkdir(exist_ok=True)
    per_line_file = OUT_DIR / "line_token_counts.txt"
    stats_file = OUT_DIR / "token_stats.txt"

    all_counts = []
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=_init_worker,
                             initargs=(MODEL_NAME,)) as pool, \
         tqdm(total=len(txt_files), desc="Files", unit="file") as pbar, \
         open(per_line_file, 'w', encoding='utf-8') as fout:

        fout.write("file_path\tline_number\ttoken_count\n")
        futures = {pool.submit(_process_file, p): p for p in txt_files}
        for fut in as_completed(futures):
            pbar.update()
            for file_path, lineno, count in fut.result():
                fout.write(f"{file_path[:20]}\t{lineno}\t{count}\n")
                all_counts.append(count)

    # Compute statistics
    counts = np.array(all_counts)
    lines_total = len(counts)
    lines_le_512 = np.sum(counts <= 512)
    percentile_512 = int(100 * lines_le_512 / lines_total)
    p99 = float(np.percentile(counts, 99))

    with open(stats_file, 'w') as sf:
        sf.write(f"Total lines: {lines_total}\n")
        sf.write(f"Lines with <=512 tokens: {lines_le_512} ({percentile_512}th percentile)\n")
        sf.write(f"99th percentile token count: {p99:.2f}\n")

    # Plot CCDF
    sorted_counts = np.sort(counts)
    ccdf = 1.0 - np.arange(1, lines_total + 1) / lines_total

    plt.figure()
    plt.loglog(sorted_counts, ccdf)
    plt.xlabel('Token count')
    plt.ylabel('CCDF')
    plt.title('CCDF of Mamba token counts per line')
    for fmt in ['svg', 'pdf', 'jpg']:
        plt.savefig(OUT_DIR / f'ccdf_static.{fmt}', dpi=300)
    plt.close()

    # Interactive plot
    df = {'token_count': sorted_counts, 'ccdf': ccdf}
    fig = px.line(df, x='token_count', y='ccdf', title='CCDF of Mamba Token Counts')
    fig.update_xaxes(type='log')
    fig.update_yaxes(type='log')
    fig.write_html(OUT_DIR / 'ccdf_interactive.html')

    print(f"Results written to {OUT_DIR.resolve()}")

# ───── RUN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
