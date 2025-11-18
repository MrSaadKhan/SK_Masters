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
import matplotlib.ticker as ticker


# ───── CONFIGURATION ─────────────────────────────────────────────────────────
# Directory containing your .txt files:
DIR_WITH_TXTS = Path("/home/iotresearch/saad/masters/preprocessed_data_single/ungrouped")
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
    global _TOKENIZER
    results = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, start=1):
            text = line.rstrip("\n")
            count = len(_TOKENIZER.encode(text, add_special_tokens=False))
            results.append((str(path), lineno, count))
    return results

def _read_counts_from_file(per_line_file: Path):
    """
    Read token counts from an existing per-line TSV file.
    Expected format: header then lines: file_path \t line_number \t token_count
    Returns list of token counts (ints).
    """
    counts = []
    if not per_line_file.exists():
        return counts

    with per_line_file.open('r', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                print(f"[Read] Skipping malformed line {lineno} in {per_line_file}")
                continue
            try:
                token_count = int(parts[2])
            except ValueError:
                print(f"[Read] Unable to parse token count on line {lineno}: {parts[2]!r}")
                continue
            counts.append(token_count)
    return counts

# ───── MAIN FUNCTION ──────────────────────────────────────────────────────────
def main():
    print("[Main] Starting compute_mamba_token_stats.py")
    if not DIR_WITH_TXTS.is_dir():
        sys.exit(f"[Error] Directory not found: {DIR_WITH_TXTS}")

    txt_files = list(DIR_WITH_TXTS.rglob("*.txt"))
    if not txt_files:
        sys.exit(f"[Error] No .txt files found in {DIR_WITH_TXTS}")
    print(f"[Main] Found {len(txt_files)} .txt files under {DIR_WITH_TXTS}")

    # Prepare output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_line_file = OUT_DIR / "line_token_counts.txt"
    stats_file = OUT_DIR / "token_stats.txt"

    all_counts = []

    if per_line_file.exists():
        print(f"[Main] Found existing {per_line_file}. Skipping re-tokenization and reading counts from it.")
        all_counts = _read_counts_from_file(per_line_file)
        print(f"[Main] Read {len(all_counts)} token counts from {per_line_file}")
    else:
        # Tokenize files in parallel and write per-line file
        print("[Main] No existing per-line counts found. Tokenizing files now...")
        with ProcessPoolExecutor(max_workers=WORKERS,
                                 initializer=_init_worker,
                                 initargs=(MODEL_NAME,)) as pool, \
             tqdm(total=len(txt_files), desc="Files", unit="file") as pbar, \
             open(per_line_file, 'w', encoding='utf-8') as fout:

            fout.write("file_path\tline_number\ttoken_count\n")
            futures = {pool.submit(_process_file, p): p for p in txt_files}
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    # If a worker fails, raise to surface the error
                    print(f"[Main] ERROR: worker failed for {src}: {e}")
                    raise
                for file_path, lineno, count in result:
                    fout.write(f"{file_path}\t{lineno}\t{count}\n")
                    all_counts.append(count)
                pbar.update()

        print(f"[Main] Tokenization complete. Wrote per-line counts to {per_line_file}")

    # If we have no counts at this point, abort
    if not all_counts:
        print("[Main] No token counts available to compute statistics. Exiting.")
        return

    # Compute statistics and always (re)write token_stats.txt
    counts = np.array(all_counts)
    lines_total = len(counts)
    lines_le_512 = int(np.sum(counts <= 512))
    percentile_512 = int(100 * lines_le_512 / lines_total)
    p99 = float(np.percentile(counts, 99))
    average_token_size = np.mean(counts)

    print(f"[Stats] Total lines: {lines_total}")
    print(f"[Stats] Lines with <=512 tokens: {lines_le_512} ({percentile_512}th percentile)")
    print(f"[Stats] 99th percentile token count: {p99:.2f}")

    with open(stats_file, 'w', encoding='utf-8') as sf:
        sf.write(f"Total lines: {lines_total}\n")
        sf.write(f"Lines with <=512 tokens: {lines_le_512} ({percentile_512}th percentile)\n")
        sf.write(f"99th percentile token count: {p99:.2f}\n")
        sf.write(f"\nAverage token size: {average_token_size:.2f}\n")
    print(f"[Main] Wrote summary stats to {stats_file}")

    # Plot CCDF (always regenerate)
    sorted_counts = np.sort(counts)
    ccdf = 1.0 - np.arange(1, lines_total + 1) / lines_total

    plt.figure()
    plt.loglog(sorted_counts, ccdf)
    plt.xlabel('Token count')
    plt.ylabel('CCDF')
    plt.title('CCDF of Mamba token counts per line')

    #######

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')


    ###########
    for fmt in ['svg', 'pdf', 'jpg']:
        out_path = OUT_DIR / f'ccdf_static.{fmt}'
        plt.savefig(out_path, dpi=300)
        print(f"[Plot] Saved static plot: {out_path}")
    plt.close()

    # Interactive plot (regenerate)
    df = {'token_count': sorted_counts, 'ccdf': ccdf}
    fig = px.line(df, x='token_count', y='ccdf', title='CCDF of Mamba Token Counts')
    fig.update_xaxes(type='log')
    fig.update_yaxes(type='log')
    interactive_path = OUT_DIR / 'ccdf_interactive.html'
    fig.write_html(interactive_path)
    print(f"[Plot] Saved interactive plot: {interactive_path}")

    print(f"[Main] All done. Results and plots written to {OUT_DIR.resolve()}")

# ───── RUN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
