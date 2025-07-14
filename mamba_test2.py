#!/usr/bin/env python3
"""
compute_mamba_token_stats.py
───────────────────────────────────────────────────────────────────────────────
Compute Mamba token counts for **two** corpora, write per‑line & summary stats,
then plot their CCDFs together (static + interactive).

All paths are set via variables below—no CLI args needed.

Outputs land in `OUT_DIR`:
  • line_token_counts_<label>.txt  ── per‑line counts
  • token_stats_<label>.txt        ── summary statistics
  • ccdf_static.{svg,pdf,jpg}      ── 300 dpi static plot (both corpora)
  • ccdf_interactive.html          ── interactive Plotly version (hover =>
                                     token‑count & percentile)

Dependencies
^^^^^^^^^^^^
    pip install transformers tqdm numpy matplotlib plotly pandas
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from tqdm import tqdm
from transformers import AutoTokenizer
import plotly.express as px

# ───── CONFIGURATION ─────────────────────────────────────────────────────────
DIRS: Dict[str, Path] = {
    "group":  Path("/home/iotresearch/saad/masters/preprocessed_data/ungrouped"),
    "single": Path("/home/iotresearch/saad/masters/preprocessed_data_single/ungrouped"),
}
OUT_DIR = Path.cwd() / "mamba_token_output"
WORKERS = None                # None → use CPU count
MODEL_NAME = "google-bert/bert-base-uncased"#"state-spaces/mamba-130m-hf"

# ───── Worker Initialisation ──────────────────────────────────────────────────
_TOKENIZER = None

def _init_worker(model_name: str):
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    print(f"[Worker] Loaded tokenizer class: {_TOKENIZER.__class__.__name__}")

# ───── Per‑file processing ───────────────────────────────────────────────────

def _process_file(path: Path):
    """Return a list of (filepath, line_no, token_count) for *every* line."""
    global _TOKENIZER
    res = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            text = line.rstrip("\n")
            n_tok = len(_TOKENIZER.encode(text, add_special_tokens=False))
            res.append((str(path), lineno, n_tok))
    return res

# ───── Corpus‑level processing ───────────────────────────────────────────────

def collect_counts(label: str, root: Path) -> List[int]:
    if not root.is_dir():
        sys.exit(f"Directory not found: {root}")

    txt_files = list(root.rglob("*.txt"))
    if not txt_files:
        sys.exit(f"No .txt files found in {root}")

    line_out = OUT_DIR / f"line_token_counts_{label}.txt"
    stats_out = OUT_DIR / f"token_stats_{label}.txt"

    counts: List[int] = []
    OUT_DIR.mkdir(exist_ok=True)

    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=_init_worker,
                             initargs=(MODEL_NAME,)) as pool, \
         tqdm(total=len(txt_files), desc=f"{label} files", unit="file") as pbar, \
         open(line_out, "w", encoding="utf-8") as fout:

        fout.write("file_path\tline_number\ttoken_count\n")
        futures = {pool.submit(_process_file, p): p for p in txt_files}
        for fut in as_completed(futures):
            pbar.update()
            for fp, ln, tok in fut.result():
                fout.write(f"{fp}\t{ln}\t{tok}\n")
                counts.append(tok)

    # ─── Stats ────────────────────────────────────────────────────────────
    arr = np.asarray(counts)
    total = len(arr)
    pct_512 = 100 * np.sum(arr <= 512) / total
    p99 = np.percentile(arr, 99)

    with open(stats_out, "w") as sf:
        sf.write(f"Corpus label           : {label}\n")
        sf.write(f"Total lines            : {total}\n")
        sf.write(f"Lines ≤512 tokens      : {np.sum(arr <= 512)} ({pct_512:.2f}th pct)\n")
        sf.write(f"99th‑percentile tokens : {p99:.2f}\n")

    return counts

# ───── Plotting ───────────────────────────────────────────────────────────────

def plot_ccdf(counts_map: Dict[str, List[int]]):
    # STATIC ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots()
    for label, counts in counts_map.items():
        s = np.sort(counts)
        ccdf = 1 - np.arange(1, len(s) + 1) / len(s)
        ax.loglog(s, ccdf, label=label)

    ax.set_xlabel("Token count")
    ax.set_ylabel("CCDF")
    ax.set_title("CCDF of Mamba token counts per line")
    ax.legend()

    # integer‑style tick labels instead of 1eX
    ax.get_xaxis().set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.get_yaxis().set_major_formatter(StrMethodFormatter("{x:.2f}"))

    for ext in ["svg", "pdf", "jpg"]:
        fig.savefig(OUT_DIR / f"ccdf_static.{ext}", dpi=300)
    plt.close(fig)

    # INTERACTIVE ─────────────────────────────────────────────────────────
    dfs = []
    for label, counts in counts_map.items():
        s = np.sort(counts)
        ccdf = 1 - np.arange(1, len(s) + 1) / len(s)
        percentiles = np.arange(1, len(s) + 1) / len(s) * 100
        dfs.append(pd.DataFrame({
            "token_count": s,
            "ccdf": ccdf,
            "percentile": percentiles,
            "dataset": label,
        }))

    df = pd.concat(dfs, ignore_index=True)
    fig_int = px.line(
        df,
        x="token_count",
        y="ccdf",
        color="dataset",
        hover_data={"percentile": ':.2f'},
        labels={"ccdf": "CCDF", "token_count": "Token count", "percentile": "Percentile (%)"},
        title="CCDF of Mamba token counts per line",
    )
    fig_int.update_xaxes(type="log")
    fig_int.update_yaxes(type="log")

    # custom hover: show token count (x) + percentile
    fig_int.update_traces(hovertemplate=
        "Dataset: %{legendgroup}<br>Token count: %{x}<br>Percentile: %{customdata[0]:.2f}%<extra></extra>")

    fig_int.write_html(OUT_DIR / "ccdf_interactive.html")

# ───── Main ──────────────────────────────────────────────────────────────────

def main():
    counts_map: Dict[str, List[int]] = {}
    for label, path in DIRS.items():
        counts_map[label] = collect_counts(label, path)

    plot_ccdf(counts_map)

    print(f"All outputs written to: {OUT_DIR.resolve()}")

# ───── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
