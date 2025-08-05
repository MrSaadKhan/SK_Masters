#!/usr/bin/env python3
"""
compute_token_stats_dual.py
───────────────────────────────────────────────────────────────────────────────
Compute token counts for **two** models (e.g., Mamba and BERT) across **two** corpora,
write per‑line & summary stats, then plot their CCDFs together (static + interactive).

If `line_token_counts_<label>.txt` exists, the script reloads counts from it; otherwise it
recomputes tokens. Stats (average, percentiles) are recalculated and `token_stats_<label>.txt`
is always overwritten.

All paths are set via variables below—no CLI args needed.

Outputs land in `BASE_OUT_DIR/<model_slug>/`:
  • line_token_counts_<label>.txt  ── per‑line counts
  • token_stats_<label>.txt        ── summary statistics (includes avg, model name)
  • ccdf_static.{svg,pdf,jpg}      ── 300 dpi static plot (both corpora)
  • ccdf_interactive.html          ── interactive Plotly version

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
    "group":  Path("/home/iotresearch/saad/masters/preprocessed_data_group/ungrouped"),
    "single": Path("/home/iotresearch/saad/masters/preprocessed_data/ungrouped"),
}
WORKERS = None                # None → use CPU count
MODELS = [
    "state-spaces/mamba-130m-hf",
    "google-bert/bert-base-uncased",
]
BASE_OUT_DIR = Path.cwd() / "token_output"

# ───── Worker Init ───────────────────────────────────────────────────────────
_TOKENIZER = None

def _init_worker(model_name: str):
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    print(f"[Worker] Loaded tokenizer for model {model_name}")

# ───── File Processing ────────────────────────────────────────────────────────

def _process_file(path: Path) -> List[int]:
    """Return a list of token counts for every line in the file."""
    global _TOKENIZER
    counts: List[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.rstrip("\n")
            n_tok = len(_TOKENIZER.encode(text, add_special_tokens=False))
            counts.append(n_tok)
    return counts

# ───── Corpus & Model Processing ──────────────────────────────────────────────

def collect_counts(label: str, root: Path, model_name: str, out_dir: Path) -> List[int]:
    if not root.is_dir(): sys.exit(f"Directory not found: {root}")
    txt_files = list(root.rglob("*.txt"))
    if not txt_files: sys.exit(f"No .txt files found in {root}")

    model_slug = model_name.split('/')[-1]
    model_out = out_dir / model_slug
    model_out.mkdir(parents=True, exist_ok=True)
    line_out = model_out / f"line_token_counts_{label}.txt"
    stats_out = model_out / f"token_stats_{label}.txt"

    # LOAD or COMPUTE counts
    if line_out.exists():
        print(f"Loading existing token counts for {model_slug}:{label}")
        df = pd.read_csv(line_out, sep='\t')
        counts = df['token_count'].tolist()
    else:
        counts = []
        with ProcessPoolExecutor(max_workers=WORKERS,
                                 initializer=_init_worker,
                                 initargs=(model_name,)) as pool, \
             tqdm(total=len(txt_files), desc=f"{model_slug}:{label}", unit="file") as pbar, \
             open(line_out, "w", encoding="utf-8") as fout:
            fout.write("file_path\tline_number\ttoken_count\n")
            futures = {pool.submit(_process_file, p): p for p in txt_files}
            for fut in as_completed(futures):
                pbar.update()
                fp = futures[fut]
                for tok in fut.result():
                    fout.write(f"{fp}\t-\t{tok}\n")
                    counts.append(tok)

    # Always recompute stats
    arr = np.array(counts)
    total = len(arr)
    avg = float(np.mean(arr))
    pct_512 = 100 * np.sum(arr <= 512) / total
    p99 = np.percentile(arr, 99)
    with open(stats_out, "w") as sf:
        sf.write(f"Model                  : {model_name}\n")
        sf.write(f"Corpus label           : {label}\n")
        sf.write(f"Total lines            : {total}\n")
        sf.write(f"Average tokens/line    : {avg:.2f}\n")
        sf.write(f"Lines ≤512 tokens      : {np.sum(arr <= 512)} ({pct_512:.2f}th pct)\n")
        sf.write(f"99th‑percentile tokens : {p99:.2f}\n")
    return counts

# ───── Plotting ─────────────────────────────────────────────────────────────

def plot_ccdf(counts_map: Dict[str, List[int]], out_dir: Path):
    fig, ax = plt.subplots()
    for key, counts in counts_map.items():
        s = np.sort(counts)
        ccdf = 1 - np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, ccdf, label=key)

    ax.set_xlabel("Token count")
    ax.set_ylabel("CCDF")
    ax.set_title("CCDF of token counts per line")
    ax.legend()
    ax.get_xaxis().set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.get_yaxis().set_major_formatter(StrMethodFormatter("{x:.2f}"))

    for ext in ["svg", "pdf", "jpg"]:
        fig.savefig(out_dir / f"ccdf_static.{ext}" , dpi=300)
    plt.close(fig)

    # Interactive
    dfs = []
    for key, counts in counts_map.items():
        s = np.sort(counts)
        ccdf = 1 - np.arange(1, len(s) + 1) / len(s)
        percentiles = np.arange(1, len(s) + 1) / len(s) * 100
        dfs.append(pd.DataFrame({
            "token_count": s,
            "ccdf": ccdf,
            "percentile": percentiles,
            "model": key.split(':')[0],
            "corpus": key.split(':')[1],
        }))
    df = pd.concat(dfs, ignore_index=True)
    fig_int = px.line(
        df,
        x="token_count",
        y="ccdf",
        color="model",
        line_dash="corpus",
        hover_data={"percentile": ':.2f'},
        labels={"ccdf": "CCDF", "token_count": "Token count", "percentile": "Percentile (%)"},
        title="CCDF of token counts per line"
    )
    fig_int.update_xaxes(type="log")
    fig_int.update_yaxes(type="log")
    fig_int.update_traces(hovertemplate=
        "Model: %{legendgroup}<br>Corpus: %{customdata[1]}<br>Token count: %{x}<br>Percentile: %{customdata[0]:.2f}%<extra></extra>")

    fig_int.write_html(out_dir / "ccdf_interactive.html")

# ───── Main ───────────────────────────────────────────────────────────────── ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def main():
    BASE_OUT_DIR.mkdir(exist_ok=True)
    for model in MODELS:
        overall_counts: Dict[str, List[int]] = {}
        for label, path in DIRS.items():
            key = f"{model.split('/')[-1]}:{label}"
            counts = collect_counts(label, path, model, BASE_OUT_DIR)
            overall_counts[key] = counts
        plot_ccdf(overall_counts, BASE_OUT_DIR / model.split('/')[-1])
    print(f"All outputs written under: {BASE_OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
