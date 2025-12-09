#!/usr/bin/env python3
# rf_baseline_vs_finetuned_plot.py
"""
Plot baseline vs finetuned Macro F1 across RF_TRAIN_PCT sweep steps.

Usage:
    python rf_baseline_vs_finetuned_plot.py \
        --baseline rf_train_pct_results_baseline.csv \
        --finetuned rf_train_pct_results_finetuned.csv \
        --out rf_train_pct_comparison

This will produce:
  - rf_train_pct_comparison.png
  - rf_train_pct_comparison.pdf
  - rf_train_pct_comparison.svg

Y-axis is fixed to [0.60, 1.00]
X-axis is fixed to [20, 70]
"""

from __future__ import annotations
import csv
import argparse
import os
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_results_csv(path: str) -> Optional[Tuple[List[float], List[float]]]:
    """Load a CSV with columns rf_train_pct and macro_f1."""
    if not os.path.exists(path):
        print(f"[warn] file not found: {path}")
        return None

    pcts, f1s = [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"[error] no header found in CSV: {path}")
            return None

        pct_field, f1_field = None, None
        for name in reader.fieldnames:
            n = name.strip().lower()
            if n in ('rf_train_pct', 'rf_train_pct%', 'train_pct', 'pct', 'percent'):
                pct_field = name
            if n in ('macro_f1', 'f1', 'f1_macro', 'macro-f1'):
                f1_field = name

        if pct_field is None:
            pct_field = reader.fieldnames[0]
        if f1_field is None:
            if len(reader.fieldnames) >= 2:
                f1_field = reader.fieldnames[1]
            else:
                print(f"[error] couldn't determine macro_f1 column in {path}")
                return None

        for row in reader:
            try:
                raw_pct = float(row[pct_field])
                pct_percent = raw_pct * 100.0 if raw_pct <= 1.0 else raw_pct
                f1 = float(row[f1_field])
            except Exception as e:
                print(f"[warn] skipping malformed row in {path}: {row} ({e})")
                continue
            pcts.append(pct_percent)
            f1s.append(f1)

    if not pcts:
        print(f"[warn] no valid rows found in {path}")
        return None

    arr_idx = np.argsort(pcts)
    pcts_sorted = [float(np.array(pcts)[i]) for i in arr_idx]
    f1s_sorted = [float(np.array(f1s)[i]) for i in arr_idx]
    return pcts_sorted, f1s_sorted

def plot_comparison(baseline: Tuple[List[float], List[float]] | None,
                    finetuned: Tuple[List[float], List[float]] | None,
                    out_base: str):
    """Plot baseline and finetuned curves, fixed axes."""
    plt.figure(figsize=(9,6))

    plotted = 0
    if baseline is not None:
        pcts, f1s = baseline
        plt.plot(pcts, f1s, marker='o', linestyle='-', linewidth=2, markersize=6, label='Baseline', zorder=3)
        plotted += 1
    if finetuned is not None:
        pcts_ft, f1s_ft = finetuned
        plt.plot(pcts_ft, f1s_ft, marker='s', linestyle='--', linewidth=2, markersize=6, label='Finetuned', zorder=4)
        plotted += 1

    if plotted == 0:
        print("[error] nothing to plot. Exiting.")
        return

    plt.xlabel("RF_TRAIN_PCT (%)", fontsize=12)
    plt.ylabel("Macro F1 (validation)", fontsize=12)
    plt.title("Baseline vs Finetuned — Macro F1 across RF_TRAIN_PCT", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right', fontsize=11)
    plt.ylim(0.60, 1.00)
    plt.xlim(20, 70)  # fixed range as requested
    plt.xticks(np.arange(20, 71, 5))
    plt.tight_layout()

    # Save plots
    out_png, out_pdf, out_svg = f"{out_base}.png", f"{out_base}.pdf", f"{out_base}.svg"
    plt.savefig(out_png, dpi=300, transparent=True)
    plt.savefig(out_pdf, dpi=300, transparent=True)
    plt.savefig(out_svg, dpi=300, transparent=True)
    print(f"[info] saved: {out_png}, {out_pdf}, {out_svg}")
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Compare baseline vs finetuned Macro F1 sweep CSVs")
    p.add_argument("--baseline", "-b", type=str, default="rf_train_pct_results_baseline.csv",
                   help="path to baseline CSV (default rf_train_pct_results_baseline.csv)")
    p.add_argument("--finetuned", "-f", type=str, default="rf_train_pct_results_finetuned.csv",
                   help="path to finetuned CSV (default rf_train_pct_results_finetuned.csv)")
    p.add_argument("--out", "-o", type=str, default="rf_train_pct_comparison",
                   help="output base filename (no extension). default: rf_train_pct_comparison")
    args = p.parse_args()

    baseline_data = load_results_csv(args.baseline)
    finetuned_data = load_results_csv(args.finetuned)

    if baseline_data is None and finetuned_data is None:
        print("[error] neither baseline nor finetuned CSV could be loaded. Exiting.")
        sys.exit(2)

    plot_comparison(baseline_data, finetuned_data, args.out)

if __name__ == "__main__":
    main()
