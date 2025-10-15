#!/usr/bin/env python3
"""
Modified version:
- Only consider train_size <= 70
- Divide train_size by 0.7 before plotting
- Fix y-axis (0.50–1.00) and x-axis (0–100)
"""

import re
import sys
import os
import matplotlib.pyplot as plt


def parse_file(input_path):
    records = []

    option_pattern = re.compile(r"^option is:\s*(?P<option>\w+)$")
    training_pattern = re.compile(
        r"^Training RF Classifier using the options: use_percentage_split:\s*(?P<use_pct>True|False),\s*train_size:\s*(?P<train_size>\d+)$"
    )
    score_pattern = re.compile(r"^RF Macro F1 Score:\s*(?P<score>\d+\.\d+)$")

    current_option = None
    current_train_size = None
    current_use_pct = None

    with open(input_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()

            m = option_pattern.match(line)
            if m:
                current_option = m.group('option')
                current_train_size = None
                current_use_pct = None
                continue

            m = training_pattern.match(line)
            if m:
                current_use_pct = True if m.group('use_pct') == 'True' else False
                current_train_size = int(m.group('train_size'))
                continue

            m = score_pattern.match(line)
            if m:
                score = float(m.group('score'))
                if (
                    current_option
                    and current_train_size is not None
                    and current_use_pct is not None
                    and current_train_size <= 70  # <-- only include up to 70
                ):
                    # divide train_size by 0.7
                    adjusted_size = current_train_size / 0.7
                    records.append((current_option, current_use_pct, adjusted_size, score))
                current_option = None
                current_train_size = None
                current_use_pct = None

    return records


def plot_records(records, output_base):
    base, ext = os.path.splitext(output_base)
    if not base:
        base = output_base
    true_path = f"{base}_true.svg"
    false_path = f"{base}_false.svg"

    by_use = {True: [], False: []}
    for option, use_pct, size, score in records:
        by_use[use_pct].append((option, size, score))

    for use_val in (True, False):
        recs = by_use[use_val]
        if not recs:
            print(f"No records for use_percentage_split={use_val}; skipping plot {use_val}.")
            continue

        data = {}
        for option, size, score in recs:
            data.setdefault(option, []).append((size, score))

        plt.figure()
        for option, vals in data.items():
            vals_sorted = sorted(vals, key=lambda x: x[0])
            xs, ys = zip(*vals_sorted)
            plt.plot(xs, ys, marker='o', label=option)

        plt.ylim(0.50, 1.00)   # <-- fixed y-axis
        plt.xlim(0, 100)       # <-- fixed x-axis

        if use_val:
            plt.xlabel('Training Percentage (train_size)')
            plt.title(f'RF Macro F1 Score vs. Training Percentage (use_percentage_split={use_val})')
        else:
            plt.xlabel('Days of training (train_size)')
            plt.title(f'RF Macro F1 Score vs. Training Days (use_percentage_split={use_val})')

        plt.ylabel('RF Macro F1 Score')
        plt.legend()
        plt.tight_layout()
        out_path = true_path if use_val else false_path
        plt.savefig(out_path, dpi=300, transparent=True)
        plt.close()
        print(f"Plot saved to {out_path}")


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'output3-05.txt'
    output_image = sys.argv[2] if len(sys.argv) > 2 else 'scores_plot-05_cooked.svg'

    # base_path = r"C:\Users\Saad Khan\OneDrive - UNSW\University\6th Yr\T3\Masters Project C\Results\EXP 1 - Iterative Classifier_Individual\Percentage-not random- freeze testing 30"

    # input_file = os.path.join(base_path, input_file)
    # output_image = os.path.join(base_path, output_image)

    records = parse_file(input_file)
    if not records:
        print("No records found. Check your input file and regex patterns.")
        sys.exit(1)

    print("Parsed records (option, use_percentage_split, train_size/0.7, score):")
    for option, use_pct, size, score in records:
        print(f"{option}, {use_pct}, {size:.2f}, {score}")

    plot_records(records, output_image)
