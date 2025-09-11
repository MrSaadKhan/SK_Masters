#!/usr/bin/env python3
"""
Script to parse RF Macro F1 scores from an input log file, print them, plot with Matplotlib, and save the figure.

Usage:
    python parse_scores.py [input_file] [output_image]

It strictly matches these lines (in order):
  option is: <option_name>
  Training RF Classifier using the options: use_percentage_split: <True|False>, train_size: <int>
  RF Macro F1 Score: <float>

Generates a line plot of RF Macro F1 Score vs. train_size for each option and saves to the specified image file.
"""
import re
import sys
import matplotlib.pyplot as plt


def parse_file(input_path):
    records = []

    # Strict regex patterns
    option_pattern = re.compile(r"^option is:\s*(?P<option>\w+)$")
    training_pattern = re.compile(
        r"^Training RF Classifier using the options: use_percentage_split:\s*(?P<use_pct>True|False),\s*train_size:\s*(?P<train_size>\d+)$"
    )
    score_pattern = re.compile(r"^RF Macro F1 Score:\s*(?P<score>\d+\.\d+)$")
    # score_pattern = re.compile(r"^RF Weighted F1 Score:\s*(?P<score>\d+\.\d+)$")

    current_option = None
    current_train_size = None

    with open(input_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()

            m = option_pattern.match(line)
            if m:
                current_option = m.group('option')
                continue

            m = training_pattern.match(line)
            if m:
                current_train_size = int(m.group('train_size'))
                continue

            m = score_pattern.match(line)
            if m:
                score = float(m.group('score'))
                if current_option and current_train_size is not None:
                    records.append((current_option, current_train_size, score))
                current_option = None
                current_train_size = None

    return records


def plot_records(records, output_image):
    # Organize by option
    data = {}
    for option, size, score in records:
        data.setdefault(option, []).append((size, score))

    plt.figure()
    for option, vals in data.items():
        vals_sorted = sorted(vals, key=lambda x: x[0])
        xs, ys = zip(*vals_sorted)
        plt.plot(xs, ys, marker='o', label=option)

    plt.xlabel('Days of training (train_size)')
    plt.ylabel('RF Macro F1 Score')
    plt.title('RF Macro F1 Score vs. Training Days')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'output3_2-05.txt'
    output_image = sys.argv[2] if len(sys.argv) > 2 else 'scores_plot2.png'

    records = parse_file(input_file)
    if not records:
        print("No records found. Check your input file and regex patterns.")
        sys.exit(1)

    # Print records
    print("Parsed records:")
    for option, size, score in records:
        print(f"{option}, {size}, {score}")

    plot_records(records, output_image)
