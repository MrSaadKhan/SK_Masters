
# DIR_WITH_TXTS = Path("/home/iotresearch/saad/masters/preprocessed_data/ungrouped")  # ← change this or supply a directory on the CLI
#!/usr/bin/env python3
"""
find_longest_mamba_line_mp.py
------------------------------------------------
Find the line with the most Mamba tokens in a directory
of .txt files, using multiprocessing for speed and a
progress bar for visibility.

Run:
    python find_longest_mamba_line_mp.py  /path/to/txts  [--workers N]

Dependencies:
    pip install transformers tqdm
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_NAME = "state-spaces/mamba-130m-hf"


# --------------------------------------------------------------------------- #
# Worker-side initialisation                                                  #
# --------------------------------------------------------------------------- #
_TOKENIZER = None  # will be initialised in each worker


def _init_worker(model_name: str) -> None:  # runs *inside* each worker
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------------------------------- #
# Worker-side task                                                            #
# --------------------------------------------------------------------------- #
def _process_file(path: Path) -> Tuple[int, str, int, str]:
    """
    Read `path`, find the line with the most tokens (Mamba tokenizer).

    Returns:
        (max_tokens, str(path), line_number, line_text)
    """
    global _TOKENIZER
    max_tokens, max_lineno, max_line = -1, None, None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.rstrip("\n")
            n_tok = len(_TOKENIZER.encode(stripped, add_special_tokens=False))
            if n_tok > max_tokens:
                max_tokens, max_lineno, max_line = n_tok, lineno, stripped

    return max_tokens, str(path), max_lineno, max_line


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(directory: Path, workers: int | None = None) -> None:
    txt_files = list(directory.rglob("*.txt"))
    if not txt_files:
        sys.exit(f"No .txt files found in {directory.resolve()}")

    print(f"Scanning {len(txt_files)} files with up to "
          f"{workers or os.cpu_count()} worker processes …")

    global_max = (-1, "", -1, "")  # (tokens, file, line_no, line_text)

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(MODEL_NAME,)) as pool, \
         tqdm(total=len(txt_files), desc="Files", unit="file") as pbar:

        futures = {pool.submit(_process_file, p): p for p in txt_files}

        for fut in as_completed(futures):
            pbar.update()
            tokens, path, lineno, text = fut.result()
            if tokens > global_max[0]:
                global_max = (tokens, path, lineno, text)

    # ---------------- Result ---------------- #
    tokens, path, lineno, text = global_max
    print("\n=== Longest line by Mamba token count ===")
    print(f"File        : {path}")
    print(f"Line number : {lineno}")
    print(f"Token count : {tokens}")
    print("Line preview:")
    print(text[:200] + ("…" if len(text) > 200 else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find longest-token line (Mamba) with multiprocessing.")
    parser.add_argument("directory", nargs="?", default="/home/iotresearch/saad/masters/preprocessed_data/ungrouped", help="Directory containing .txt files")
    parser.add_argument("--workers", type=int, help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()

    target_dir = Path(args.directory).expanduser()
    if not target_dir.is_dir():
        sys.exit(f"Directory not found: {target_dir}")

    # Required for Windows & macOS spawn-method safety
    main(target_dir, args.workers)
