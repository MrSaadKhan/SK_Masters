#!/usr/bin/env python3
"""
bert_token_counter_json.py
--------------------------
Walk through all *.json files in TXT_DIR (recursively) and, once per second,
print how many BERT tokens each non‑empty line contains.

Dependencies:  pip install transformers  (and torch)
"""

import sys, time
from pathlib import Path
from transformers import AutoTokenizer

# ======== CONFIGURE ME =========
TXT_DIR    = Path("/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix")
MODEL_NAME = "bert-base-uncased"
DELAY      = 0.2          # seconds between prints
# ===============================

print(f"[DEBUG] Looking in: {TXT_DIR}")
if not TXT_DIR.exists():
    sys.exit("[ERROR] That path does not exist.")
if not TXT_DIR.is_dir():
    sys.exit("[ERROR] That path is not a directory.")

# search for .json files (recursively)
json_files = list(TXT_DIR.rglob("*.json"))
print(f"[DEBUG] Found {len(json_files)} .json file(s).")
print("-------------------------------------------------------------")

if not json_files:
    sys.exit("[ERROR] No .json files found – nothing to do.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for json_file in sorted(json_files):
    rel_path = json_file.relative_to(TXT_DIR)
    print(f"Processing {rel_path} …")
    with json_file.open("r", encoding="utf-8", errors="ignore") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            n_tokens = len(tokenizer.encode(line, add_special_tokens=False))
            print(f"{rel_path}:{line_no:>5} – {n_tokens} tokens")
            time.sleep(DELAY)     # throttle CPU usage
