#!/usr/bin/env python3
# new_pipeline2.py  (updated interleaving to avoid truncation; balanced round-robin)
import os
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import textwrap

# ---------------- Config (percentages) ----------------
device_list = [
    "irobot_roomba.json",
    "line_clova_wave.json",
    "nature_remo.json",
    "qrio_hub.json",
    "xiaomi_mijia_led.json"
]
INPUT_FOLDER = "preprocessed_data_merged/ungrouped"

# RF halves as explicit percentages of each device file (you can set these independently)
RF_TRAIN_PCT = 0.02   # percentage of file used for RF training (e.g. 0.10 = 10% of file)
RF_VAL_PCT   = 0.10   # percentage of file used for RF validation (e.g. 0.10 = 10% of file)

# Fine-tune and final validation percentages (fractions of file)
FINE_PCT  = 0.01     # middle block per device used for fine-tuning (interleaved)
FINAL_PCT = 0.01     # final block per device used for final validation

BERT_MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 2e-6
EPOCHS = 3

# Hyperparams for anchor/prototype losses and training
BATCH_SIZE = 16
LAMBDA_ANCHOR = 0.5
LAMBDA_PROTO = 0.5
RETRAIN_RF_AFTER_EPOCH = True

# ---------------- Helper Functions ----------------
def count_nonempty_lines(path):
    cnt = 0
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt

def load_flows(file_path, n_flows, start=0):
    """
    Read lines from file preserving order and return flows[start : start + n_flows].
    start is 0-based index.
    """
    flows = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            s = line.strip()
            if not s:
                continue
            flows.append(s)
            if len(flows) >= n_flows:
                break
    return flows

def get_bert_embedding(sentence, tokenizer, model, max_length=512):
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(sentence, return_tensors='pt', truncation=True,
                            max_length=max_length, padding='max_length').to(DEVICE)
        outputs = model(**encoded)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def short(s, width=120):
    return textwrap.shorten(s.replace("\n", "\\n"), width=width, placeholder="…")

# ---------------- Initialize BERT ----------------
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
device_label_map = {dev: idx for idx, dev in enumerate(device_list)}

# ---------------- Print configuration summary ----------------
print("Using the following percentages (per-device):")
print(f"  RF_TRAIN_PCT = {RF_TRAIN_PCT} ({RF_TRAIN_PCT*100:.1f}%)")
print(f"  RF_VAL_PCT   = {RF_VAL_PCT} ({RF_VAL_PCT*100:.1f}%)")
print(f"  FINE_PCT     = {FINE_PCT}  ({FINE_PCT*100:.1f}%)")
print(f"  FINAL_PCT    = {FINAL_PCT} ({FINAL_PCT*100:.1f}%)\n")

# ---------------- Prepare per-device counts and splits ----------------
splits_per_device = {}

for device in device_list:
    file_name = device.replace('.json', '.json.txt')
    file_path = os.path.join(INPUT_FOLDER, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    total_lines = count_nonempty_lines(file_path)
    if total_lines == 0:
        raise ValueError(f"No non-empty lines in {file_path}")

    # compute counts directly from percentages
    rf_train_count = int(total_lines * RF_TRAIN_PCT)
    rf_val_count   = int(total_lines * RF_VAL_PCT)
    fine_count     = int(total_lines * FINE_PCT)
    final_count    = int(total_lines * FINAL_PCT)

    # ensure at least 1 line where appropriate
    rf_train_count = max(1, rf_train_count)
    rf_val_count   = max(1, rf_val_count)

    # total required from top-of-file through final block
    required = rf_train_count + rf_val_count + fine_count + final_count
    if required > total_lines:
        # trimming policy: trim final_count first, then fine_count if needed
        print(f"Warning: device {device} requires {required} lines but file has {total_lines}. Trimming to fit.")
        space_left = total_lines - (rf_train_count + rf_val_count + fine_count)
        if space_left < 0:
            final_count = 0
            fine_count = max(0, total_lines - (rf_train_count + rf_val_count))
        else:
            final_count = space_left

    # compute 0-based start indices
    start_rf_train = 0
    start_rf_val   = start_rf_train + rf_train_count
    start_fine     = start_rf_val + rf_val_count
    start_final    = start_fine + fine_count

    # load the splits preserving order
    rf_train_half = load_flows(file_path, rf_train_count, start=start_rf_train)
    rf_val_half   = load_flows(file_path, rf_val_count, start=start_rf_val)
    fine_block    = load_flows(file_path, fine_count, start=start_fine)
    final_block   = load_flows(file_path, final_count, start=start_final)

    # basic disjointness sanity check
    set_train = set(rf_train_half)
    set_val   = set(rf_val_half)
    set_fine  = set(fine_block)
    set_final = set(final_block)
    if set_train & set_val:
        raise AssertionError(f"Overlap between RF train and RF val for {device}")
    if set_train & set_fine:
        raise AssertionError(f"Overlap between RF train and fine for {device}")
    if set_train & set_final:
        raise AssertionError(f"Overlap between RF train and final for {device}")
    if set_val & set_fine:
        raise AssertionError(f"Overlap between RF val and fine for {device}")
    if set_val & set_final:
        raise AssertionError(f"Overlap between RF val and final for {device}")
    if set_fine & set_final:
        raise AssertionError(f"Overlap between fine and final for {device}")

    # store info (including human-friendly 1-based ranges)
    splits_per_device[device] = {
        "file_path": file_path,
        "total_lines": total_lines,
        "rf_train_count": rf_train_count,
        "rf_val_count": rf_val_count,
        "fine_count": fine_count,
        "final_count": final_count,
        "rf_train": rf_train_half,
        "rf_val": rf_val_half,
        "fine_block": fine_block,
        "final_block": final_block,
        "rf_train_lines_1b": (start_rf_train + 1, start_rf_train + rf_train_count),
        "rf_val_lines_1b": (start_rf_val + 1, start_rf_val + rf_val_count),
        "fine_lines_1b": (start_fine + 1, start_fine + fine_count),
        "final_lines_1b": (start_final + 1, start_final + final_count)
    }

    # print summary
    print(f"Device: {device}")
    print(f"  File: {file_path}")
    print(f"  Total non-empty lines: {total_lines}")
    print(f"  RF TRAIN  (1-based lines): {splits_per_device[device]['rf_train_lines_1b']}  count={rf_train_count}")
    if rf_train_count > 0:
        print(f"    sample first: {short(rf_train_half[0])}")
        print(f"    sample last:  {short(rf_train_half[-1])}")
    print(f"  RF VAL    (1-based lines): {splits_per_device[device]['rf_val_lines_1b']}  count={rf_val_count}")
    if rf_val_count > 0:
        print(f"    sample first: {short(rf_val_half[0])}")
        print(f"    sample last:  {short(rf_val_half[-1])}")
    print(f"  FINE-TUNE (1-based lines): {splits_per_device[device]['fine_lines_1b']}  count={fine_count}")
    if fine_count > 0:
        print(f"    sample first: {short(fine_block[0])}")
        print(f"    sample last:  {short(fine_block[-1])}")
    print(f"  FINAL-VAL (1-based lines): {splits_per_device[device]['final_lines_1b']}  count={final_count}")
    if final_count > 0:
        print(f"    sample first: {short(final_block[0])}")
        print(f"    sample last:  {short(final_block[-1])}")
    print("")

print("All device splits prepared and verified.\n")

# ---------------- Build interleaved sequences (balanced, non-truncating round-robin) ----------------
# Create per-device queues for rf_train, rf_val, fine
def interleave_round_robin(per_device_lists, devices_order):
    """
    Given dict device -> list, produce a single list interleaving items in a round-robin
    fashion without truncation: cycle through devices and append next item if available.
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
        # loop continues until all lists exhausted
    return combined, labels

# Build dicts for each split type
per_device_rf_train = {d: splits_per_device[d]["rf_train"] for d in device_list}
per_device_rf_val   = {d: splits_per_device[d]["rf_val"]   for d in device_list}
per_device_fine     = {d: splits_per_device[d]["fine_block"] for d in device_list}

# Interleave them without truncation; preserves order within each device's block
rf_train_texts, rf_train_labels = interleave_round_robin(per_device_rf_train, device_list)
rf_val_texts,   rf_val_labels   = interleave_round_robin(per_device_rf_val, device_list)
fine_texts,     fine_labels     = interleave_round_robin(per_device_fine, device_list)

# Print per-device counts and combined totals
print("Per-device counts (rf_train, rf_val, fine, final):")
total_rf_train = total_rf_val = total_fine = total_final = 0
for d in device_list:
    a = splits_per_device[d]["rf_train_count"]
    b = splits_per_device[d]["rf_val_count"]
    c = splits_per_device[d]["fine_count"]
    e = splits_per_device[d]["final_count"]
    total_rf_train += a
    total_rf_val += b
    total_fine += c
    total_final += e
    print(f"  {d}: rf_train={a}, rf_val={b}, fine={c}, final={e}")
print(f"Combined totals (kept all samples): rf_train={len(rf_train_texts)} (sum per-device={total_rf_train}), rf_val={len(rf_val_texts)} (sum={total_rf_val}), fine={len(fine_texts)} (sum={total_fine}), final blocks sum={total_final}\n")

# ---------------- Step 1: Train RF on rf_train_texts ----------------
print("Embedding flows for RF training (RF-training interleaved across devices)...")
X_train = []
y_train = []
for text, lbl in tqdm(zip(rf_train_texts, rf_train_labels), total=len(rf_train_texts), desc="RF train embeddings"):
    emb = get_bert_embedding(text, tokenizer, bert_model)
    X_train.append(emb)
    y_train.append(lbl)
X_train = np.vstack(X_train)
y_train = np.array(y_train)

print("\nTraining Random Forest classifier on RF-train set...")
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)

# ---------------- RF Validation on rf_val_texts (pre-finetune) ----------------
print("\nValidating RF on RF-validation set (pre-finetune)...")
y_true_preval = []
y_pred_preval = []
for text, lbl in tqdm(zip(rf_val_texts, rf_val_labels), total=len(rf_val_texts), desc="RF pre-finetune validation"):
    emb = get_bert_embedding(text, tokenizer, bert_model)
    pred = rf_clf.predict(emb)[0]
    y_pred_preval.append(int(pred))
    y_true_preval.append(lbl)

cm_rf_pre = confusion_matrix(y_true_preval, y_pred_preval)
f1_macro_rf_pre = f1_score(y_true_preval, y_pred_preval, average='macro')
print("\nRF Confusion Matrix (before BERT fine-tuning) on RF-val set:")
print(cm_rf_pre)
print(f"Macro F1: {f1_macro_rf_pre:.4f}\n")

# ===== Fine-tune BERT (anchor + prototype) =====
orig_bert = bert_model
orig_bert.eval()
for p in orig_bert.parameters():
    p.requires_grad = False

print("Computing original CLS embeddings and class centroids for fine-tune set...")
orig_cls_list = []
for i in range(0, len(fine_texts), BATCH_SIZE):
    batch_texts = fine_texts[i:i+BATCH_SIZE]
    enc = tokenizer(batch_texts, return_tensors='pt', truncation=True, max_length=512, padding=True).to(DEVICE)
    with torch.no_grad():
        out = orig_bert(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
    orig_cls_list.append(cls)
orig_cls_all = np.vstack(orig_cls_list)
labels_arr = np.array(fine_labels)

num_classes = len(device_list)
centroids = np.zeros((num_classes, orig_cls_all.shape[1]), dtype=np.float32)
for c in range(num_classes):
    idxs = np.where(labels_arr == c)[0]
    if len(idxs) == 0:
        continue
    centroids[c] = orig_cls_all[idxs].mean(axis=0)
print("Centroids computed.\n")

class FTData(Dataset):
    def __init__(self, texts, labels, orig_embs, centroids):
        self.texts = texts
        self.labels = labels
        self.orig_embs = orig_embs.astype(np.float32)
        self.centroids = centroids
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        label = self.labels[idx]
        return self.texts[idx], int(label), self.orig_embs[idx], self.centroids[label]

dataset = FTData(fine_texts, fine_labels, orig_cls_all, centroids)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

student_bert = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
mse_loss = nn.MSELoss()
optimizer = torch.optim.AdamW(student_bert.parameters(), lr=LR)

print("\nStarting BERT-only fine-tune (anchor + prototype)...")
for epoch in range(EPOCHS):
    student_bert.train()
    running_loss = 0.0
    steps = 0
    last_anchor = 0.0
    last_proto = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        texts_b, labels_b, orig_emb_b, proto_emb_b = batch
        orig_emb_b = orig_emb_b.to(DEVICE)
        proto_emb_b = proto_emb_b.to(DEVICE)

        enc = tokenizer(list(texts_b), return_tensors='pt', truncation=True, max_length=512, padding=True).to(DEVICE)
        optimizer.zero_grad()
        out = student_bert(**enc)
        cls_emb = out.last_hidden_state[:, 0, :]

        loss_anchor = mse_loss(cls_emb, orig_emb_b)
        loss_proto = mse_loss(cls_emb, proto_emb_b)
        loss = LAMBDA_ANCHOR * loss_anchor + LAMBDA_PROTO * loss_proto
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        steps += 1
        last_anchor = loss_anchor.item()
        last_proto = loss_proto.item()

    avg_loss = running_loss / max(1, steps)
    print(f"Epoch {epoch+1}/{EPOCHS} - avg loss: {avg_loss:.6f} (anchor {last_anchor:.6f}, proto {last_proto:.6f})")

    if RETRAIN_RF_AFTER_EPOCH:
        print("Recomputing embeddings for RF retrain (RF-training set interleaved across devices)...")
        X_new = []
        y_new = []
        for text, lbl in zip(rf_train_texts, rf_train_labels):
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length').to(DEVICE)
            student_bert.eval()
            with torch.no_grad():
                out = student_bert(**enc)
                cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            X_new.append(cls)
            y_new.append(int(lbl))
        X_new = np.vstack(X_new)
        y_new = np.array(y_new)
        rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
        rf_clf.fit(X_new, y_new)
        print("RF retrained on student embeddings (RF-training set).")

    # Evaluate RF on final blocks (per-device)
    y_true_val_epoch = []
    y_pred_val_epoch = []
    student_bert.eval()
    for device_name in device_list:
        for f in splits_per_device[device_name]["final_block"]:
            enc = tokenizer(f, return_tensors='pt', truncation=True, max_length=512, padding='max_length').to(DEVICE)
            with torch.no_grad():
                out = student_bert(**enc)
                cls = out.last_hidden_state[:, 0, :]
                rf_pred = rf_clf.predict(cls.cpu().numpy())[0]
            y_pred_val_epoch.append(int(rf_pred))
            y_true_val_epoch.append(device_label_map[device_name])
    cm_epoch = confusion_matrix(y_true_val_epoch, y_pred_val_epoch)
    f1m_epoch = f1_score(y_true_val_epoch, y_pred_val_epoch, average='macro')
    print(f"After epoch {epoch+1} RF-on-student embeddings Macro F1 (final-val) = {f1m_epoch:.4f}")
    print(cm_epoch)
    print("")

# ---------------- Final validation ----------------
y_true_val = []
y_pred_val = []
print("\nValidating RF + fine-tuned BERT on flows (FINAL)...")
for device in tqdm(device_list, desc="Validation Devices"):
    for flow in splits_per_device[device]["final_block"]:
        encoded = tokenizer(flow, return_tensors='pt', truncation=True, max_length=512, padding='max_length').to(DEVICE)
        student_bert.eval()
        with torch.no_grad():
            out = student_bert(**encoded)
            cls_emb = out.last_hidden_state[:, 0, :]
            rf_pred = rf_clf.predict(cls_emb.cpu().numpy())[0]
        y_pred_val.append(int(rf_pred))
        y_true_val.append(device_label_map[device])

assert len(y_true_val) == len(y_pred_val) and len(y_true_val) > 0, "No validation predictions collected!"
cm_final = confusion_matrix(y_true_val, y_pred_val)
f1_macro_final = f1_score(y_true_val, y_pred_val, average='macro')

print("\nConfusion Matrix AFTER BERT fine-tuning (final):")
print(cm_final)
print(f"Macro F1 AFTER fine-tuning: {f1_macro_final:.4f}")
