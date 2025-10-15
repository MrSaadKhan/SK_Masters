# -*- coding: utf-8 -*-
"""
Flow-level BERT classification pipeline with last-layer fine-tuning.
- Inputs: txt files in 'preprocessed_data_merged/ungrouped/'
- Each line in the txt = one flow sentence
- Train on first 500 flows per device using chronologically mixed batches
- Test on next 500 flows individually
- Classifier head: small MLP
- Outputs confusion matrix and macro F1
"""

import os
from tqdm import tqdm
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import f1_score, confusion_matrix

# ===========================
# CONFIGURABLE PARAMETERS
# ===========================
DATA_DIR = "preprocessed_data_merged/ungrouped/"
DEVICE_LIST = [
    "irobot_roomba.json.txt",
    "line_clova_wave.json.txt",
    "nature_remo.json.txt",
    "qrio_hub.json.txt",
    "xiaomi_mijia_led.json.txt"
]
TRAIN_FLOWS = 500
TEST_FLOWS = 500
PRETRAINED_MODEL = "bert-base-uncased"
BATCH_SIZE_PER_DEVICE = 16  # flows per device in each batch
EPOCHS = 3
LR = 1e-5  # slightly lower for fine-tuning
MAX_LEN = 128  # max tokens per flow sentence
HIDDEN_DIM = 256  # MLP hidden layer size

# ===========================
# LOAD DATA PER DEVICE
# ===========================
train_sentences_per_device = []
test_sentences, test_labels = [], []

for device_idx, device_file in enumerate(DEVICE_LIST):
    file_path = os.path.join(DATA_DIR, device_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    if len(lines) < TRAIN_FLOWS + TEST_FLOWS:
        raise ValueError(f"{device_file} does not have enough flows ({len(lines)} found)")

    train_sentences_per_device.append(lines[:TRAIN_FLOWS])
    test_sentences.extend(lines[TRAIN_FLOWS:TRAIN_FLOWS+TEST_FLOWS])
    test_labels.extend([device_idx] * TEST_FLOWS)

# ===========================
# TOKENIZER AND MODEL
# ===========================
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

class BertMLPClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes, hidden_dim=256):
        super(BertMLPClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        # Xavier initialization for classifier
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertMLPClassifier(PRETRAINED_MODEL, len(DEVICE_LIST), HIDDEN_DIM)
model.to(device)

# ===========================
# FREEZE ALL BERT LAYERS EXCEPT LAST
# ===========================
for name, param in model.bert.named_parameters():
    param.requires_grad = False
for param in model.bert.encoder.layer[-1].parameters():
    param.requires_grad = True  # fine-tune last layer

# ===========================
# OPTIMIZER & LOSS
# ===========================
# Only update classifier + last BERT layer
optimizer = AdamW(
    list(model.classifier.parameters()) + list(model.bert.encoder.layer[-1].parameters()),
    lr=LR
)
criterion = nn.CrossEntropyLoss()

# ===========================
# CREATE CHRONOLOGICALLY MIXED BATCHES
# ===========================
def create_mixed_batches(sentences_per_device, batch_size_per_device):
    """
    Each batch contains batch_size_per_device flows from each device,
    preserving chronological order.
    """
    num_devices = len(sentences_per_device)
    max_flows = len(sentences_per_device[0])
    batches = []

    for start_idx in range(0, max_flows, batch_size_per_device):
        batch_sentences, batch_labels = [], []
        for device_idx in range(num_devices):
            end_idx = min(start_idx + batch_size_per_device, max_flows)
            batch_sentences.extend(sentences_per_device[device_idx][start_idx:end_idx])
            batch_labels.extend([device_idx] * (end_idx - start_idx))
        batches.append((batch_sentences, batch_labels))
    return batches

# ===========================
# TRAINING
# ===========================
print("Starting training...")
model.train()
batches = create_mixed_batches(train_sentences_per_device, BATCH_SIZE_PER_DEVICE)
for epoch in range(EPOCHS):
    loop = tqdm(batches, leave=False)
    for batch_sentences, batch_labels in loop:
        optimizer.zero_grad()
        encoding = tokenizer(batch_sentences,
                             add_special_tokens=True,
                             max_length=MAX_LEN,
                             padding='max_length',
                             truncation=True,
                             return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss=loss.item())

# ===========================
# TESTING (individual flows)
# ===========================
print("\nStarting testing...")
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    loop = tqdm(zip(test_sentences, test_labels), total=len(test_sentences))
    for sentence, label in loop:
        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

# ===========================
# METRICS
# ===========================
cm = confusion_matrix(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')

print("\nConfusion Matrix:")
print(cm)
print(f"\nMacro F1 Score: {f1_macro:.4f}")
