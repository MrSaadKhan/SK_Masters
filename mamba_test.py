import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

from tqdm import tqdm  # progress bar

# ======== CONFIGURATION ========
data_folder = "0-5/128/BERT/Ungrouped/10_1/bert_embeddings"  # CHANGE to your folder path containing the txt files
model_name = "state-spaces/mamba-2.8b-slimpj"  # Mamba2 model checkpoint
tokenizer_name = "EleutherAI/gpt-neox-20b"  # Use GPT-NeoX tokenizer as a substitute
num_train_epochs = 3
batch_size = 4
window_size = 5      # number of sentences per sliding window
window_stride = 3    # overlap stride
max_length = 512     # maximum tokens for tokenizer

# ======== HELPER FUNCTIONS ========
def extract_label_from_filename(filename):
    """
    Extracts label from filename.
    Expects format e.g., 'irobot_roomba.json_seen_bert_embeddings.txt'
    where the label is assumed to be the first part before the first '.'.
    """
    base = os.path.basename(filename)
    label = base.split('.')[0]
    return label

def sliding_window(lines, window_size, stride):
    """
    Generates sliding windows over a list of sentences.
    Each window is a grouping of window_size sentences with the given stride (overlap).
    """
    windows = []
    for i in range(0, len(lines) - window_size + 1, stride):
        window_text = " ".join([line.strip() for line in lines[i:i+window_size] if line.strip()])
        windows.append(window_text)
    return windows

def load_data(file_pattern):
    """
    Loads files matching the given glob pattern from data_folder.
    Returns a list of dictionaries:
      - 'text': sliding window text (multiple per file)
      - 'label': extracted from filename.
    Shows a tqdm progress bar while loading.
    """
    data = []
    file_list = glob.glob(os.path.join(data_folder, file_pattern))
    for filename in tqdm(file_list, desc=f"Loading files ({file_pattern})"):
        label = extract_label_from_filename(filename)
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Generate sliding windows from file's lines.
        windows = sliding_window(lines, window_size=window_size, stride=window_stride)
        for window_text in windows:
            data.append({"text": window_text, "label": label})
    return data

# Load seen (for training) and unseen (for evaluation) data with progress indicators
train_data = load_data("*seen*.txt")
test_data  = load_data("*unseen*.txt")

# ==== Build label mapping (from label string to integer) ====
all_labels = set([item["label"] for item in train_data] + [item["label"] for item in test_data])
label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
id2label = {idx: label for label, idx in label2id.items()}

# Add numeric label to each sample
for item in train_data:
    item["label_id"] = label2id[item["label"]]
for item in test_data:
    item["label_id"] = label2id[item["label"]]

# ======== DATASET CLASS ========
class SentenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label_id"]
        # Tokenize with truncation and padding up to max_length
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length, 
            return_tensors="pt"
        )
        # Remove extra batch dimension
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

# ======== CUSTOM MODEL: Mamba for Sequence Classification ========
class MambaForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MambaForSequenceClassification, self).__init__()
        # Load the base Mamba model
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(self.base_model.config.hidden_dropout_prob)
        # Classification head: projects hidden_size to number of labels
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        pooled = hidden_states.mean(dim=1)           # mean pooling over the sequence
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.classifier.out_features), labels.view(-1))
        return {"loss": loss, "logits": logits}

# ======== TOKENIZER & MODEL ========
# Use the GPT-NeoX tokenizer as a substitute.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Instantiate the custom Mamba sequence classification model.
model = MambaForSequenceClassification(model_name=model_name, num_labels=len(label2id))

# ======== PREPARE DATASET ========
train_dataset = SentenceDataset(train_data, tokenizer)
test_dataset  = SentenceDataset(test_data, tokenizer)

# ======== TRAINING SETUP ========
training_args = TrainingArguments(
    output_dir="./mamba_classifier",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ======== TRAINING ========
print("Starting training...")
trainer.train()  # Trainer uses its own progress bar

# ======== EVALUATION ========
print("Starting evaluation on unseen files...")
predictions_output = trainer.predict(test_dataset)
logits, labels = predictions_output.predictions, predictions_output.label_ids
predictions = np.argmax(logits, axis=-1)

# Print classification report
print("\nClassification Report:")
print(classification_report(labels, predictions, target_names=[id2label[i] for i in range(len(label2id))]))

# Compute and plot the confusion matrix
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(label2id))
plt.xticks(tick_marks, [id2label[i] for i in range(len(label2id))], rotation=45)
plt.yticks(tick_marks, [id2label[i] for i in range(len(label2id))])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
