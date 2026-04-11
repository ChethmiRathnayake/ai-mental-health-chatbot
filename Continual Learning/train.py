import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SESSIONS_DIR = "dataset"
SESSION_FILES = [
    f"{SESSIONS_DIR}/session_1_week1.csv",
    f"{SESSIONS_DIR}/session_2_week2.csv",
    f"{SESSIONS_DIR}/session_3_week3.csv"
]
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
MAX_LEN = 128


# DATASET CLASS
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# MODEL
class BertEmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.out(self.dropout(pooled_output))

# TOKENIZER

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# PREPARE LABELS
all_labels = []
for f in SESSION_FILES:
    df = pd.read_csv(f)
    df['label'] = df['label'].str.lower().str.strip()
    all_labels.extend(df['label'].tolist())

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
NUM_CLASSES = len(label_encoder.classes_)
print("Emotion labels:", label_encoder.classes_)


def create_dataloader(file_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].str.lower().str.strip()
    y = label_encoder.transform(df['label'])
    dataset = EmotionDataset(df['text'].tolist(), y, tokenizer, MAX_LEN)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True), y

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# EMOTIONAL MEMORY
EMOTIONAL_MEMORY_FILE = "dataset/emotional_memory.csv"
from datetime import datetime, timedelta
import pandas as pd

def fix_emotional_memory(file_path):
    new_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"Skipping malformed line: {line}")
                continue
            student_id = parts[0]
            timestamp = parts[1]
            detected_emotion = parts[-2]
            confidence = parts[-1]
            text = " ".join(parts[2:-2])
            new_rows.append([student_id, timestamp, text, detected_emotion, confidence])

    df_fixed = pd.DataFrame(new_rows, columns=["student_id", "timestamp", "text", "detected_emotion", "confidence"])

    # Convert timestamp
    base_date = datetime(2025, 1, 1)
    def convert_time(ts_str):
        try:
            minutes, seconds = ts_str.split(":")
            return base_date + timedelta(minutes=int(minutes), seconds=float(seconds))
        except:
            return base_date

    df_fixed['timestamp'] = df_fixed['timestamp'].apply(convert_time)

    fixed_path = file_path.replace(".csv", "_fixed.csv")
    df_fixed.to_csv(fixed_path, index=False)
    print(f"--> Emotional memory fixed and saved to: {fixed_path}")
    return fixed_path, df_fixed


fixed_memory_file, df_memory = fix_emotional_memory(EMOTIONAL_MEMORY_FILE)


# TRAINING

model = BertEmotionClassifier(n_classes=NUM_CLASSES).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

#  class weights
all_y = []
for f in SESSION_FILES:
    _, y = create_dataloader(f)
    all_y.extend(y)
class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=all_y)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

for i, session_file in enumerate(SESSION_FILES, 1):
    print(f"\n Training on session {i}: {session_file}")
    dataloader, _ = create_dataloader(session_file)
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, dataloader, optimizer, criterion)
        acc = evaluate_accuracy(model, dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
    save_path = f"{MODEL_SAVE_DIR}/bert_emotion_session{i}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"--> Model saved to {save_path}")

print("\n--> Training complete! Model ready ...")
