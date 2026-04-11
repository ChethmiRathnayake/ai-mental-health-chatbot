import torch
from torch.utils.data import Dataset
import pandas as pd

class EmotionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, label_map=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        #  label mapping
        if label_map is None:
            unique_labels = self.data['label'].str.lower().unique()
            self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label_str = str(self.data.iloc[idx]['label']).lower()
        label = self.label_map[label_str]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
