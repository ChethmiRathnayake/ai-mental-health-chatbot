import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
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
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

# Load sessions sequentially
def load_session(file_path, tokenizer, label_encoder):
    df = pd.read_csv(file_path)
    # Encode labels
    y = label_encoder.transform(df['label'])
    dataset = EmotionDataset(df['text'].tolist(), y, tokenizer)
    return dataset


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    #  emotion labels
    EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'neutral']
    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTIONS)
    
    dataset = load_session("dataset/session_1_week1.csv", tokenizer, label_encoder)
    print("First sample:", dataset[0])
