import torch
from torch import nn
from transformers import BertModel

class BertEmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        dropped = self.drop(pooled_output)
        return self.out(dropped)
