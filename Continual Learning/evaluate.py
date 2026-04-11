import torch
from sklearn.metrics import accuracy_score
from continual_loader import load_session

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc

# Example: evaluate on old session
dataset_eval = load_session("dataset/session_1_week1.csv", tokenizer, label_encoder)
dataloader_eval = DataLoader(dataset_eval, batch_size=4)
acc = evaluate(model, dataloader_eval)
print(f"Accuracy on old session: {acc:.2f}")
