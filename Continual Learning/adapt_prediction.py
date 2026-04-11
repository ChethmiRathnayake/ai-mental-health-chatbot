import torch

def predict_with_memory(model, tokenizer, text, student_id, memory, label_encoder, device):
    # Tokenize input
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Predict with model
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        predicted_emotion = label_encoder.inverse_transform([pred_idx.item()])[0]
        confidence = confidence.item()

    # Adjust prediction based on past memory
    trends = memory.get_emotion_trends(student_id)
    if trends:

        for emotion, weight in trends.items():
            if emotion == predicted_emotion:
                confidence = min(confidence + weight * 0.1, 1.0)

    #  to memory
    memory.add(student_id, text, predicted_emotion, confidence)

    return predicted_emotion, confidence
