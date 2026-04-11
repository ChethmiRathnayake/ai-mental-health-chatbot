import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from emotional_memory import EmotionalMemory
from adapt_prediction import predict_with_memory
from responses import get_response
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EMOTIONS
EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'neutral']
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)
NUM_CLASSES = len(EMOTIONS)

#  MODEL
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

# LOAD MODEL
MODEL_PATH = "saved_models/bert_emotion_session3.pt"
model = BertEmotionClassifier(n_classes=NUM_CLASSES).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"--> Loaded model from {MODEL_PATH}")

# --> TOKENIZER & MEMORY
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
memory = EmotionalMemory()

# CHAT
print("***--> Chatbot is ready! Type 'exit' to quit.")
student_id = input("Enter student ID: ")
history = memory.get_student_history(student_id)
if history is not None:
    last_entry = history.iloc[-1]
    print(f"--> Last emotion for student {student_id}: {last_entry['detected_emotion']} (Confidence: {last_entry['confidence']:.2f})")
else:
    print(f"--> No previous emotion found for student {student_id}")

while True:
    try:
        text = input(f"{student_id}: ")
    except KeyboardInterrupt:
        print("\n**--> Chatbot session ended.")
        break

    if text.lower() in ["exit", "quit"]:
        print("**--> Chatbot session ended.")
        break

    predicted_emotion, confidence = predict_with_memory(
        model, tokenizer, text, student_id, memory, label_encoder, DEVICE
    )

    response = get_response(predicted_emotion)

    print(f"--> Memory updated for student {student_id}")
    print(f"--> Emotion: {predicted_emotion} (Confidence: {confidence:.2f})")
    print(f"--> {response}\n")
