from flask import Flask, request, jsonify, session
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from datetime import datetime

# Your original imports and code
from emotional_memory import EmotionalMemory
from adapt_prediction import predict_with_memory
from responses import get_response

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session
CORS(app, supports_credentials=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EMOTIONS
EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'neutral']
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)
NUM_CLASSES = len(EMOTIONS)

# MODEL (same as your original)
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
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"--> Loaded model from {MODEL_PATH}")
else:
    print(f"--> Warning: Model not found at {MODEL_PATH}")

# TOKENIZER & MEMORY
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
memory = EmotionalMemory()

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Start a new chat session for a student"""
    data = request.get_json()
    student_id = data.get('student_id')
    
    if not student_id:
        return jsonify({'error': 'Student ID required'}), 400
    
    # Store student_id in session
    session['student_id'] = student_id
    
    # Get student history
    history = memory.get_student_history(student_id)
    last_emotion = None
    last_confidence = None
    
    if history is not None and not history.empty:
        last_entry = history.iloc[-1]
        last_emotion = last_entry['detected_emotion']
        last_confidence = float(last_entry['confidence'])
    
    return jsonify({
        'success': True,
        'student_id': student_id,
        'last_emotion': last_emotion,
        'last_confidence': last_confidence,
        'history_exists': history is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict emotion from text"""
    data = request.get_json()
    text = data.get('text')
    student_id = data.get('student_id') or session.get('student_id')
    
    if not text:
        return jsonify({'error': 'Text required'}), 400
    
    if not student_id:
        return jsonify({'error': 'Student ID required'}), 400
    
    # Call your original predict_with_memory function
    predicted_emotion, confidence = predict_with_memory(
        model, tokenizer, text, student_id, memory, label_encoder, DEVICE
    )
    
    # Get response using your original get_response function
    response_text = get_response(predicted_emotion)
    
    return jsonify({
        'success': True,
        'student_id': student_id,
        'text': text,
        'predicted_emotion': predicted_emotion,
        'confidence': float(confidence),
        'response': response_text,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/history/<student_id>', methods=['GET'])
def get_history(student_id):
    """Get emotion history for a student"""
    history = memory.get_student_history(student_id)
    
    if history is None or history.empty:
        return jsonify({
            'success': True,
            'student_id': student_id,
            'history': [],
            'total_interactions': 0
        })
    
    # Convert DataFrame to list of dictionaries
    history_list = []
    for _, row in history.iterrows():
        history_list.append({
            'timestamp': row['timestamp'],
            'user_input': row['user_input'],
            'detected_emotion': row['detected_emotion'],
            'confidence': float(row['confidence'])
        })
    
    # Get emotion trends
    emotion_counts = history['detected_emotion'].value_counts().to_dict()
    
    return jsonify({
        'success': True,
        'student_id': student_id,
        'history': history_list,
        'total_interactions': len(history_list),
        'emotion_distribution': emotion_counts,
        'most_common_emotion': history['detected_emotion'].mode()[0] if not history.empty else None
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'model_loaded': os.path.exists(MODEL_PATH),
        'emotions': EMOTIONS
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)