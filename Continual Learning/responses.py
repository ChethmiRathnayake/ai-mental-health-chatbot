# responses.py
EMOTION_RESPONSES = {
    "joy": [
        "That's wonderful! Keep it up!",
        "I'm so happy to hear that!",
        "Great job! You should be proud."
    ],
    "sadness": [
        "I'm sorry you're feeling down. Do you want to talk about it?",
        "I understand, that must be hard.",
        "Take your time, I'm here to listen."
    ],
    "anger": [
        "I see that you're frustrated. Let's try to calm down.",
        "It's okay to feel angry sometimes.",
        "Take a deep breath, I'm here to help."
    ],
    "fear": [
        "I understand your anxiety. You're not alone.",
        "It's normal to feel nervous. Let's focus on solutions.",
        "Try to take it one step at a time."
    ],
    "neutral": [
        "Thanks for sharing.",
        "I see. Let's continue.",
        "Okay, got it."
    ]
}

import random

 
def get_response(emotion):
    responses = {
        "joy": "That's wonderful! Keep it up!",
        "sadness": "Take your time, I'm here to listen.",
        "anger": "I understand, let's try to calm down together.",
        "fear": "Don't worry, we will face it step by step.",
        "neutral": "Okay, noted!"
    }
    return responses.get(emotion, "I see.")