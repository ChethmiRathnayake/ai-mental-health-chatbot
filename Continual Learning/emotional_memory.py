import pandas as pd

class EmotionalMemory:
    def __init__(self, file_path="dataset/emotional_memory.csv"):
        self.file_path = file_path
        try:
            self.df = pd.read_csv(file_path)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=[
                "student_id", "timestamp", "text", "detected_emotion", "confidence"
            ])

    def add(self, student_id, text, emotion, confidence):
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        new_row = {
            "student_id": int(student_id),
            "timestamp": timestamp,
            "text": text,
            "detected_emotion": emotion,
            "confidence": confidence
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save()  # save immediately

    def save(self):
        self.df.to_csv(self.file_path, index=False)

    def get_student_history(self, student_id):
        student_id = int(student_id)
        history = self.df[self.df["student_id"] == student_id]
        if history.empty:
            return None
        return history

    def get_emotion_trends(self, student_id):
        history = self.get_student_history(student_id)
        if history is None:
            return None
        return history['detected_emotion'].value_counts(normalize=True).to_dict()
