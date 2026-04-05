import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load model
model = load_model("stress_detection_model.h5")

# Load Haar Cascade 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parameters 
stress_count = 0
start_time = time.time()
stress_level = "-"  # Initial stress level

# classify stress level
def classify_stress_level(stress_count):
    if stress_count < 50:
        return "no_stress"
    elif 50 <= stress_count < 110:
        return "low"
    elif 110 <= stress_count < 230:
        return "moderate"
    else:  # stress_count >= 230
        return "high"

# Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Preprocess
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_array = np.expand_dims(face_resized, axis=0)  # Add batch dimension
        face_array = face_array / 255.0 

        # Predict stress level
        prediction = model.predict(face_array)
        label = "Stress" if prediction > 0.5 else "No Stress"

        # Increment stress
        if prediction > 0.5:
            stress_count += 1

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    elapsed_time = time.time() - start_time
    if elapsed_time >= 30:
        stress_level = classify_stress_level(stress_count)
        print(f"Stress Level: {stress_level.replace('_', ' ').capitalize()}")
        start_time = time.time()
        stress_count = 0

    # Display stress frame count and stress level
    cv2.putText(frame, f"Stress Level: {stress_level.replace('_', ' ').capitalize()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Stress Frames (30s): {stress_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Stress Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
