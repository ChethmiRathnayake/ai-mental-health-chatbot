import cv2
import numpy as np
import time
import threading
import serial
import traceback
import sys
import re
from tensorflow.keras.models import load_model
import pandas as pd
import joblib


# LOAD CNN FACIAL STRESS MODEL

face_model = load_model("stress_detection_model.h5")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# LOAD RANDOM FOREST STRESS CLASSIFIER (BPM + stress_count)

clf = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")


# SERIAL MONITOR CLASS
class SerialMonitor:
    def __init__(self, port="COM3", baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.running = False
        self.latest_bpm = "--"

    def connect(self):
        print("Connecting Serial...")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self.running = True
            print(f"Serial connected: {self.port}\n")
        except Exception as e:
            print("ERROR opening serial port!")
            traceback.print_exc()
            sys.exit(1)

    def start(self):
        thread = threading.Thread(target=self.read_loop, daemon=True)
        thread.start()

    def read_loop(self):
        while self.running:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                try:
                    text = raw.decode("utf-8", errors="ignore").strip()
                except:
                    continue
                match = re.search(r"BPM\s*=\s*([0-9.]+)", text)
                if match:
                    self.latest_bpm = float(match.group(1))
            except Exception:
                traceback.print_exc()
                time.sleep(0.3)


# START SERIAL MONITOR

serial_monitor = SerialMonitor("COM5", 115200)
serial_monitor.connect()
serial_monitor.start()


# STRESS COUNTER

stress_count = 0


# LIVE GRAPH DATA
graph_width = 300
graph_height = 150
bpm_values = [0]*graph_width


# START CAMERA
cap = cv2.VideoCapture(0)
start_time = time.time()
predicted_stress_level = "-"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        prediction = face_model.predict(face_array, verbose=0)[0][0]
        label = "Stress" if prediction > 0.5 else "No Stress"

        # Increment stress count if face shows stress
        if prediction > 0.5:
            stress_count += 1

        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Read current BPM
    bpm_display = serial_monitor.latest_bpm if serial_monitor.latest_bpm != "--" else 0


    if time.time() - start_time >= 5:  # update every 5 sec
        sample = pd.DataFrame({'BPM':[bpm_display], 'Stress_Count':[stress_count]})
        sample_scaled = scaler.transform(sample)
        pred_class = clf.predict(sample_scaled)
        predicted_stress_level = le.inverse_transform(pred_class)[0]
        print(f"Predicted Stress Level (ML): {predicted_stress_level}")
        stress_count = 0
        start_time = time.time()

    # Display BPM and predicted stress
    cv2.putText(frame, f"BPM: {bpm_display}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(frame, f"Stress Level (ML): {predicted_stress_level}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


    # Update transparent BPM graph
    bpm_values.append(bpm_display)
    bpm_values.pop(0)

    overlay = frame.copy()
    graph_x = frame_width - graph_width - 10
    graph_y = 10

    cv2.rectangle(overlay, (graph_x-5, graph_y-5),
                  (graph_x+graph_width+5, graph_y+graph_height+5),
                  (50, 50, 50), -1)

    for i in range(1, len(bpm_values)):
        bpm_prev = max(0, min(180, bpm_values[i-1]))
        bpm_curr = max(0, min(180, bpm_values[i]))
        x1 = graph_x + i - 1
        y1 = graph_y + graph_height - int((bpm_prev) * graph_height / 180)
        x2 = graph_x + i
        y2 = graph_y + graph_height - int((bpm_curr) * graph_height / 180)
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.imshow("Stress & BPM Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
serial_monitor.running = False
