import cv2
import numpy as np
import time
import threading
import serial
import traceback
import sys
import re
import json
import random
from tensorflow.keras.models import load_model
from playsound import playsound
import os


# LOAD STRESS MODEL
model = load_model("stress_detection_model.h5")

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# STRESS LOGIC

stress_count = 0
start_time = time.time()
stress_level = "-"

def classify_stress_level(stress_count):
    if stress_count < 50:
        return "no_stress"
    elif 50 <= stress_count < 110:
        return "low"
    elif 110 <= stress_count < 230:
        return "moderate"
    else:
        return "high"


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


# RECOMMENDATION MANAGER

class RecommendationManager:
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.last_shown = {}  

    def get_next_recommendation(self, stress_level):
        options = self.data.get(stress_level, [])
        if not options:
            return None
        
        last_type = self.last_shown.get(stress_level)
        filtered = [rec for rec in options if rec["type"] != last_type]
        if not filtered:
            filtered = options
        
        choice = random.choice(filtered)
        self.last_shown[stress_level] = choice["type"]
        return choice

    def adapt_recommendation(self, stress_level, user_response):
        if user_response.lower() in ["no", "n"]:
            self.last_shown[stress_level] = None


# INITIALIZE
serial_monitor = SerialMonitor("COM3", 115200)
serial_monitor.connect()
serial_monitor.start()

rec_manager = RecommendationManager("recommendations.json")

graph_width = 300
graph_height = 150
bpm_values = [0]*graph_width

cap = cv2.VideoCapture(0)


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

        prediction = model.predict(face_array, verbose=0)[0][0]
        label = "Stress" if prediction > 0.5 else "No Stress"

        if prediction > 0.5:
            stress_count += 1

        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Every 30 seconds classify stress
    if time.time() - start_time >= 30:
        stress_level = classify_stress_level(stress_count)
        print("30-second Stress Level:", stress_level)
        
        # Adaptive recommendation
        rec = rec_manager.get_next_recommendation(stress_level)
        if rec:
            if rec["type"] == "text":
                print(f"Recommendation (TEXT): {rec['content']}")
                user_input = input("Did you follow the advice? (yes/no): ")
            elif rec["type"] == "audio":
                print(f"Recommendation (AUDIO): {rec['content']}")
                if os.path.exists(rec["content"]):
                    playsound(rec["content"])
                user_input = input("Did you follow the advice? (yes/no): ")
            elif rec["type"] == "video":
                print(f"Recommendation (VIDEO): {rec['content']}")
                if os.path.exists(rec["content"]):
                    video_cap = cv2.VideoCapture(rec["content"])
                    while video_cap.isOpened():
                        ret_v, frame_v = video_cap.read()
                        if not ret_v:
                            break
                        cv2.imshow("Video Recommendation", frame_v)
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break
                    video_cap.release()
                user_input = input("Did you follow the advice? (yes/no): ")

            rec_manager.adapt_recommendation(stress_level, user_input)

        stress_count = 0
        start_time = time.time()

    # Show BPM
    bpm_display = serial_monitor.latest_bpm if serial_monitor.latest_bpm != "--" else 0
    cv2.putText(frame, f"BPM: {bpm_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Stress Frames: {stress_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Stress Level: {stress_level}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Update BPM graph
    bpm_values.append(bpm_display)
    bpm_values.pop(0)
    overlay = frame.copy()
    graph_x = frame_width - graph_width - 10
    graph_y = 10

    cv2.rectangle(overlay, (graph_x-5, graph_y-5),
                  (graph_x+graph_width+5, graph_y+graph_height+5),
                  (50, 50, 50), -1)

    cv2.line(overlay, (graph_x, graph_y), (graph_x, graph_y+graph_height), (200, 200, 200), 1)
    cv2.line(overlay, (graph_x, graph_y+graph_height), (graph_x+graph_width, graph_y+graph_height), (200, 200, 200), 1)

    for i in range(1, len(bpm_values)):
        bpm_prev = max(0, min(180, bpm_values[i-1]))
        bpm_curr = max(0, min(180, bpm_values[i]))
        x1 = graph_x + i - 1
        y1 = graph_y + graph_height - int((bpm_prev) * graph_height / 180)
        x2 = graph_x + i
        y2 = graph_y + graph_height - int((bpm_curr) * graph_height / 180)
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(overlay, "BPM", (graph_x, graph_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(overlay, "Time", (graph_x+graph_width-40, graph_y+graph_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(overlay, "180", (graph_x-35, graph_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(overlay, "0", (graph_x-20, graph_y+graph_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.imshow("Stress & BPM Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
serial_monitor.running = False
