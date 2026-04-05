from flask import Flask, jsonify, request, Response
from flask_cors import CORS
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
import base64
from tensorflow.keras.models import load_model
from playsound import playsound
import os
from datetime import datetime
import queue

app = Flask(__name__)
CORS(app)

# Global state
class StressDetectionSystem:
    def __init__(self):
        # Model and cascade
        self.model = load_model("stress_detection_model.h5")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Camera
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Detection state
        self.stress_count = 0
        self.total_stress_frames = 0
        self.stress_episodes = 0
        self.start_time = time.time()
        self.stress_level = "no_stress"
        self.fps = 0
        self.face_detected = False
        self.last_prediction_time = 0
        self.prediction_interval = 0.1  # 100ms between predictions
        
        # BPM monitoring
        self.serial_monitor = None
        self.bpm_history = []
        self.max_bpm_history = 300
        self.latest_bpm = 0
        
        # Recommendations
        self.rec_manager = RecommendationManager("recommendations.json")
        self.current_recommendation = None
        self.recommendation_history = []
        
        # Configuration
        self.confidence_threshold = 0.5
        self.classification_interval = 30  # seconds
        
        # Threading
        self.camera_thread = None
        self.stats_thread = None
        self.running = False
        
        # Frame queue for streaming
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Session info
        self.session_start = None
        self.session_active = False

    def start_camera(self):
        if self.camera_running:
            return True
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
            
            self.camera_running = True
            self.running = True
            self.session_start = datetime.now()
            self.session_active = True
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            
            # Start stats thread
            self.stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
            self.stats_thread.start()
            
            # Start serial monitor
            self._start_serial_monitor()
            
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def stop_camera(self):
        self.running = False
        self.camera_running = False
        self.session_active = False
        if self.cap:
            self.cap.release()
        if self.serial_monitor:
            self.serial_monitor.running = False

    def _start_serial_monitor(self):
        try:
            self.serial_monitor = SerialMonitor("COM3", 115200)
            self.serial_monitor.connect()
            self.serial_monitor.start()
        except Exception as e:
            print(f"Serial monitor error: {e}")

    def _camera_loop(self):
        frame_count = 0
        fps_start = time.time()
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Calculate FPS
                if time.time() - fps_start >= 1.0:
                    self.fps = frame_count
                    frame_count = 0
                    fps_start = time.time()
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Store frame for streaming
                with self.frame_lock:
                    self.current_frame = processed_frame
                
                # Put frame in queue for streaming (if not full)
                if self.frame_queue.qsize() < 10:
                    self.frame_queue.put(processed_frame)
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                time.sleep(0.1)

    def _process_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        self.face_detected = len(faces) > 0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Stress prediction (throttled)
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval:
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (128, 128))
                face_array = np.expand_dims(face_resized / 255.0, axis=0)
                
                prediction = self.model.predict(face_array, verbose=0)[0][0]
                
                if prediction > self.confidence_threshold:
                    self.stress_count += 1
                    self.total_stress_frames += 1
                    
                    # Detect stress episodes (continuous stress for > 5 seconds)
                    if self.stress_count > 50:  # Roughly 5 seconds at 10fps
                        self.stress_episodes += 1
                
                label = "Stress" if prediction > self.confidence_threshold else "No Stress"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                self.last_prediction_time = current_time
        
        # Add overlay text
        cv2.putText(frame, f"BPM: {self.latest_bpm:.1f}" if self.latest_bpm else "BPM: --", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Stress Frames: {self.stress_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Level: {self.stress_level}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {self.fps}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def _stats_loop(self):
        while self.running:
            try:
                # Classify stress level periodically
                if time.time() - self.start_time >= self.classification_interval:
                    self.stress_level = classify_stress_level(self.stress_count)
                    print(f"Stress Level updated: {self.stress_level}")
                    
                    # Get recommendation
                    self.current_recommendation = self.rec_manager.get_next_recommendation(self.stress_level)
                    if self.current_recommendation:
                        self.recommendation_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'level': self.stress_level,
                            'recommendation': self.current_recommendation
                        })
                    
                    self.stress_count = 0
                    self.start_time = time.time()
                
                # Update BPM history
                if self.serial_monitor and self.serial_monitor.latest_bpm != "--":
                    bpm = float(self.serial_monitor.latest_bpm)
                    self.latest_bpm = bpm
                    self.bpm_history.append(bpm)
                    if len(self.bpm_history) > self.max_bpm_history:
                        self.bpm_history.pop(0)
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Stats loop error: {e}")
                time.sleep(1)

    def get_current_frame_base64(self):
        with self.frame_lock:
            if self.current_frame is None:
                return None
            
            try:
                # Resize frame for streaming
                frame_resized = cv2.resize(self.current_frame, (640, 480))
                _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"Frame encoding error: {e}")
                return None

    def get_detections(self):
        return {
            'stress_level': self.stress_level,
            'bpm': self.latest_bpm,
            'stress_frames': self.stress_count,
            'total_stress_frames': self.total_stress_frames,
            'stress_episodes': self.stress_episodes,
            'fps': self.fps,
            'face_detected': self.face_detected,
            'timestamp': datetime.now().isoformat()
        }

    def get_stats(self):
        return {
            'history': {
                'bpm': self.bpm_history[-50:] if len(self.bpm_history) > 50 else self.bpm_history,
                'total_stress_frames': self.total_stress_frames,
                'stress_episodes': self.stress_episodes,
                'session_duration': (datetime.now() - self.session_start).total_seconds() if self.session_start else 0
            },
            'recommendations': self.recommendation_history[-10:]  # Last 10 recommendations
        }

    def reset_stats(self):
        self.total_stress_frames = 0
        self.stress_episodes = 0
        self.bpm_history.clear()
        self.recommendation_history.clear()
        self.stress_count = 0
        self.start_time = time.time()

    def update_config(self, config):
        if 'confidence' in config:
            self.confidence_threshold = config['confidence']
        if 'classification_interval' in config:
            self.classification_interval = config['classification_interval']

    def send_feedback(self, feedback_data):
        if feedback_data.get('feedback') == 'no':
            self.rec_manager.adapt_recommendation(
                self.stress_level, 
                feedback_data.get('recommendation_id', '')
            )
        return True


# Serial Monitor Class (from original code)
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


# Recommendation Manager (from original code)
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


# Helper functions
def classify_stress_level(stress_count):
    if stress_count < 50:
        return "no_stress"
    elif 50 <= stress_count < 110:
        return "low"
    elif 110 <= stress_count < 230:
        return "moderate"
    else:
        return "high"


# Initialize system
system = StressDetectionSystem()


# API Routes
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/connect', methods=['POST'])
def connect():
    success = system.start_camera()
    if success:
        return jsonify({'status': 'connected', 'message': 'Camera started successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start camera'}), 500


@app.route('/disconnect', methods=['POST'])
def disconnect():
    system.stop_camera()
    return jsonify({'status': 'disconnected'})


@app.route('/stream', methods=['GET'])
def stream():
    if not system.camera_running:
        return jsonify({'error': 'Camera not running'}), 400
    
    frame_base64 = system.get_current_frame_base64()
    detections = system.get_detections()
    
    return jsonify({
        'frame': frame_base64,
        'detections': detections,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(system.get_stats())


@app.route('/recommendation', methods=['GET'])
def get_recommendation():
    if system.current_recommendation:
        return jsonify({'recommendation': system.current_recommendation})
    else:
        # Force generate recommendation
        system.stress_level = classify_stress_level(system.stress_count)
        system.current_recommendation = system.rec_manager.get_next_recommendation(system.stress_level)
        return jsonify({'recommendation': system.current_recommendation})


@app.route('/recommendation/feedback', methods=['POST'])
def recommendation_feedback():
    data = request.json
    success = system.send_feedback(data)
    if success:
        return jsonify({'status': 'feedback recorded'})
    else:
        return jsonify({'error': 'Failed to record feedback'}), 500


@app.route('/snapshot', methods=['POST'])
def take_snapshot():
    if system.current_frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, system.current_frame)
        return jsonify({'status': 'saved', 'filename': filename})
    return jsonify({'error': 'No frame available'}), 400


@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    system.reset_stats()
    return jsonify({'status': 'stats reset'})


@app.route('/config', methods=['POST'])
def update_config():
    data = request.json
    system.update_config(data)
    return jsonify({'status': 'config updated', 'config': {
        'confidence': system.confidence_threshold,
        'classification_interval': system.classification_interval
    }})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'camera_running': system.camera_running,
        'session_active': system.session_active,
        'session_start': system.session_start.isoformat() if system.session_start else None,
        'face_detected': system.face_detected,
        'stress_level': system.stress_level
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)