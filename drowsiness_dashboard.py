import cv2
import mediapipe as mp
import math
import joblib
import streamlit as st
import threading
import pygame

# Alarm function
def sound_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load('alarm.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# EAR calculation
def EAR(landmarks, eye_indices):
    def get_point(i): return [landmarks[i].x, landmarks[i].y]
    eye = [get_point(i) for i in eye_indices]
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
model = joblib.load("drowsiness_model.pkl")
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]

CONSEC_FRAMES = 30
COUNTER = 0
ALARM_ON = False

# Streamlit UI
st.title("Driver Drowsiness AI Dashboard")
frame_display = st.image([])  # For video feed
ear_display = st.metric(label="Eye Aspect Ratio (EAR)", value="0.00")
mouth_display = st.metric(label="Mouth Opening", value="0.00")
alert_display = st.empty()

# Start video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear = 0
    mouth_open = 0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        leftEAR = EAR(landmarks, LEFT_EYE)
        rightEAR = EAR(landmarks, RIGHT_EYE)
        ear = (leftEAR + rightEAR) / 2.0

        mouth_open = math.dist(
            [landmarks[MOUTH[0]].x, landmarks[MOUTH[0]].y],
            [landmarks[MOUTH[1]].x, landmarks[MOUTH[1]].y]
        )

        pred = model.predict([[ear, mouth_open]])[0]

        if pred == 1:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = threading.Thread(target=sound_alarm)
                    t.daemon = True
                    t.start()
                alert_display.error("DROWSINESS ALERT!")
        else:
            COUNTER = 0
            ALARM_ON = False
            alert_display.info("Driver is alert")

    # Update metrics
    ear_display.metric("Eye Aspect Ratio (EAR)", f"{ear:.2f}")
    mouth_display.metric("Mouth Opening", f"{mouth_open:.2f}")

    frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
