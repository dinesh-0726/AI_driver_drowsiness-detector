import cv2
import mediapipe as mp
import math
import joblib
import pygame
import threading

def sound_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load('alarm.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]

# Load trained model
model = joblib.load("drowsiness_model.pkl")

# Parameters
CONSEC_FRAMES = 30  # Adjust to control sensitivity (30 ~ 1 second at 30 fps)
COUNTER = 0
ALARM_ON = False

def EAR(landmarks, eye_indices):
    def get_point(i): return [landmarks[i].x, landmarks[i].y]
    eye = [get_point(i) for i in eye_indices]
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Starting drowsiness detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        leftEAR = EAR(landmarks, LEFT_EYE)
        rightEAR = EAR(landmarks, RIGHT_EYE)
        ear = (leftEAR + rightEAR) / 2.0

        mouth_open = math.dist(
            [landmarks[MOUTH[0]].x, landmarks[MOUTH[0]].y],
            [landmarks[MOUTH[1]].x, landmarks[MOUTH[1]].y]
        )

        # Predict using the trained model
        pred = model.predict([[ear, mouth_open]])[0]

        if pred == 1:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = threading.Thread(target=sound_alarm)
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        # Display EAR and mouth opening for debugging
        cv2.putText(frame, f"EAR: {ear:.2f} MOUTH: {mouth_open:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    cv2.imshow("AI Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
