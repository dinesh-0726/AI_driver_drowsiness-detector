import cv2
import mediapipe as mp
import pandas as pd
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]  # Upper lip, lower lip

data = []

def EAR(landmarks, eye_indices):
    def get_point(i): return [landmarks[i].x, landmarks[i].y]
    eye = [get_point(i) for i in eye_indices]
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

print("[INFO] Press 'a' for alert, 'd' for drowsy, 'q' to quit")

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

        cv2.putText(frame, f"EAR: {ear:.2f} MOUTH: {mouth_open:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') and results.multi_face_landmarks:
        data.append([ear, mouth_open, 0])  # Alert
    elif key == ord('d') and results.multi_face_landmarks:
        data.append([ear, mouth_open, 1])  # Drowsy
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data, columns=["EAR", "Mouth", "Drowsy"])
df.to_csv("dataset.csv", index=False)
print("[INFO] Data saved to dataset.csv")
