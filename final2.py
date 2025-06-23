import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import time
import winsound
import tkinter as tk
from threading import Thread

# ==== Load your trained eye state model ====
model = load_model(r'C:\Drowsy\eye_state_model_new2.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==== Load DNN face detector ====
face_net = cv2.dnn.readNetFromCaffe(
    r"C:\Drowsy\deploy.prototxt",
    r"C:\Drowsy\res10_300x300_ssd_iter_140000.caffemodel"
)

# ==== MediaPipe Face Mesh ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# ==== Eye Landmark Points ====
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]

# ==== Configuration ====
IMAGE_SIZE = 64
EYES_CLOSED_THRESHOLD = 2  # Seconds
ALERT_COOLDOWN = 2
MAX_MISSED_FRAMES = 50
SAVE_FOLDER = r"C:\Drowsy\Saved_Images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==== Runtime Variables ====
cap = cv2.VideoCapture(0)
missed_frames = 0
eyes_closed_start = None
last_alert_time = 0
image_saved = False
consecutive_alerts = 0

# ==== Helper Functions ====
def get_eye_image(frame, landmarks, eye_indices):
    h, w, _ = frame.shape
    x = [int(landmarks[i].x * w) for i in eye_indices]
    y = [int(landmarks[i].y * h) for i in eye_indices]
    x_min, x_max = max(min(x)-10, 0), min(max(x)+10, w)
    y_min, y_max = max(min(y)-10, 0), min(max(y)+10, h)
    eye = frame[y_min:y_max, x_min:x_max]
    if eye.size == 0:
        return None
    eye = cv2.resize(eye, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
    return np.expand_dims(eye, axis=0)

def draw_eye_box(frame, landmarks, eye_indices, color):
    h, w, _ = frame.shape
    x = [int(landmarks[i].x * w) for i in eye_indices]
    y = [int(landmarks[i].y * h) for i in eye_indices]
    x_min, x_max = max(min(x)-10, 0), min(max(x)+10, w)
    y_min, y_max = max(min(y)-10, 0), min(max(y)+10, h)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

def show_popup():
    def popup():
        root = tk.Tk()
        root.title("Drowsiness Alert")
        root.geometry("300x100")
        root.resizable(False, False)
        label = tk.Label(root, text="⚠ ALERT: Drowsy Detected", font=("Arial", 14), fg="red")
        label.pack(expand=True)
        root.after(3000, root.destroy)
        root.mainloop()
    Thread(target=popup).start()

# ==== Main Loop ====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    face_detected = False
    eyes_detected = False
    predictions = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_detected = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            text = f"Face: {confidence * 100:.2f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Face ROI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Left and right eye image
                left_eye_img = get_eye_image(frame, landmarks, LEFT_EYE)
                right_eye_img = get_eye_image(frame, landmarks, RIGHT_EYE)

                draw_eye_box(frame, landmarks, LEFT_EYE, (255, 0, 0))
                draw_eye_box(frame, landmarks, RIGHT_EYE, (255, 0, 0))

                for eye in [left_eye_img, right_eye_img]:
                    if eye is not None:
                        pred = model.predict(eye, verbose=0)[0][0]
                        predictions.append(pred)

                if len(predictions) == 2:
                    eyes_detected = True
                    if all(p < 0.65 for p in predictions):  # Eyes closed
                        status = "closed"
                        color = (0, 0, 255)
                        if eyes_closed_start is None:
                            eyes_closed_start = time.time()
                        elif time.time() - eyes_closed_start > EYES_CLOSED_THRESHOLD:
                            if time.time() - last_alert_time > ALERT_COOLDOWN:
                                winsound.Beep(1000, 500)
                                last_alert_time = time.time()
                                consecutive_alerts += 1

                                # === Show popup and save face image ===
                                if consecutive_alerts >= 2:
                                    show_popup()
                                    consecutive_alerts = 0

                                    if not image_saved:
                                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                                        face_crop = frame[y:y1, x:x1]
                                        if face_crop.size != 0:
                                            face_path = os.path.join(SAVE_FOLDER, f"face_{timestamp}.jpg")
                                            cv2.imwrite(face_path, face_crop)
                                            print(f"Face snapshot saved at: {face_path}")
                                            image_saved = True
                    else:
                        status = "open"
                        color = (0, 255, 0)
                        eyes_closed_start = None
                        consecutive_alerts = 0

                    cv2.putText(frame, f"Eyes: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            break

    # Handle missed frames
    if not (face_detected and eyes_detected):
        missed_frames += 1
    else:
        missed_frames = 0

    if missed_frames > MAX_MISSED_FRAMES:
        print("No face or eyes detected for 50 frames. Exiting...")
        break

    cv2.imshow("Drowsy Driver Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manual Exit.")
        break

cap.release()
cv2.destroyAllWindows()
