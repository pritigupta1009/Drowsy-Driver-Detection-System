import cv2
import numpy as np
import time

# Load OpenCV DNN face detector (CNN-based)
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
config_path = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Timeout settings
timeout_interval = 5
last_face_detected_time = time.time()
face_and_eyes_saved = False

def save_face(face_roi, file_name):
    cv2.imwrite(file_name, face_roi)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    h, w = frame.shape[:2]

    # Prepare image for DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_detected = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # Clamp box to frame size
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray_roi_resized = cv2.resize(gray_roi, (200, 200))

            eyes = eye_cascade.detectMultiScale(
                gray_roi_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            # Calculate and print eye detection accuracy
            eye_accuracy = (len(eyes) / 2) * 100
            print(f"Eyes detected: {len(eyes)} - Accuracy: {eye_accuracy:.2f}%")

            # Draw rectangles around detected eyes
            for (ex, ey, ew, eh) in eyes:
                x_scale = face_roi.shape[1] / 200
                y_scale = face_roi.shape[0] / 200
                ex, ey, ew, eh = int(ex * x_scale), int(ey * y_scale), int(ew * x_scale), int(eh * y_scale)
                cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            if len(eyes) == 2 and not face_and_eyes_saved:
                file_name = f"detected_face_{time.time()}.png"
                save_face(face_roi, file_name)
                print(f"Face and both eyes detected. Image saved as {file_name}")
                face_and_eyes_saved = True

    if face_detected:
        last_face_detected_time = time.time()
    else:
        face_and_eyes_saved = False

    if time.time() - last_face_detected_time > timeout_interval:
        print(f"No face detected for {timeout_interval} seconds. Stopping detection.")
        break

    cv2.imshow('Face and Eye Detection (OpenCV Only)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
