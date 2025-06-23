import cv2
import os

# Full paths to DNN model files
prototxt_path = r"C:\Users\srita\OneDrive\Desktop\Face detection\deploy.prototxt"
model_path = r"C:\Users\srita\OneDrive\Desktop\Face detection\res10_300x300_ssd_iter_140000.caffemodel"

# Load the DNN face detector
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Save folder
save_folder = r"C:\Users\srita\OneDrive\Desktop\Face detection\Saved_Images"
os.makedirs(save_folder, exist_ok=True)

# Flags
saved = False
missed_frames = 0
max_missed_frames = 50  # Changed to 50 frames now

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit manually.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    face_detected = False
    eyes_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            face_detected = True
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x, y, x1, y1) = box.astype("int")

            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            text = f"Face: {confidence * 100:.2f}%"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            face_roi = frame[y:y1, x:x1]
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            eyes = eye_cascade.detectMultiScale(gray_roi, scaleFactor=1.1,
                                                minNeighbors=10, minSize=(30, 30))

            if len(eyes) > 0:
                eyes_detected = True

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            eye_accuracy = min(len(eyes), 2) / 2 * 100
            eye_text = f"Eyes: {eye_accuracy:.2f}%"
            cv2.putText(frame, eye_text, (x, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Save image once on first detection
            if not saved and eyes_detected:
                image_path = os.path.join(save_folder, "detection.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Image saved at: {image_path}")
                saved = True

    if not (face_detected and eyes_detected):
        missed_frames += 1
    else:
        missed_frames = 0

    if missed_frames > max_missed_frames:
        print("No face or eyes detected for 50 frames. Exiting program.")
        break

    cv2.imshow("Face and Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manual exit.")
        break

cap.release()
cv2.destroyAllWindows()
