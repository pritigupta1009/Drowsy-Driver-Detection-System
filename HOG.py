# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 01:32:09 2025

@author: vkg
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 02:17:24 2024

@author: vkg
"""

import cv2
import dlib
from imutils import face_utils

# Load pre-trained Haar cascade for eye detection
# Modify the path according to your installation
eye_cascade_path = 'C:/Users/vkg/.conda/envs/opencv_env/Library/etc/haarcascades/haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Opens the default webcam for video capturing

# Check if the camera is opened properly
if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()
frame_width=430
frame_height=430
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

no_eye_count = 0  # Counter for frames without detected eyes
max_no_eye_frames = 20   # Exit if no eyes detected for 20 frames

# Dlib face detector for frontal face detection
face_detect = dlib.get_frontal_face_detector()

while True: 
    ret, frame = video_capture.read()  # Capture a frame from the webcam

    # If frame capture fails, ret will be False
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    rects, scores, _ = face_detect.run(gray, 1)  # Detect faces with confidence scores

    eyes_detected = False  # Flag to track if eyes are detected

    for (i, rect) in enumerate(rects):  # Loop through detected faces
        (x, y, w, h) = face_utils.rect_to_bb(rect)  # Convert dlib rectangle to bounding box coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the face

        # Show confidence score
        confidence = scores[i]
        cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Detect eyes within the detected face region
        face_region = gray[y:y+h, x:x+w]  # Extract the region of interest (face)
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(eyes) > 0:  # If eyes are detected
            eyes_detected = True  # Set the flag to True
            no_eye_count = 0  # Reset eye counter

        for (ex, ey, ew, eh) in eyes:  # Loop through detected eyes
            # Draw a rectangle around each detected eye
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

    # If no eyes are detected, increment the counter
    if not eyes_detected:
        no_eye_count += 1
    else:
        no_eye_count = 0  # Reset if eyes are detected
    
    cv2.imshow('Video', frame) # Display the processed frame

    # Exit if no eyes have been detected for too many frames
    if no_eye_count > max_no_eye_frames:
        print("No eyes detected for 20 frames. Exiting...")
        break
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
