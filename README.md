# Drowsy-Driver-Detection-System


**<h3>Overview</h3>**

Driver drowsiness is a significant contributor to road accidents worldwide, often resulting in serious injuries or fatalities. Fatigue impairs alertness, slows reaction times, and hampers decision-making—especially during long trips, monotonous highway driving, or late-night travel when circadian rhythms naturally promote sleep.

This project presents the Drowsy Driver Detection System, a real-time application that uses computer vision and deep learning to identify signs of drowsyness based on eye state (open / closed). It monitors the driver's eyes through a camera feed and triggers alerts if prolonged eye closure (typically beyond 2–3 seconds) is detected, helping prevent accidents before they happen. The system explored multiple face detection techniques—including CNN, HOG, and DNN—and performed a comparative analysis to evaluate their accuracy and robustness. Similarly, Haar Cascade and MediaPipe were compared for eye detection, with the final implementation using DNN for face detection and MediaPipe Face Mesh for eye localization due to their superior performance in real-time conditions.

To ensure robust performance under varied lighting and occlusion conditions, the system employs:
* Deep Neural Networks (DNN) for fast and accurate face detection,
* MediaPipe Face Mesh for high-precision eye localization,
* A Convolutional Neural Network (CNN) for classifying eye states as open or closed.

The solution is non-intrusive, scalable, and adaptable, designed for real-world deployment in personal and commercial vehicles. With real-time alerts and detailed incident logging, it not only enhances immediate driver safety but also supports future system improvements through post-analysis.



**<h3>Objectives:</h3>** 

The goal of this project is to design a real-time, non-intrusive drowsiness detection system that enhances driver safety by leveraging computer vision and deep learning techniques. The key objectives are:

- Real-Time Face and Eye Detection: Detect the driver’s face and accurately localize eye regions from live video using advanced techniques such as DNN and MediaPipe.

- Eye State Classification: Develop and train a CNN-based binary classifier to determine whether the driver's eyes are open or closed.--

- Drowsiness Detection: Continuously monitor eye state over time and identify signs of drowsiness based on prolonged eye closure exceeding a set threshold (e.g., 2–3 seconds).

- Alert Mechanism: Trigger audio alarms and pop-up messages to immediately alert the driver upon detecting drowsiness.

- Incident Logging: Record key data (e.g., timestamps, screenshots) for each drowsiness event to enable later analysis and performance monitoring.

- Robustness and Adaptability: Ensure the system performs reliably under varying conditions, including changes in lighting, partial facial occlusions, and different user profiles.

- Ease of Deployment: Design the system for easy integration into real-world environments using readily available hardware (e.g., webcam, speaker) and standard platforms (Windows OS).



**<h3>Features:</h3>** 

* Real-time face and eye detection using webcam
* CNN-based binary classifier for eye state (open/closed)
* Alert mechanism (audio & popup) for drowsiness detection
* Incident logging with timestamps and screenshots
* Robust detection under low-light or partially occluded conditions
* GUI using Tkinter for non-blocking alerts



**<h3>Technologies Used:</h3>**

* Programming Language: Python 3.8+
* Libraries/Frameworks: OpenCV, TensorFlow/Keras, Dlib, Mediapipe, Tkinter, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, winsound(tools)
* Development Environment: Jupyter Notebook, Spyder, VS Code, Kaggle
* Deployment Options: Windows 10/11 (Local)
* Hardware: Webcam, Speaker


