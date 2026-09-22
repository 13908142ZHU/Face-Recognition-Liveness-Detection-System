# Real-Time Face Recognition & Liveness Detection System

## INTRODUCTION

A real-time face recognition and liveness detection system developed based on OpenCV and dlib/face_recognition, featuring SQLite database storage and shortcut key interaction.

### Key Features
*   **Real-Time Liveness Detection**: Utilizes the Eye Aspect Ratio (EAR) algorithm to track eye landmarks and count blinks. The system verifies the presence of a live person before granting recognition access, successfully preventing photo-based spoofing attacks.
*   **Two-Phase Matching Algorithm**: Enhances matching efficiency and accuracy by initially filtering candidates based on 128-dimensional eye region encodings, followed by a comprehensive facial encoding comparison.
*   **Seamless Registration**: Features an interactive dual-mode interface. Users can easily toggle between *Recognise Mode* and *Register Mode* using keyboard shortcuts, allowing for on-the-fly facial data capture and enrollment via terminal input.
*   **Local Database Integration**: Employs SQLite to securely and efficiently store users' names, facial encodings, and eye encodings in binary large objects (BLOBs) for fast retrieval.

## INSTALLATION
### MacOS
**Please run in the terminal:**

    1. python3 -m venv venv
    2. source ./venv/bin/activate
    3. pip install -r requirements.txt
    4. python main.py
---
### Windows

**Please run in the terminal:**

    1. python -m venv venv
    2. .\venv\Scripts\activate(better to use cmd instead of PowerShell)
    3. pip install -r requirements.txt
    4. python main.py

## SHORT KEY GUYS

Once the webcam window opens, click on the window and use the following hotkeys to control the application:

| Key | Mode / Function | Description |
| :---: | :--- | :--- |
| **`R`** | Toggle Mode | Switch between **RECOGNISE MODE** and **REGISTER MODE**. |
| **`S`** | Save Face | **Active only in Register Mode**. Press to input a name in the terminal and register the face. |
| **`Q`** | Quit | Safely release the camera and exit the application. |
