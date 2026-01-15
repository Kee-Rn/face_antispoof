🧠 Face Anti-Spoofing System 
📌 Project Overview

This project is a Face Anti-Spoofing (Liveness Detection) System that detects whether a face shown to a camera is REAL (live human) or SPOOF (photo / fake attack).

The system uses:

Convolutional Neural Networks (CNN)

NUAA Face Anti-Spoofing Dataset

MediaPipe for face detection

ONNX Runtime for fast real-time inference

OpenCV for webcam processing

🤖 AI Assistance & Motivation (Important Note)

⚠️ Transparency Statement

This project was fully created with the help of AI, mainly using Google AI Studio and AI-based guidance.

I got motivated after seeing a similar face anti-spoofing project shared on Reddit

I do not claim full originality of the idea

I used AI extensively to:

Understand concepts

Write and debug code

Design the project flow

Learn how face anti-spoofing systems work

This project represents learning through AI-assisted development, not independent research.

🗂️ Project Structure
face_antispoof/
│
├── 00_capture_data.py        # Collect custom real & spoof images (optional)
├── 01_organize_data.py       # Organize NUAA dataset into train/test
├── 02_train_nuaa.py          # Train CNN model
├── convert_final.py          # Convert TensorFlow model to ONNX
├── 03_run_live.py            # Run real-time anti-spoof detection
│
├── dataset/
│   ├── train/
│   │   ├── real/
│   │   └── spoof/
│   └── test/
│       ├── real/
│       └── spoof/
│
├── saved_model_tf/           # TensorFlow SavedModel (generated)
├── nuaa_antispoof.onnx       # Final ONNX model (generated)
└── README.md

🧪 Dataset Used

NUAA Face Anti-Spoofing Dataset

ClientRaw → Real faces

ImposterRaw → Spoof (photo attack) faces

You must download and unzip the dataset in the project root folder.

⚙️ Requirements

Make sure you have Python 3.8+ installed.

Required Python Libraries
pip install tensorflow opencv-python mediapipe numpy onnxruntime tf2onnx

🚀 How to Run This Project on Your Device
🔹 Step 1: Prepare Dataset

Download the NUAA dataset and unzip it so you have:

ClientRaw/
ImposterRaw/


in the same directory as the scripts.

🔹 Step 2: Organize Dataset
python 01_organize_data.py


This will create:

dataset/
 ├── train/
 └── test/

🔹 Step 3: Train the Model
python 02_train_nuaa.py


Trains a CNN model

Saves the model in saved_model_tf/

🔹 Step 4: Convert Model to ONNX
python convert_final.py


Converts TensorFlow model to ONNX

Creates:

nuaa_antispoof.onnx

🔹 Step 5: Run Real-Time Face Anti-Spoofing
python 03_run_live.py


Webcam opens

Green box → REAL

Red box → SPOOF

Press q to quit

🟢 Optional: Collect Your Own Data
python 00_capture_data.py


Press r → Save real face

Press s → Save spoof face

Press q → Quit

🧠 How It Works (Simple Explanation)

Webcam captures a frame

MediaPipe detects face

Face is cropped and preprocessed

CNN model predicts liveness

Result displayed in real time

⚠️ Limitations

Uses 2D image-based liveness detection

Vulnerable to:

High-quality video replay

Advanced spoof attacks

Not suitable for production security systems

🎓 Learning Outcome

Through this project, I learned:

CNN fundamentals

Dataset preprocessing

Model training and evaluation

Model conversion (TensorFlow → ONNX)

Real-time computer vision pipelines

AI-assisted development workflow

📜 Disclaimer

This project is for educational purposes only.
Do not use it for real-world authentication or security systems.

⭐ Credits

NUAA Face Anti-Spoofing Dataset

Google AI Studio

Open-source AI community

Reddit (for inspiration)
