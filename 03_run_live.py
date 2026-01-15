import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import os
import sys

# --- 1. FIX PATHS (The Important Part) ---
# Get the folder where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "nuaa_antispoof.onnx"
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_FILENAME)

# Check if model exists before crashing
if not os.path.exists(MODEL_PATH):
    print(f"\nERROR: Model file not found at: {MODEL_PATH}")
    print("Please check if the file name in the folder matches the code!")
    sys.exit()

print(f"Loading ONNX model from: {MODEL_PATH}...")

try:
    # --- 2. LOAD MODEL ---
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print("Model loaded successfully!")

    # --- 3. SETUP FACEDETECTION ---
    print("Initializing MediaPipe...")
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
    print("MediaPipe ready!")

    # --- 4. RUN WEBCAM ---
    print("Opening Camera... (Check your taskbar if window doesn't appear)")
    cap = cv2.VideoCapture(0) # Try changing to 1 if you have multiple cameras

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Helper function to preprocess image
        def preprocess(img):
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img

        # MediaPipe needs RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Fix boundaries
                x, y = max(0, x), max(0, y)
                w, h = min(w, iw - x), min(h, ih - y)

                face_crop = frame[y:y+h, x:x+w]

                if face_crop.size > 0:
                    try:
                        input_data = preprocess(face_crop)
                        outputs = session.run(None, {input_name: input_data})
                        score = outputs[0][0][0]
                        

                        # LOGIC: 0=Real, 1=Spoof (Update this if your results are flipped)
                        THRESHOLD = 0.5
                        if score > THRESHOLD: 
                            label = f"REAL ({score:.2f})"
                            color = (0, 255, 0) # Green
                        else:
                            label = f"SPOOF ({score:.2f})"
                            color = (0, 0, 255) # Red

                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    except Exception as e:
                        print("Error during prediction:", e)

        cv2.imshow('NUAA Anti-Spoof System', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    print("Try re-running '02_train_nuaa.py' to regenerate the model.")