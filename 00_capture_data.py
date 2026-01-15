import cv2
import os
import time

# Create folders
os.makedirs("dataset/train/real", exist_ok=True)
os.makedirs("dataset/train/spoof", exist_ok=True)

cap = cv2.VideoCapture(0)

print("--------------------------------------------------")
print("INSTRUCTIONS:")
print("1. Sit normally. Press 'r' repeatedly to save REAL faces.")
print("   (Move your head slightly, change expression)")
print("2. Hold your phone with your photo. Press 's' to save SPOOF faces.")
print("   (Move the phone around, change angles)")
print("3. Aim for 50-100 images of each.")
print("4. Press 'q' to quit.")
print("--------------------------------------------------")

count_real = len(os.listdir("dataset/train/real"))
count_spoof = len(os.listdir("dataset/train/spoof"))

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.putText(frame, f"Real: {count_real} | Spoof: {count_spoof}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Data Collector", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Save Real
        filename = f"dataset/train/real/real_{int(time.time()*1000)}.jpg"
        cv2.imwrite(filename, frame)
        count_real += 1
        print(f"Saved Real: {count_real}")
    elif key == ord('s'):
        # Save Spoof
        filename = f"dataset/train/spoof/spoof_{int(time.time()*1000)}.jpg"
        cv2.imwrite(filename, frame)
        count_spoof += 1
        print(f"Saved Spoof: {count_spoof}")

cap.release()
cv2.destroyAllWindows()