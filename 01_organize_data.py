import os
import shutil
import random

# --- CONFIGURATION ---
# Where your unzipped NUAA folders are currently located
SOURCE_REAL = "ClientRaw"      # Real Faces
SOURCE_SPOOF = "ImposterRaw"   # Fake/Photo Faces

# Where we want to move them for training
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Split ratio (80% for training, 20% for testing)
SPLIT_SIZE = 0.8

def create_structure():
    # Create dataset/train/real, dataset/train/spoof, etc.
    for root in [TRAIN_DIR, TEST_DIR]:
        for label in ["real", "spoof"]:
            os.makedirs(os.path.join(root, label), exist_ok=True)

def move_files(source_folder, label):
    print(f"Processing {label} images from {source_folder}...")
    
    # Get all image files
    all_files = []
    for root_dir, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_files.append(os.path.join(root_dir, file))

    # Shuffle them so we get random selection
    random.shuffle(all_files)

    # Calculate split index
    split_point = int(len(all_files) * SPLIT_SIZE)
    train_files = all_files[:split_point]
    test_files = all_files[split_point:]

    # Copy files
    for f in train_files:
        shutil.copy(f, os.path.join(TRAIN_DIR, label))
    
    for f in test_files:
        shutil.copy(f, os.path.join(TEST_DIR, label))

    print(f"Done! {len(train_files)} -> Train, {len(test_files)} -> Test")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_REAL) or not os.path.exists(SOURCE_SPOOF):
        print("ERROR: Could not find 'ClientRaw' or 'ImposterRaw' folders.")
        print("Make sure you unzipped the NUAA dataset in this same folder!")
    else:
        create_structure()
        move_files(SOURCE_REAL, "real")
        move_files(SOURCE_SPOOF, "spoof")
        print("\nData organized successfully! You can now run the training script.")