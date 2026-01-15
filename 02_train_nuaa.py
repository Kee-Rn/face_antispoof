import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, 'dataset', 'train')
# We save to a FOLDER, not a file
SAVED_MODEL_FOLDER = os.path.join(SCRIPT_DIR, 'saved_model_tf') 

if not os.path.exists(TRAIN_DIR):
    print("CRITICAL ERROR: Dataset not found!")
    exit()

# --- DATA ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    shuffle=True
)

# --- MODEL ---
print("Building LivenessNet...")
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# --- TRAIN ---
print("Starting Training...")
model.fit(train_generator, epochs=15)

# --- SAVE TO FOLDER ---
print(f"\n--- Saving model to folder: {SAVED_MODEL_FOLDER} ---")
if os.path.exists(SAVED_MODEL_FOLDER):
    shutil.rmtree(SAVED_MODEL_FOLDER)
    
tf.saved_model.save(model, SAVED_MODEL_FOLDER)
print("SUCCESS! Training finished. Now run the conversion command in the terminal.")