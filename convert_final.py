import numpy as np
import sys
import os
import tf2onnx.convert

# --- 1. FIND THE PATHS AUTOMATICALLY ---
# Get the folder where THIS script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full paths
INPUT_DIR = os.path.join(SCRIPT_DIR, "saved_model_tf")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "nuaa_antispoof.onnx")

print(f"Looking for model in: {INPUT_DIR}")

# Check if it actually exists
if not os.path.exists(INPUT_DIR):
    print("\nCRITICAL ERROR: The folder 'saved_model_tf' was not found!")
    print("Did you run the training script (02_train_nuaa.py) successfully?")
    sys.exit(1)

# --- 2. APPLY NUMPY PATCH ---
print("Applying Numpy Patch...")
try:
    np.bool = bool
    np.object = object
except AttributeError:
    pass

# --- 3. CONVERT ---
print("Starting Conversion...")

# Use the FULL PATHS we calculated above
sys.argv = [
    "tf2onnx.convert",
    "--saved-model", INPUT_DIR,
    "--output", OUTPUT_FILE,
    "--opset", "13"
]

try:
    tf2onnx.convert.main()
    print(f"\nSUCCESS! Model saved to: {OUTPUT_FILE}")
except SystemExit:
    print("\nConversion process finished.")
except Exception as e:
    print(f"\nError: {e}")