# real_time_eye_detection_pytorch.py

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from config import IMG_SIZE, MODEL_PATH  # Keep your config
# Assuming your PyTorch model class is saved in model.py or here
from model import SleepCNN  # <-- Create this file or paste the class below

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Load the PyTorch model
# -------------------------------
model = SleepCNN(num_classes=5)  # Must match your training script
model.load_state_dict(torch.load(MODEL_PATH.replace(".h5", ".pth"), map_location=device))
# Or if you saved the full model instead of state_dict, use: torch.load(..., map_location=device)
model.to(device)
model.eval()

# -------------------------------
# Preprocessing (same as training)
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class names (change to your actual 5 eye/sleep states)
CLASS_NAMES = ['Closed', 'Open', 'Yawning', 'Drowsy', 'Looking Away']  # Update this!

# -------------------------------
# Prediction function (replaces predict_eye_state)
# -------------------------------
@torch.no_grad()
def predict_eye_state(frame, model, threshold=0.6):
    # Extract face or eye region here if needed (same as your old utils.py)
    # For now, we'll use the whole frame (or crop face like before)
    
    # Convert BGR (OpenCV) -> RGB -> PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Preprocess
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)  # Add batch dim

    # Predict
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, pred_idx = torch.max(probabilities, 1)

    state = CLASS_NAMES[pred_idx.item()], confidence.item()

    return f"{CLASS_NAMES[pred_idx.item()]} ({confidence.item():.2f})"

# -------------------------------
# Real-time loop
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get prediction
    state_text = predict_eye_state(frame, model)

    # Display
    cv2.putText(frame, f"Eye: {state_text}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Real-Time Eye State Detection (PyTorch)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")