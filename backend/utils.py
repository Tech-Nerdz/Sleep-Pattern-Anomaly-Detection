# utils.py   (PyTorch version – NO TensorFlow!)
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Update these to match your actual 6 classes
LABELS = ["open", "closed", "blink", "yawn", "drowsy", "sleep"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing pipeline (exactly same as during training)
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),                    # from config.py
    transforms.ToTensor(),                          # → [0,1] + CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])   # ImageNet stats (recommended)
])

def predict_state(frame, model):
    """
    Input:
        frame   : OpenCV frame (BGR, uint8)
        model   : Loaded PyTorch model (already on correct device and in eval mode)
    Output:
        label   : string class name
        confidence : float (0~1)
    """
    # BGR → RGB → PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Apply preprocessing → add batch dim → send to device
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)   # shape: (1, 3, H, W)

    # Inference (no grad = faster + less memory)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, class_id = torch.max(probabilities, 1)

    label = LABELS[class_id.item()]
    return label, confidence.item()