# backend/utils.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2

# Your exact model architecture (MUST match train_model.py)
class SleepCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Settings
IMG_SIZE = (128, 128)
class_names = ['blink', 'close', 'drowsy', 'nodrowsy', 'open', 'yawn']

# Preprocessing (exact same as training!)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SleepCNN(num_classes=6).to(device)
model.load_state_dict(torch.load("models/sleep_model.pth", map_location=device))
model.eval()

print(f"[utils.py] Your trained model loaded successfully on {device}")

def predict_frame(frame):
    """Input: OpenCV frame (BGR), Output: label, confidence"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = class_names[predicted.item()]
    conf = confidence.item()
    return label, conf