# app.py
from flask import Flask, Response, jsonify
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import SleepCNN
from config import MODEL_PATH, IMG_SIZE

app = Flask(__name__)

# -------------------------------
# Device & Model Loading
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on {device}")

model = SleepCNN(num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH.replace(".h5", ".pth"), map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Preprocessing Pipeline
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ['Closed', 'Open', 'Yawning', 'Drowsy', 'Looking Away']

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam!")

# -------------------------------
# Prediction Function
# -------------------------------
@torch.no_grad()
def predict_state(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, pred_idx = torch.max(probabilities, 1)

    label = CLASS_NAMES[pred_idx.item()]
    score = confidence.item()

    return label, round(score, 4)

# -------------------------------
# API Routes
# -------------------------------
@app.route("/predict")
def predict():
    ret, frame = camera.read()
    if not ret:
        return jsonify({"error": "Failed to grab frame"}), 500

    label, score = predict_state(frame)
    return jsonify({
        "state": label,
        "confidence": score
    })

@app.route("/video")
def video():
    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break

            label, score = predict_state(frame)
            text = f"{label}: {score:.2f}"

            # FIXED cv2.putText
            cv2.putText(frame, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 0), 3)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def index():
    return """
    <h1>Sleep/Eye State Detection - PyTorch</h1>
    <h3>Endpoints:</h3>
    <ul>
        <li><a href="/video">/video</a> → Live webcam stream</li>
        <li><a href="/predict">/predict</a> → JSON prediction</li>
    </ul>
    <img src="/video" width="640">
    """

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        camera.release()
