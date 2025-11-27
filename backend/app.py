from flask import Flask, Response, jsonify
import cv2
from tensorflow.keras.models import load_model
from utils import predict_state

app = Flask(__name__)

model = load_model("models/sleep_model.h5")
camera = cv2.VideoCapture(0)

@app.route("/predict")
def predict():
    ret, frame = camera.read()
    label, score = predict_state(frame, model)
    return jsonify({"state": label, "confidence": score})

@app.route("/video")
def video():
    def gen():
        while True:
            ret, frame = camera.read()
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(host="0.0.0.0", port=5000)
