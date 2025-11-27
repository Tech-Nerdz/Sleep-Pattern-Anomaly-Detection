import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import IMG_SIZE

LABELS = ["open", "closed", "blink", "yawn", "drowsy", "sleep"]

def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_state(frame, model):
    processed = preprocess(frame)
    pred = model.predict(processed, verbose=0)[0]
    class_id = int(np.argmax(pred))
    return LABELS[class_id], float(np.max(pred))
