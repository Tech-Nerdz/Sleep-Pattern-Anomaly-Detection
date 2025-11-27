import os

# Base directories
BASE_DIR = "E:/orojects/Sleep-Pattern-Anomaly-Detection/backend"
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Dataset paths
EYES_DATASET = os.path.join(DATASET_DIR, "eyes_open_close")
# BLINK_DATASET = os.path.join(DATASET_DIR, "eye_blink")
DROWSINESS_DATASET = os.path.join(DATASET_DIR, "drowsiness")
# HEAD_FACES_DATASET = os.path.join(DATASET_DIR, "head_faces")

# Model save path
MODEL_PATH = os.path.join(MODEL_DIR, "sleep_anomaly_model.h5")

# Training parameters
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
