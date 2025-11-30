import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from config import IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH

DATASET_DIR = "datasets/sleep_dataset"

# -----------------------------------
# Preprocessing
# -----------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=8,
    zoom_range=0.1,
    horizontal_flip=True
)

train = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# -----------------------------------
# CNN MODEL (Fixed Input Shape)
# -----------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(5, activation='softmax')   # EXACTLY 5 classes
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------
# TRAIN
# -----------------------------------
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS
)

# -----------------------------------
# SAVE
# -----------------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)

print("🔥 Unified Sleep Anomaly Model Saved:", MODEL_PATH)
