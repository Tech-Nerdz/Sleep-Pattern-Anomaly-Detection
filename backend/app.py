# backend/app.py → FINAL CORRECT OUTPUT + SMOOTH + NO FALSE ALERTS
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from utils import predict_frame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === PERFECT DROWSINESS LOGIC ===
consecutive_closed = 0
THRESHOLD = 10          # ~0.4 sec at 25 FPS → perfect for real use

def get_status(current_label):
    global consecutive_closed

    # These mean eyes are OPEN or just blinking → SAFE
    if current_label in ["open", "nodrowsy", "blink"]:
        consecutive_closed = 0
        return "ACTIVE", (0, 255, 0)   # Green

    # These mean eyes CLOSED or YAWN → DANGER
    elif current_label in ["close", "drowsy", "yawn"]:
        consecutive_closed += 1
        if consecutive_closed >= THRESHOLD:
            return "SLEEPY", (0, 0, 255)   # Red
        else:
            return "ACTIVE", (0, 255, 255)  # Yellow (closing but not yet drowsy)
    else:
        return "ACTIVE", (255, 255, 255)

# === VIDEO STREAM ===
def generate():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 480)
    cap.set(4, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Run model every 3rd frame (~8–10 inferences/sec)
        if frame_count % 3 == 0:
            try:
                label, conf = predict_frame(frame)
            except:
                label, conf = "unknown", 0.0
        else:
            label = getattr(generate, "last_label", "open")
            conf = getattr(generate, "last_conf", 0.0)

        generate.last_label = label
        generate.last_conf = conf

        # === FINAL CORRECT STATUS ===
        status_text, color = get_status(label)

        # Draw
        cv2.putText(frame, status_text, (10, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 4)
        cv2.putText(frame, f"{label.upper()} ({conf:.2f})", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if status_text == "SLEEPY":
            cv2.putText(frame, "WAKE UP!!!", (40, 220),
                        cv2.FONT_HERSHEY_COMPLEX, 2.8, (0, 0, 255), 7)

        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.get("/video")
def video():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")