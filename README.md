Sleep Pattern Anomaly Detection (Eye Blink Based) :
Sleep Pattern Anomaly Detection is a real-time eye-blink based drowsiness and fatigue monitoring system that uses webcam video to track eye landmarks and detect abnormal sleep behaviour. It continuously analyzes Eye Aspect Ratio (EAR), blink rate, and eye closure duration to identify microsleep, prolonged eye closure, and irregular blink patterns, then raises alerts and logs events for further analysis.​

Features: 
1.Real-time eye and blink tracking using webcam video.​
2.EAR-based detection of drowsiness, microsleep, and abnormal blink patterns.​
3.Visual dashboard showing live video, EAR value, blink count, and drowsiness status.​
4.Graphs for EAR over time and blink frequency for better insight into sleep behaviour.​
5.Alert system with on-screen warnings and anomaly logging, with optional reports.​
6.Suitable for driver drowsiness monitoring, workplace safety, student concentration, and medical sleep analysis.​

Tech Stack :
Frontend:

1.React.js for UI and component-based frontend.​

2.Axios for REST API calls.​

3.Chart.js or Recharts for visualizing EAR and blink rate graphs.​

4.Tailwind CSS or Bootstrap for responsive styling.​

Backend:

1.Python with Flask or FastAPI for REST APIs.​

2.OpenCV for video frame processing.​

3.MediaPipe & PyTorch for facial and eye landmark detection.​

4.NumPy (and optionally SciPy / scikit-learn) for EAR calculation and anomaly logic.​

Database (optional):

- > MongoDB for storing anomaly logs and historical sleep reports.​

Hardware:

Laptop/desktop with webcam or external camera.​

If you want, the next step can be a short “How to run” section for your README (installation, commands, etc.).

# Environment Setup
cd backend
python -m venv env
env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# if run in gpu 
#verify 
nvidia-smi 
nvcc --version
# run in cpu skip above steps

# run command
python train_model.py       # trains & saves model
python app.py               # starts Flask server 
npm start                   # react services