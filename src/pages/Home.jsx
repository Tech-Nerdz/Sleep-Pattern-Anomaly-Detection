import { useState } from "react";
import "./Home.css";

export default function Home() {
  const [showDetection, setShowDetection] = useState(false);

  return (
    <div className="home-container">
      <h1 className="home-title">Sleep Detection System</h1>
      <p className="home-intro">
        Monitor your eyes to detect drowsiness in real-time.
      </p>

      <button className="start-btn" onClick={() => setShowDetection(true)}>
        ▶ Start Detection
      </button>

      {showDetection && (
        <div className="popup-overlay">
          <div className="popup-box">
            <button className="back-btn" onClick={() => setShowDetection(false)}>
              ⬅ Back
            </button>

            <h2 className="popup-title">Live Eye Detection</h2>

            <div className="camera-box">
              <video className="camera-preview" autoPlay muted loop>
                <source src="/demo.mp4" type="video/mp4" />
              </video>                                                                     
              <p className="camera-text">Camera preview will appear here</p>
            </div>

            <div className="metrics-box">
              <div className="metric-card">
                <h4>EAR Value</h4>
                <p>0.28</p>
              </div>
              <div className="metric-card">
                <h4>Blink Count</h4>
                <p>12</p>
              </div>
              <div className="metric-card alert">
                <h4>Status</h4>
                <p>ACTIVE</p>
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}

  
