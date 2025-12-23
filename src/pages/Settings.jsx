import { useState } from "react";
import "./Settings.css";

export default function Settings() {
  const [showPopup, setShowPopup] = useState(false);

  const [earThreshold, setEarThreshold] = useState(0.25);
  const [blinkSensitivity, setBlinkSensitivity] = useState(12);
  const [alertEnabled, setAlertEnabled] = useState(true);

  return (
    <div className="settings-container">
      {!showPopup && (
        <button className="open-popup-btn" onClick={() => setShowPopup(true)}>
          Open Settings
        </button>
      )}

      {showPopup && (
        <div className="popup-overlay">
          <div className="popup-box settings-popup animate-slide-up">

  
            <button className="back-btn" onClick={() => setShowPopup(false)}>
              ⬅ Back
            </button>

            <h2 className="popup-title">⚙ Settings</h2>
            <div className="settings-box animate-fade">
              <h3>EAR Threshold</h3>
              <input
                type="number"
                step="0.01"
                min="0.1"
                max="0.5"
                value={earThreshold}
                onChange={(e) => setEarThreshold(e.target.value)}
              />
              <p className="small-text">Set the eye aspect ratio threshold</p>
            </div>
            <div className="settings-box animate-fade delay-1">
              <h3>Blink Sensitivity</h3>
              <input
                type="number"
                min="5"
                max="30"
                value={blinkSensitivity}
                onChange={(e) => setBlinkSensitivity(e.target.value)}
              />
              <p className="small-text">Set max blink count before alert</p>
            </div>

            <div className="settings-box animate-fade delay-2">
              <h3>Alert Settings</h3>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={alertEnabled}
                  onChange={(e) => setAlertEnabled(e.target.checked)}
                />
                <span className="slider"></span>
              </label>
              <p className="small-text">Toggle on/off alerts for drowsiness</p>
            </div>

            <button className="save-btn" onClick={() => alert("Settings saved!")}>
              💾 Save Settings
            </button>

          </div>
        </div>
      )}
    </div>
  );
}
