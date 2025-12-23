import { useState } from "react";
import "./Reports.css";

export default function Reports() {
  const [showPopup, setShowPopup] = useState(false);

  
  const anomalies = [
    "Drowsiness detected at 10:32 AM",
    "Low EAR detected at 11:05 AM",
    "High blink frequency at 11:45 AM",                 //sample data
  ];

  const historyLogs = [
    "10:00 AM - Normal",
    "10:32 AM - Drowsy",
    "11:05 AM - Alert",
    "11:45 AM - Drowsy",
  ];

  return (
    <div className="reports-container">
      {!showPopup && (
        <button className="open-popup-btn" onClick={() => setShowPopup(true)}>
          Open Reports
        </button>
      )}

      {showPopup && (
        <div className="popup-overlay">
          <div className="popup-box reports-popup">
            <button className="back-btn" onClick={() => setShowPopup(false)}>
              ⬅ Back
            </button>

            <h2 className="popup-title">📋 Reports</h2>
            <div className="report-box animate-fade">
              <h3 className="box-title">⚠ Anomalies Summary</h3>
              <ul>
                {anomalies.map((item, index) => (
                  <li key={index}>
                    <span className="highlight">{item.split(" ")[0]}</span> {item.split(" ").slice(1).join(" ")}
                  </li>
                ))}
              </ul>
            </div>
            <div className="report-box animate-fade delay-1">
              <h3 className="box-title">🕒 History Logs</h3>
              <ul>
                {historyLogs.map((log, index) => (
                  <li key={index}>
                    <span className="highlight">{log.split(" - ")[1]}</span> ({log.split(" - ")[0]})
                  </li>
                ))}
              </ul>
            </div>
            <div className="report-box animate-fade delay-2">
              <h3 className="box-title">💾 Downloadable Report</h3>
              <button className="download-btn">📄 Download PDF</button>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
