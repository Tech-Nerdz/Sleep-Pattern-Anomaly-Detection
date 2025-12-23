import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";
import "./Dashboard.css";

export default function Dashboard() {
  const [showPopup, setShowPopup] = useState(false); 

  const chartData = [
    { time: "1s", ear: 0.30, blink: 1 },
    { time: "2s", ear: 0.28, blink: 2 },
    { time: "3s", ear: 0.26, blink: 3 },
    { time: "4s", ear: 0.24, blink: 1 }
  ];

  const [ear, setEar] = useState(0.28);
  const [blink, setBlink] = useState(12);
  const [alert, setAlert] = useState(false);

  useEffect(() => {
    
  }, []);

  return (
    <div className="dashboard-wrapper">
      {!showPopup && (
        <button className="open-dashboard-btn" onClick={() => setShowPopup(true)}>
          Open Live Dashboard
        </button>                                                                     //axios
      )}

      {showPopup && (
        <div className="popup-overlay">
          <div className="popup-box dashboard-popup">

            <button className="back-btn" onClick={() => setShowPopup(false)}>
              ⬅ Back
            </button>

            <h2 className="popup-title">Live Detection Dashboard</h2>

            <div className="camera-box">
              <video className="camera-preview" autoPlay muted loop>
                <source src="/demo.mp4" type="video/mp4"/>
              </video>
              <p className="camera-text">Live camera preview</p>
            </div>

            <div className="metrics-box">
              <div className="metric-card">
                <h4>EAR Value</h4>
                <p>{ear}</p>
              </div>
              <div className="metric-card">
                <h4>Blink Count</h4>
                <p>{blink}</p>
              </div>
              <div className={`metric-card ${alert ? "alert-red" : "alert-green"}`}>
                <h4>Status</h4>
                <p>{alert ? "DROWSY" : "ACTIVE"}</p>
              </div>
            </div>

            <div className="chart-box">
              <h3 className="chart-title">EAR Trend</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <XAxis dataKey="time" stroke="#94a3b8"/>
                  <YAxis stroke="#94a3b8"/>
                  <Tooltip/>
                  <Line dataKey="ear" stroke="#38bdf8" strokeWidth={3}/>
                </LineChart>
              </ResponsiveContainer>

              <h3 className="chart-title mt-4">Blink Trend</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <XAxis dataKey="time" stroke="#94a3b8"/>
                  <YAxis stroke="#94a3b8"/>
                  <Tooltip/>
                  <Line dataKey="blink" stroke="#facc15" strokeWidth={3}/>
                </LineChart>
              </ResponsiveContainer>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}

