import { useEffect, useState } from "react";

function App() {
  const [state, setState] = useState("Loading...");

  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:5000/predict")
        .then(res => res.json())
        .then(data => setState(data.state));
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Sleep Anomaly Detection</h1>

      <img 
        src="http://localhost:5000/video" 
        alt="camera" 
        width="600"
        style={{ borderRadius: "10px" }}
      />

      <h2 style={{ marginTop: "20px" }}>Status: {state}</h2>
    </div>
  );
}

export default App;
