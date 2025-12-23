import React from 'react';

function App() {
  return (
    <div style={{
      background: '#000',
      color: '#fff',
      minHeight: '100vh',
      textAlign: 'center',
      fontFamily: 'Arial',
      padding: '20px'
    }}>
      <h1 style={{fontSize: '3.5rem', color: '#00ff88', marginBottom: '20px'}}>
        Sleep Pattern Anomaly Detection
      </h1>

      <img
        src="http://localhost:8000/video"
        alt="Live Detection"
        style={{
          width: '90%',
          maxWidth: '900px',
          borderRadius: '20px',
          border: '8px solid #00ff88',
          boxShadow: '0 0 30px #00ff88'
        }}
      />

      <div style={{marginTop: '20px', fontSize: '2rem', color: '#ccc'}}>
        Normal blinking → <strong style={{color:'#00ff88'}}>ACTIVE</strong><br/>
        Eyes closed long / yawn → <strong style={{color:'#ff0066'}}>SLEEPY</strong>
      </div>
    </div>
  );
}

export default App;