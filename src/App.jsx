import { useState } from "react";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import "./App.css";

export default function App() {
  const [active, setActive] = useState("home");

  return (
    <div className="app-bg">
      <h1 className="title">Sleep Anomaly  Detection</h1>

      <div className="menu-list">
        <div className={`menu-card ${active==="home" && "active"}`} onClick={()=>setActive("home")}>
          🏠 Home
        </div>
        <div className={`menu-card ${active==="dashboard" && "active"}`} onClick={()=>setActive("dashboard")}>
          📊 Dashboard
        </div>
        <div className={`menu-card ${active==="reports" && "active"}`} onClick={()=>setActive("reports")}>
          📁 Reports
        </div>
        <div className={`menu-card ${active==="settings" && "active"}`} onClick={()=>setActive("settings")}>
          ⚙ Settings
        </div>
      </div>

      <div className="content-box">
        {active === "home" && <Home />}
        {active === "dashboard" && <Dashboard />}
        {active === "reports" && <Reports />}
        {active === "settings" && <Settings />}
      </div>
    </div>
  );
}