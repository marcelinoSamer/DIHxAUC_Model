import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import AnalyticsPanel from "./components/AnalyticsPanel";
import InventoryPanel from "./components/InventoryPanel";
import PricingPanel from "./components/PricingPanel";
import ChatPanel from "./components/ChatPanel";
import SettingsPanel from "./components/SettingsPanel";
import "./App.css";

const panels = {
  dashboard: Dashboard,
  analytics: AnalyticsPanel,
  inventory: InventoryPanel,
  pricing: PricingPanel,
  chat: ChatPanel,
  settings: SettingsPanel,
};

export default function App() {
  const [active, setActive] = useState("dashboard");
  const [collapsed, setCollapsed] = useState(false);

  const ActivePanel = panels[active] || Dashboard;

  return (
    <div className="app-layout">
      <Sidebar
        active={active}
        onNavigate={setActive}
        collapsed={collapsed}
        onToggle={() => setCollapsed((c) => !c)}
      />
      <main className={`main-content ${collapsed ? "expanded" : ""}`}>
        <ActivePanel />
      </main>
    </div>
  );
}
