import { useState, useEffect } from "react";
import { Zap, CheckCircle, XCircle, RefreshCw, Database, Brain } from "lucide-react";
import { getChatConfig, getChatHealth, getChatUsage } from "../services/api";

export default function SettingsPanel() {
  const [config, setConfig] = useState(null);
  const [health, setHealth] = useState(null);
  const [usage, setUsage] = useState(null);

  const fetchStatus = () => {
    getChatConfig().then(setConfig).catch(() => setConfig(null));
    getChatHealth().then(setHealth).catch(() => setHealth(null));
    getChatUsage().then(setUsage).catch(() => setUsage(null));
  };

  useEffect(fetchStatus, []);

  return (
    <div className="settings-panel">
      <div className="page-header">
        <h1>Settings</h1>
        <p>Configure your AI assistant and view system status</p>
      </div>

      <div className="settings-grid">
        {/* LLM Configuration (read-only) */}
        <div className="card settings-card">
          <h3><Brain size={18} /> LLM Configuration</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", margin: "0.5rem 0 1rem" }}>
            The AI assistant is automatically configured from the server environment. No manual setup needed.
          </p>
          <div className="status-grid">
            <div className="status-row">
              <span>Provider</span>
              <span className="status-value">{config?.provider || "—"}</span>
            </div>
            <div className="status-row">
              <span>Model</span>
              <span className="status-value">{config?.model || "—"}</span>
            </div>
            <div className="status-row">
              <span>Temperature</span>
              <span className="status-value">{config?.temperature ?? "—"}</span>
            </div>
            <div className="status-row">
              <span>Max Tokens</span>
              <span className="status-value">{config?.max_tokens || "—"}</span>
            </div>
          
            <div className="status-row">
              <span>Data Context Loaded</span>
              {config?.is_context_loaded ? (
                <span className="status-badge status-optimal"><Database size={14} /> Loaded</span>
              ) : (
                <span className="status-badge status-low">Not loaded</span>
              )}
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="card settings-card">
          <h3><Zap size={18} /> System Status</h3>
          <div className="status-grid">
            <div className="status-row">
              <span>LLM Connection</span>
              {config?.is_configured ? (
                <span className="status-badge status-optimal"><CheckCircle size={14} /> Connected</span>
              ) : (
                <span className="status-badge status-critical"><XCircle size={14} /> Not configured</span>
              )}
            </div>
            {config && (
              <>
                <div className="status-row">
                  <span>Provider</span>
                  <span className="status-value">{config.provider}</span>
                </div>
                <div className="status-row">
                  <span>Model</span>
                  <span className="status-value">{config.model}</span>
                </div>
                <div className="status-row">
                  <span>Temperature</span>
                  <span className="status-value">{config.temperature}</span>
                </div>
                <div className="status-row">
                  <span>Active Sessions</span>
                  <span className="status-value">{config.active_sessions || 0}</span>
                </div>
              </>
            )}
            {health && (
              <div className="status-row">
                <span>Health Check</span>
                {health.ok ? (
                  <span className="status-badge status-optimal"><CheckCircle size={14} /> Healthy</span>
                ) : (
                  <span className="status-badge status-critical"><XCircle size={14} /> {health.error}</span>
                )}
              </div>
            )}
            {usage && (
              <>
                <div className="status-row">
                  <span>Total API Calls</span>
                  <span className="status-value">{usage.total_calls}</span>
                </div>
                <div className="status-row">
                  <span>Total Tokens Used</span>
                  <span className="status-value">{usage.total_tokens?.toLocaleString() || 0}</span>
                </div>
              </>
            )}
          </div>
          <button className="btn btn-outline" onClick={fetchStatus} style={{ marginTop: "1rem" }}>
            <RefreshCw size={14} /> Refresh Status
          </button>
        </div>
      </div>
    </div>
  );
}
