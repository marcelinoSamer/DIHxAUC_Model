import { useState, useEffect, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, AreaChart, Area,
} from "recharts";
import { TrendingUp, ShoppingCart, Store, UtensilsCrossed, AlertTriangle, Package, Users, Clock, Upload, X, Lightbulb, ChevronDown, ChevronUp, Loader2, Wifi, WifiOff, RefreshCw } from "lucide-react";
import { executiveSummary as staticSummary, hourlyPatterns as staticHourlyPatterns, featureImportance as staticFeatureImportance, bcgChartData as staticBcgChartData, chartImages } from "../data/dashboardData.js";
import { getMonthlySuggestions, submitInventoryData, getDashboardData, runAnalysis } from "../services/api.js";

const formatNum = (n) => {
  if (n == null) return "0";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
};

const RADIAN = Math.PI / 180;
const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={12} fontWeight={600}>
      {(percent * 100).toFixed(0)}%
    </text>
  );
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="tooltip-label">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? formatNum(p.value) : p.value}
        </p>
      ))}
    </div>
  );
};

export default function Dashboard() {
  // Dashboard data state (live from API or fallback to static)
  const [dashboardData, setDashboardData] = useState(null);
  const [dataLoading, setDataLoading] = useState(true);
  const [isLive, setIsLive] = useState(false);

  // Use live data or fallback to static
  const executiveSummary = dashboardData?.executiveSummary || staticSummary;
  const hourlyPatterns = dashboardData?.hourlyPatterns || staticHourlyPatterns;
  const featureImportance = dashboardData?.featureImportance || staticFeatureImportance;
  const bcgChartData = dashboardData?.bcgChartData || staticBcgChartData;

  // Compute KPIs from executive summary
  const kpis = useMemo(() => [
    { label: "Total Orders", value: formatNum(executiveSummary.totalOrders), icon: ShoppingCart, color: "#6366f1", sub: `${executiveSummary.avgOrdersPerDay}/day avg` },
    { label: "Menu Items", value: formatNum(executiveSummary.menuItems), icon: UtensilsCrossed, color: "#22c55e", sub: `Across ${executiveSummary.restaurants} restaurants` },
    { label: "Restaurants", value: executiveSummary.restaurants, icon: Store, color: "#f59e0b", sub: "Active locations" },
    { label: "Avg Order Value", value: `${(executiveSummary.avgOrderValue || 0).toFixed(0)} DKK`, icon: TrendingUp, color: "#3b82f6", sub: `Median: ${executiveSummary.medianOrderValue} DKK` },
    { label: "Critical Items", value: formatNum(executiveSummary.criticalItems), icon: AlertTriangle, color: "#ef4444", sub: `${executiveSummary.lowStockItems} low stock` },
    { label: "Model R²", value: (executiveSummary.modelR2 || 0).toFixed(3), icon: Users, color: "#8b5cf6", sub: `MAE: ${executiveSummary.modelMAE}` },
    { label: "Peak Hour", value: executiveSummary.peakHourLabel, icon: Clock, color: "#ec4899", sub: executiveSummary.peakDay },
    { label: "Excess Stock", value: formatNum(executiveSummary.excessItems), icon: Package, color: "#14b8a6", sub: "Overstocked items" },
  ], [executiveSummary]);

  // State for monthly suggestions
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(true);
  const [suggestionsError, setSuggestionsError] = useState(null);

  // State for data input modal
  const [showModal, setShowModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState(null);

  // State for analysis toggle
  const [showAnalysis, setShowAnalysis] = useState(true);

  // State for re-analyze
  const [analyzing, setAnalyzing] = useState(false);

  // Fetch dashboard data on moun
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setDataLoading(true);
        const data = await getDashboardData();
        setDashboardData(data);
        setIsLive(true);
      } catch (err) {
        console.log("Using static dashboard data (backend unavailable)");
        setIsLive(false);
      } finally {
        setDataLoading(false);
      }
    };
    fetchDashboardData();
  }, []);

  // Fetch suggestions on mount
  useEffect(() => {
    const fetchSuggestions = async () => {
      try {
        setSuggestionsLoading(true);
        const data = await getMonthlySuggestions();
        setSuggestions(data || []);
        setSuggestionsError(null);
      } catch (err) {
        setSuggestionsError("Unable to load suggestions. Make sure the backend is running.");
        setSuggestions([]);
      } finally {
        setSuggestionsLoading(false);
      }
    };
    fetchSuggestions();
  }, []);

  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith(".csv")) {
      setSelectedFile(file);
      setSubmitMessage(null);
    } else if (file) {
      setSubmitMessage({ type: "error", text: "Please select a valid CSV file (.csv extension required)." });
      setSelectedFile(null);
    }
  };

  // Handle file submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setSubmitMessage({ type: "error", text: "Please select a CSV file first." });
      return;
    }
    setSubmitting(true);
    setSubmitMessage(null);
    try {
      const result = await submitInventoryData(selectedFile);
      setSubmitMessage({ type: "success", text: `Success! ${result.items_processed || 0} items imported.` });
      setSelectedFile(null);
      // Reset file input
      const fileInput = document.getElementById("csv-file-input");
      if (fileInput) fileInput.value = "";
      // Refresh suggestions after data submission
      const data = await getMonthlySuggestions();
      setSuggestions(data || []);
    } catch (err) {
      // Show actual error message from backend if available
      const errorMessage = err.response?.data?.detail || err.message || "Failed to upload file. Is the backend running?";
      setSubmitMessage({ type: "error", text: errorMessage });
    } finally {
      setSubmitting(false);
    }
  };

  // Handle re-analyze
  const handleReanalyze = async () => {
    setAnalyzing(true);
    try {
      await runAnalysis();
      // Refresh all dashboard data after analysis
      const data = await getDashboardData();
      setDashboardData(data);
      setIsLive(true);
      // Refresh suggestions
      const suggestionsData = await getMonthlySuggestions();
      setSuggestions(suggestionsData || []);
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="dashboard">
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1>Dashboard {dataLoading ? <Loader2 size={20} className="spin" style={{ marginLeft: 8, verticalAlign: "middle" }} /> : null}</h1>
          <p style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            Menu Engineering & Demand Intelligence Overview
            {!dataLoading && (
              <span className={`status-indicator ${isLive ? "live" : "offline"}`}>
                {isLive ? <><Wifi size={12} /> Live</> : <><WifiOff size={12} /> Offline</>}
              </span>
            )}
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button className="btn btn-secondary" onClick={handleReanalyze} disabled={analyzing}>
            {analyzing ? <><Loader2 size={18} className="spin" /> Analyzing...</> : <><RefreshCw size={18} /> Re-analyze</>}
          </button>
          <button className="btn btn-primary" onClick={() => setShowModal(true)}>
            <Upload size={18} /> Import Data
          </button>
        </div>
      </div>

      {/* Data Input Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Add Inventory Data</h3>
              <button className="btn-icon" onClick={() => setShowModal(false)}>
                <X size={18} />
              </button>
            </div>
            <form onSubmit={handleSubmit} className="modal-form">
              <div className="file-upload-area">
                <input
                  type="file"
                  id="csv-file-input"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="file-input"
                />
                <label htmlFor="csv-file-input" className="file-upload-label">
                  <Upload size={24} />
                  <span>{selectedFile ? selectedFile.name : "Choose CSV file or drag & drop"}</span>
                </label>
              </div>
              <div className="file-format-hint">
                <strong>Required columns:</strong> item_id, current_stock, reorder_point, safety_stock
              </div>
              {submitMessage && (
                <div className={`submit-message ${submitMessage.type}`}>
                  {submitMessage.text}
                </div>
              )}
              <button type="submit" className="btn btn-primary" disabled={submitting || !selectedFile} style={{ width: "100%" }}>
                {submitting ? <><Loader2 size={16} className="spin" /> Uploading...</> : "Upload CSV"}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Monthly Suggestions Section */}
      <div className="card suggestions-card">
        <h3><Lightbulb size={18} style={{ color: "#f59e0b" }} /> Weekly Suggestions</h3>
        <p className="chart-subtitle">ML-powered recommendations based on inventory and sales data</p>

        {suggestionsLoading ? (
          <div className="suggestions-loading">
            <Loader2 size={24} className="spin" />
            <span>Loading suggestions...</span>
          </div>
        ) : suggestionsError ? (
          <div className="suggestions-empty">{suggestionsError}</div>
        ) : suggestions.length === 0 ? (
          <div className="suggestions-empty">No suggestions available. Add inventory data to get started.</div>
        ) : (
          <div className="recommendations-list">
            {suggestions.map((rec, idx) => (
              <div key={idx} className={`rec-item ${rec.priority?.toLowerCase() || ""}`}>
                <strong>{rec.type}: {rec.item}</strong>
                <p>{rec.message}</p>
                {rec.priority && <span className="badge">{rec.priority} Priority</span>}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* KPI Grid */}
      <div className="kpi-grid">
        {kpis.map((kpi) => (
          <div key={kpi.label} className="kpi-card">
            <div className="kpi-icon" style={{ background: kpi.color + "22", color: kpi.color }}>
              <kpi.icon size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-value">{kpi.value}</span>
              <span className="kpi-label">{kpi.label}</span>
              <span className="kpi-sub">{kpi.sub}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Row 1 */}
      <div className="charts-row">
        {/* BCG Pie Chart */}
        <div className="card chart-card">
          <h3>BCG Matrix Distribution</h3>
          <p className="chart-subtitle">{(bcgChartData.reduce((s, d) => s + d.value, 0)).toLocaleString()} total items classified</p>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={bcgChartData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={110}
                paddingAngle={3}
                dataKey="value"
                labelLine={false}
                label={renderCustomLabel}
              >
                {bcgChartData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} stroke="transparent" />
                ))}
              </Pie>
              <Legend
                verticalAlign="bottom"
                formatter={(value) => <span style={{ color: "#cbd5e1", fontSize: 13 }}>{value}</span>}
              />
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Orders */}
        <div className="card chart-card wide">
          <h3>Orders by Hour of Day</h3>
          <p className="chart-subtitle">Peak activity at {executiveSummary.peakHourLabel} on {executiveSummary.peakDay}s</p>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={hourlyPatterns}>
              <defs>
                <linearGradient id="orderGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="hour" tick={{ fill: "#94a3b8", fontSize: 11 }} interval={2} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={formatNum} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="orders" stroke="#6366f1" fill="url(#orderGrad)" strokeWidth={2} name="Orders" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="charts-row">
        {/* Feature Importance */}
        <div className="card chart-card wide">
          <h3>Feature Importance (Demand Model)</h3>
          <p className="chart-subtitle">R² = {executiveSummary.modelR2} · MAE = {executiveSummary.modelMAE} · RMSE = {executiveSummary.modelRMSE}</p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={featureImportance} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={(v) => (v * 100).toFixed(0) + "%"} />
              <YAxis type="category" dataKey="feature" tick={{ fill: "#cbd5e1", fontSize: 11 }} width={120} />
              <Tooltip content={<CustomTooltip />} formatter={(v) => (v * 100).toFixed(1) + "%"} />
              <Bar dataKey="importance" radius={[0, 6, 6, 0]} name="Importance">
                {featureImportance.map((_, i) => (
                  <Cell key={i} fill={i < 4 ? "#6366f1" : "#475569"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Revenue by Hour */}
        <div className="card chart-card">
          <h3>Revenue by Hour</h3>
          <p className="chart-subtitle">Total revenue distribution across the day</p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={hourlyPatterns}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="hour" tick={{ fill: "#94a3b8", fontSize: 10 }} interval={3} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={formatNum} />
              <Tooltip content={<CustomTooltip />} formatter={formatNum} />
              <Bar dataKey="revenue" fill="#22c55e" radius={[4, 4, 0, 0]} name="Revenue (DKK)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Analysis Visualizations Section with Toggle */}
      <div className="page-header analysis-header" style={{ marginTop: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <div>
            <h2>Analysis Visualizations</h2>
            <p>Generated from the latest data analysis run</p>
          </div>
        </div>
        <button className="toggle-btn" onClick={() => setShowAnalysis(!showAnalysis)}>
          {showAnalysis ? (
            <><ChevronUp size={18} /> Hide Analysis</>
          ) : (
            <><ChevronDown size={18} /> Show Analysis</>
          )}
        </button>
      </div>

      <div className={`analysis-section ${showAnalysis ? "expanded" : "collapsed"}`}>
        <div className="image-gallery">
          {[
            { src: chartImages.executiveDashboard, label: "Executive Dashboard" },
            { src: chartImages.menuEngineering, label: "Menu Engineering Matrix" },
            { src: chartImages.pricingAnalysis, label: "Pricing Analysis" },
            { src: chartImages.inventoryStatus, label: "Inventory Status" },
            { src: chartImages.modelPerformance, label: "Model Performance" },
            { src: chartImages.temporalPatterns, label: "Temporal Patterns" },
            { src: chartImages.clusteringAnalysis, label: "Clustering Analysis" },
            { src: chartImages.campaignAnalysis, label: "Campaign Analysis" },
          ].map((img) => (
            <div key={img.label} className="card image-card">
              <h4>{img.label}</h4>
              <img src={img.src} alt={img.label} loading="lazy" onClick={() => window.open(img.src, "_blank")} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
