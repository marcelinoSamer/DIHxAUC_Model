import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, AreaChart, Area,
} from "recharts";
import { TrendingUp, ShoppingCart, Store, UtensilsCrossed, AlertTriangle, Package, Users, Clock } from "lucide-react";
import { executiveSummary, hourlyPatterns, featureImportance, bcgChartData, chartImages } from "../data/dashboardData.js";

const formatNum = (n) => {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
};

const kpis = [
  { label: "Total Orders", value: formatNum(executiveSummary.totalOrders), icon: ShoppingCart, color: "#6366f1", sub: `${executiveSummary.avgOrdersPerDay}/day avg` },
  { label: "Menu Items", value: formatNum(executiveSummary.menuItems), icon: UtensilsCrossed, color: "#22c55e", sub: `Across ${executiveSummary.restaurants} restaurants` },
  { label: "Restaurants", value: executiveSummary.restaurants, icon: Store, color: "#f59e0b", sub: "Active locations" },
  { label: "Avg Order Value", value: `${executiveSummary.avgOrderValue.toFixed(0)} DKK`, icon: TrendingUp, color: "#3b82f6", sub: `Median: ${executiveSummary.medianOrderValue} DKK` },
  { label: "Critical Items", value: formatNum(executiveSummary.criticalItems), icon: AlertTriangle, color: "#ef4444", sub: `${executiveSummary.lowStockItems} low stock` },
  { label: "Model R²", value: executiveSummary.modelR2.toFixed(3), icon: Users, color: "#8b5cf6", sub: `MAE: ${executiveSummary.modelMAE}` },
  { label: "Peak Hour", value: executiveSummary.peakHourLabel, icon: Clock, color: "#ec4899", sub: executiveSummary.peakDay },
  { label: "Excess Stock", value: formatNum(executiveSummary.excessItems), icon: Package, color: "#14b8a6", sub: "Overstocked items" },
];

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
  return (
    <div className="dashboard">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Menu Engineering & Demand Intelligence Overview</p>
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

      {/* Analysis Charts Gallery */}
      <div className="page-header" style={{ marginTop: "1rem" }}>
        <h2>Analysis Visualizations</h2>
        <p>Generated from the latest data analysis run</p>
      </div>

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
  );
}
