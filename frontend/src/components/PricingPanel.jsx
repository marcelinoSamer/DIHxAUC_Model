import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { TrendingUp, DollarSign, ArrowUpRight } from "lucide-react";
import { pricingSuggestions } from "../data/dashboardData";

const formatNum = (n) => {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
};

const totalGain = pricingSuggestions.reduce((s, p) => s + p.gain, 0);
const avgIncrease = pricingSuggestions.reduce((s, p) => s + ((p.suggested - p.price) / p.price) * 100, 0) / pricingSuggestions.length;

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <p className="tooltip-label">{d.name}</p>
      <p>Current: {d.price} DKK</p>
      <p>Suggested: {d.suggested} DKK</p>
      <p style={{ color: "#22c55e" }}>Gain: +{formatNum(d.gain)} DKK</p>
    </div>
  );
};

export default function PricingPanel() {
  return (
    <div className="pricing-panel">
      <div className="page-header">
        <h1>Pricing Optimization</h1>
        <p>Data-driven pricing suggestions to maximize revenue</p>
      </div>

      {/* KPI Row */}
      <div className="pricing-kpis">
        <div className="kpi-card">
          <div className="kpi-icon" style={{ background: "#22c55e22", color: "#22c55e" }}>
            <DollarSign size={22} />
          </div>
          <div className="kpi-content">
            <span className="kpi-value">{formatNum(totalGain)} DKK</span>
            <span className="kpi-label">Potential Revenue Gain</span>
            <span className="kpi-sub">From top {pricingSuggestions.length} items</span>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon" style={{ background: "#6366f122", color: "#6366f1" }}>
            <TrendingUp size={22} />
          </div>
          <div className="kpi-content">
            <span className="kpi-value">{avgIncrease.toFixed(1)}%</span>
            <span className="kpi-label">Avg Price Increase</span>
            <span className="kpi-sub">Conservative adjustments</span>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon" style={{ background: "#f59e0b22", color: "#f59e0b" }}>
            <ArrowUpRight size={22} />
          </div>
          <div className="kpi-content">
            <span className="kpi-value">{pricingSuggestions.length}</span>
            <span className="kpi-label">Items to Adjust</span>
            <span className="kpi-sub">Highest impact suggestions</span>
          </div>
        </div>
      </div>

      {/* Revenue Gain Chart */}
      <div className="card chart-card">
        <h3>Revenue Gain Potential by Item</h3>
        <p className="chart-subtitle">Top items sorted by potential revenue increase</p>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={pricingSuggestions} margin={{ left: 10, right: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} angle={-30} textAnchor="end" height={80} />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={formatNum} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="gain" radius={[6, 6, 0, 0]} name="Revenue Gain">
              {pricingSuggestions.map((_, i) => (
                <Cell key={i} fill={i < 3 ? "#22c55e" : "#6366f1"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Pricing Table */}
      <div className="card table-card">
        <h3>Pricing Suggestions</h3>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Item</th>
                <th>Orders</th>
                <th>Current Price</th>
                <th>Suggested Price</th>
                <th>Change</th>
                <th>Current Revenue</th>
                <th>Potential Revenue</th>
                <th>Revenue Gain</th>
              </tr>
            </thead>
            <tbody>
              {pricingSuggestions.map((item, i) => (
                <tr key={i}>
                  <td className="item-name">{item.name}</td>
                  <td>{item.orders.toLocaleString()}</td>
                  <td>{item.price.toFixed(0)} DKK</td>
                  <td className="text-primary">{item.suggested.toFixed(1)} DKK</td>
                  <td className="text-green">+{((item.suggested - item.price) / item.price * 100).toFixed(0)}%</td>
                  <td>{formatNum(item.revenue)} DKK</td>
                  <td>{formatNum(item.potential)} DKK</td>
                  <td className="text-green">+{formatNum(item.gain)} DKK</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
