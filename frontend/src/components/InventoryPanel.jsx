import { useState } from "react";
import { AlertTriangle, ArrowUpDown, Search, ChevronDown, ChevronUp } from "lucide-react";
import { inventoryAlerts } from "../data/dashboardData.js";
import { executiveSummary } from "../data/dashboardData.js";

const statusConfig = {
  critical: { label: "🔴 Critical", className: "status-critical", color: "#ef4444" },
  low: { label: "🟠 Low Stock", className: "status-low", color: "#f59e0b" },
  optimal: { label: "🟢 Optimal", className: "status-optimal", color: "#22c55e" },
  excess: { label: "🔵 Excess", className: "status-excess", color: "#3b82f6" },
  high: { label: "🟡 High", className: "status-high", color: "#eab308" },
};

export default function InventoryPanel() {
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState("daysOfStock");
  const [sortDir, setSortDir] = useState("asc");

  const filtered = inventoryAlerts
    .filter((item) => item.name.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const va = a[sortField] ?? 0;
      const vb = b[sortField] ?? 0;
      return sortDir === "asc" ? va - vb : vb - va;
    });

  const toggleSort = (field) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  const SortIcon = ({ field }) => {
    if (sortField !== field) return <ArrowUpDown size={12} className="sort-icon muted" />;
    return sortDir === "asc" ? <ChevronUp size={12} className="sort-icon" /> : <ChevronDown size={12} className="sort-icon" />;
  };

  return (
    <div className="inventory-panel">
      <div className="page-header">
        <h1>Inventory Alerts</h1>
        <p>Real-time stock monitoring and reorder recommendations</p>
      </div>

      {/* Summary Cards */}
      <div className="inv-summary">
        <div className="inv-stat-card critical">
          <AlertTriangle size={24} />
          <div>
            <span className="inv-stat-value">{executiveSummary.criticalItems.toLocaleString()}</span>
            <span className="inv-stat-label">Critical Items</span>
          </div>
        </div>
        <div className="inv-stat-card warning">
          <AlertTriangle size={24} />
          <div>
            <span className="inv-stat-value">{executiveSummary.lowStockItems.toLocaleString()}</span>
            <span className="inv-stat-label">Low Stock</span>
          </div>
        </div>
        <div className="inv-stat-card info">
          <AlertTriangle size={24} />
          <div>
            <span className="inv-stat-value">{executiveSummary.excessItems.toLocaleString()}</span>
            <span className="inv-stat-label">Excess Stock</span>
          </div>
        </div>
      </div>

      {/* Search */}
      <div className="card table-card">
        <div className="table-header">
          <h3>Critical Stock Items</h3>
          <div className="search-box">
            <Search size={16} />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search items..."
            />
          </div>
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Item</th>
                <th onClick={() => toggleSort("avgDailyDemand")} className="sortable">
                  Daily Demand <SortIcon field="avgDailyDemand" />
                </th>
                <th onClick={() => toggleSort("currentStock")} className="sortable">
                  Current Stock <SortIcon field="currentStock" />
                </th>
                <th onClick={() => toggleSort("daysOfStock")} className="sortable">
                  Days Left <SortIcon field="daysOfStock" />
                </th>
                <th onClick={() => toggleSort("reorderPoint")} className="sortable">
                  Reorder Point <SortIcon field="reorderPoint" />
                </th>
                <th onClick={() => toggleSort("orderQty")} className="sortable">
                  Order Qty <SortIcon field="orderQty" />
                </th>
                <th onClick={() => toggleSort("price")} className="sortable">
                  Price (DKK) <SortIcon field="price" />
                </th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => {
                const sc = statusConfig[item.status] || statusConfig.critical;
                return (
                  <tr key={item.id}>
                    <td className="item-name">{item.name}</td>
                    <td>{item.avgDailyDemand.toFixed(1)}</td>
                    <td className={item.currentStock === 0 ? "text-danger" : ""}>{item.currentStock}</td>
                    <td className={item.daysOfStock < 1 ? "text-danger" : item.daysOfStock < 3 ? "text-warning" : ""}>
                      {item.daysOfStock.toFixed(1)}
                    </td>
                    <td>{item.reorderPoint}</td>
                    <td className="text-primary">{item.orderQty}</td>
                    <td>{item.price.toFixed(0)}</td>
                    <td>
                      <span className={`status-badge ${sc.className}`}>{sc.label}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
