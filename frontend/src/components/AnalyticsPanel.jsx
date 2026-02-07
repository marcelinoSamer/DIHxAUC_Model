import { chartImages } from "../data/dashboardData";

const charts = [
  { src: chartImages.menuEngineering, label: "Menu Engineering (BCG Matrix)", desc: "Classification of menu items by popularity and profitability into Stars, Plowhorses, Puzzles, and Dogs." },
  { src: chartImages.pricingAnalysis, label: "Pricing Analysis", desc: "Price distribution and optimization opportunities across the menu." },
  { src: chartImages.clusteringAnalysis, label: "Item Clustering", desc: "K-means clustering of menu items based on performance features." },
  { src: chartImages.campaignAnalysis, label: "Campaign Effectiveness", desc: "Analysis of promotional campaign performance and ROI." },
  { src: chartImages.temporalPatterns, label: "Temporal Patterns", desc: "Order patterns across hours, days, and seasonal trends." },
  { src: chartImages.featureImportance, label: "Feature Importance", desc: "Most important features for demand prediction model." },
  { src: chartImages.modelPerformance, label: "Model Performance", desc: "ML model accuracy metrics and prediction quality." },
  { src: chartImages.inventoryStatus, label: "Inventory Status", desc: "Stock levels, safety stock, and reorder point analysis." },
];

export default function AnalyticsPanel() {
  return (
    <div className="analytics-panel">
      <div className="page-header">
        <h1>Analytics</h1>
        <p>Deep-dive analysis visualizations from the latest data run</p>
      </div>

      <div className="analytics-grid">
        {charts.map((chart) => (
          <div key={chart.label} className="card analytics-card">
            <div className="analytics-card-header">
              <h3>{chart.label}</h3>
              <p>{chart.desc}</p>
            </div>
            <div className="analytics-img-wrapper">
              <img
                src={chart.src}
                alt={chart.label}
                loading="lazy"
                onClick={() => window.open(chart.src, "_blank")}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
