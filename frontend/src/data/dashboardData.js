/**
 * Pre-processed data from /docs analysis results.
 * This serves as the starting point for the dashboard.
 * When a new analysis is run via the backend, this data gets refreshed.
 */

export const executiveSummary = {
  totalOrders: 399810,
  totalOrderItems: 1999341,
  restaurants: 359,
  menuItems: 87713,
  peakHour: 16,
  peakHourLabel: "16:00",
  peakDay: "Friday",
  weekendPct: 26.9,
  avgOrdersPerDay: 399.8,
  avgItemsPerOrder: 1.7,
  avgQuantityPerOrder: 2.3,
  avgOrderValue: 445.44,
  medianOrderValue: 80.0,
  modelMAE: 2.23,
  modelRMSE: 6.77,
  modelR2: 0.622,
  trainSamples: 174530,
  testSamples: 43633,
  trainingTime: 63.37,
  criticalItems: 2130,
  lowStockItems: 3245,
  excessItems: 1604,
};

export const hourlyPatterns = [
  { hour: "12am", orders: 3265, quantity: 8228, revenue: 212453 },
  { hour: "1am", orders: 1998, quantity: 5662, revenue: 122423 },
  { hour: "2am", orders: 1060, quantity: 2678, revenue: 65912 },
  { hour: "3am", orders: 326, quantity: 803, revenue: 20215 },
  { hour: "4am", orders: 56, quantity: 106, revenue: 2863 },
  { hour: "5am", orders: 430, quantity: 1034, revenue: 29103 },
  { hour: "6am", orders: 3048, quantity: 6131, revenue: 210181 },
  { hour: "7am", orders: 7649, quantity: 16058, revenue: 595291 },
  { hour: "8am", orders: 11712, quantity: 25563, revenue: 1001871 },
  { hour: "9am", orders: 16861, quantity: 37138, revenue: 1483445 },
  { hour: "10am", orders: 31005, quantity: 63678, revenue: 2698243 },
  { hour: "11am", orders: 35015, quantity: 74850, revenue: 3120579 },
  { hour: "12pm", orders: 32142, quantity: 68832, revenue: 3035961 },
  { hour: "1pm", orders: 30287, quantity: 63651, revenue: 2955161 },
  { hour: "2pm", orders: 28163, quantity: 58981, revenue: 2935988 },
  { hour: "3pm", orders: 32129, quantity: 70770, revenue: 3154331 },
  { hour: "4pm", orders: 45451, quantity: 115611, revenue: 6937373 },
  { hour: "5pm", orders: 44658, quantity: 116127, revenue: 6937373 },
  { hour: "6pm", orders: 29930, quantity: 74310, revenue: 4431786 },
  { hour: "7pm", orders: 17020, quantity: 38853, revenue: 2095869 },
  { hour: "8pm", orders: 10502, quantity: 25088, revenue: 1024028 },
  { hour: "9pm", orders: 7029, quantity: 18066, revenue: 527719 },
  { hour: "10pm", orders: 5628, quantity: 20148, revenue: 380220 },
  { hour: "11pm", orders: 4444, quantity: 11172, revenue: 296667 },
];

export const featureImportance = [
  { feature: "14-Day Lag Avg", importance: 0.4613 },
  { feature: "Avg Daily Qty", importance: 0.179 },
  { feature: "Std Daily Qty", importance: 0.1707 },
  { feature: "7-Day Lag Avg", importance: 0.1428 },
  { feature: "Day of Week", importance: 0.0176 },
  { feature: "Restaurant Orders", importance: 0.0077 },
  { feature: "Week of Year", importance: 0.0075 },
  { feature: "Avg Order Value", importance: 0.0062 },
  { feature: "Avg Price", importance: 0.0059 },
  { feature: "Month", importance: 0.0008 },
  { feature: "Is Weekend", importance: 0.0005 },
];

export const bcgBreakdown = {
  stars: 12329,
  plowhorses: 13081,
  puzzles: 13119,
  dogs: 11619,
};

export const bcgChartData = [
  { name: "⭐ Stars", value: 12329, color: "#22c55e" },
  { name: "🐴 Plowhorses", value: 13081, color: "#3b82f6" },
  { name: "❓ Puzzles", value: 13119, color: "#f59e0b" },
  { name: "🐕 Dogs", value: 11619, color: "#ef4444" },
];

export const inventoryAlerts = [
  { id: 436334, name: "Lammekølle", avgDailyDemand: 2.71, currentStock: 2.0, status: "critical", daysOfStock: 0.7, reorderPoint: 9, orderQty: 37, price: 414.03 },
  { id: 615278, name: "Grillplatte Ris", avgDailyDemand: 3.65, currentStock: 4.0, status: "critical", daysOfStock: 1.1, reorderPoint: 15, orderQty: 64, price: 185.0 },
  { id: 456067, name: "Hjallerup Special", avgDailyDemand: 1.21, currentStock: 0.5, status: "critical", daysOfStock: 0.4, reorderPoint: 3, orderQty: 54, price: 80.29 },
  { id: 456080, name: "Sultani", avgDailyDemand: 1.44, currentStock: 1.0, status: "critical", daysOfStock: 0.7, reorderPoint: 5, orderQty: 52, price: 102.33 },
  { id: 456081, name: "Meat Lover", avgDailyDemand: 1.5, currentStock: 0.0, status: "critical", daysOfStock: 0.0, reorderPoint: 5, orderQty: 46, price: 140.43 },
  { id: 678655, name: "Sodavand (stor)", avgDailyDemand: 2.26, currentStock: 1.5, status: "critical", daysOfStock: 0.7, reorderPoint: 8, orderQty: 138, price: 22.0 },
  { id: 678648, name: "Ice Kaffe", avgDailyDemand: 1.0, currentStock: 0.0, status: "critical", daysOfStock: 0.0, reorderPoint: 2, orderQty: 64, price: 45.0 },
  { id: 615271, name: "Arabisk Snackkurv", avgDailyDemand: 1.13, currentStock: 0.5, status: "critical", daysOfStock: 0.4, reorderPoint: 3, orderQty: 46, price: 95.0 },
  { id: 532051, name: "HOSO Avocado", avgDailyDemand: 1.5, currentStock: 0.0, status: "critical", daysOfStock: 0.0, reorderPoint: 4, orderQty: 84, price: 40.0 },
  { id: 678624, name: "Mexican", avgDailyDemand: 11.26, currentStock: 5.5, status: "critical", daysOfStock: 0.5, reorderPoint: 34, orderQty: 180, price: 68.0 },
  { id: 456173, name: "Pommes Frites", avgDailyDemand: 4.84, currentStock: 3.0, status: "critical", daysOfStock: 0.6, reorderPoint: 16, orderQty: 162, price: 35.0 },
  { id: 678620, name: "Bagel Menu", avgDailyDemand: 5.45, currentStock: 0.0, status: "critical", daysOfStock: 0.0, reorderPoint: 18, orderQty: 118, price: 80.23 },
  { id: 677717, name: "Hvidlog", avgDailyDemand: 1.63, currentStock: 0.0, status: "critical", daysOfStock: 0.0, reorderPoint: 5, orderQty: 387, price: 2.0 },
  { id: 676980, name: "Cocio", avgDailyDemand: 1.0, currentStock: 0.0, status: "critical", daysOfStock: 0.0, reorderPoint: 2, orderQty: 108, price: 15.75 },
  { id: 676959, name: "1/2 sodavand", avgDailyDemand: 1.92, currentStock: 1.0, status: "critical", daysOfStock: 0.5, reorderPoint: 6, orderQty: 130, price: 21.0 },
];

export const pricingSuggestions = [
  { name: "Chinabox Lille", orders: 143976, price: 50.0, suggested: 56.0, revenue: 7198800, potential: 8062656, gain: 863856 },
  { name: "Øl 40,-", orders: 97839, price: 40.0, suggested: 44.8, revenue: 3913560, potential: 4383187, gain: 469627 },
  { name: "Øl/Vand/Spiritus", orders: 143596, price: 24.0, suggested: 26.88, revenue: 3446304, potential: 3859860, gain: 413556 },
  { name: "ØL", orders: 110748, price: 25.0, suggested: 28.0, revenue: 2768700, potential: 2942800, gain: 332244 },
  { name: "Juice (55)", orders: 49461, price: 55.0, suggested: 61.6, revenue: 2720355, potential: 3046798, gain: 326443 },
  { name: "FAD 1", orders: 52440, price: 50.0, suggested: 56.0, revenue: 2622000, potential: 2936640, gain: 314640 },
  { name: "Shakes", orders: 43932, price: 55.0, suggested: 61.6, revenue: 2416260, potential: 2706211, gain: 289951 },
  { name: "Carlsberg", orders: 36760, price: 55.0, suggested: 61.6, revenue: 2021800, potential: 2264416, gain: 242616 },
];

export const chartImages = {
  executiveDashboard: "/charts/executive_dashboard.png",
  menuEngineering: "/charts/menu_engineering_matrix.png",
  pricingAnalysis: "/charts/pricing_analysis.png",
  campaignAnalysis: "/charts/campaign_analysis.png",
  clusteringAnalysis: "/charts/clustering_analysis.png",
  featureImportance: "/charts/feature_importance.png",
  inventoryStatus: "/charts/inventory_status.png",
  modelPerformance: "/charts/model_performance.png",
  temporalPatterns: "/charts/temporal_patterns.png",
};
