# 🍽️ FlavorFlow Craft: AI-Powered Menu Intelligence & Inventory Optimization Platform

> **Deloitte x AUC Hackathon 2024-2025 - Menu Engineering & Inventory Challenge**

Transform restaurant operations from gut instinct to data-driven insights. FlavorFlow Craft analyzes 2M+ order transactions to optimize menus, forecast demand, and prevent stockouts/overstocking.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Results & Insights](#-results--insights)
- [Team Members](#-team-members)

---

## 🎯 Problem Statement

FlavorFlow restaurants have rich transactional data but struggle with:
- **Menu Engineering**: Which items are profitable vs. popular?
- **Inventory Management**: Overstocking waste vs. stockout lost sales
- **Demand Forecasting**: Predicting customer behavior patterns
- **Customer Insights**: Understanding purchasing patterns and timing

**This isn't just operational inefficiency; it's millions in lost revenue and wasted inventory.**

---

## 💡 Solution Overview

**FlavorFlow Craft Intelligence Platform** provides two complementary analyses:

### 1. Menu Engineering (BCG Matrix Analysis)
- Classifies menu items using traditional BCG methodology
- Identifies Stars, Plowhorses, Puzzles, and Dogs
- Provides strategic recommendations for each category

### 2. Inventory Optimization (Order-Level Analysis)
- **Trains on 2M+ order transactions** (not aggregated data)
- Forecasts demand using temporal patterns and ML models
- Calculates optimal inventory levels with safety stock
- Generates actionable alerts for restocking decisions

---

## ✨ Features

### 1. Menu Engineering (BCG Matrix)
![Menu Engineering Matrix](docs/menu_engineering_matrix.png)

| Category | Popularity | Profitability | Recommended Action |
|----------|------------|---------------|-------------------|
| ⭐ **Stars** | High | High | Promote heavily, protect margins |
| 🐴 **Plowhorses** | High | Low | Re-engineer pricing (+10-15%) |
| ❓ **Puzzles** | Low | High | Increase visibility, marketing |
| 🐕 **Dogs** | Low | Low | Bundle, re-engineer, or remove |

### 2. Inventory Optimization & Demand Forecasting
![Executive Dashboard](docs/executive_dashboard.png)

- **Demand Forecasting**: ML models trained on 2M+ order transactions
- **Customer Behavior Analysis**: Temporal patterns, purchase frequency, co-purchasing
- **Inventory Optimization**: Safety stock, reorder points, EOQ calculations
- **Smart Alerts**: Critical stockouts, low stock warnings, excess inventory alerts

### 3. Advanced Analytics
- **Temporal Analysis**: Hourly/daily/monthly demand patterns
- **Feature Importance**: What drives demand (price, seasonality, trends)
- **Restaurant-Level Insights**: Performance across 359 locations
- **Executive Dashboards**: 4-panel summary with key metrics

### 4. Pricing & Campaign Analysis
![Pricing Analysis](docs/pricing_analysis.png)

- Price distribution and elasticity analysis
- Campaign effectiveness tracking
- ROI optimization recommendations

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Statistical Analysis** | SciPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **API Framework** | FastAPI |
| **Testing** | pytest |
| **Dimensionality Reduction** | PCA |
| **Clustering** | KMeans |

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-team/flavorflow-craft.git
cd flavorflow-craft
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option 1: Run Inventory Optimization (Recommended)

```bash
# Run inventory optimization analysis (trains on 2M+ order transactions)
python -m src.main inventory

# Or run with custom paths
python -m src.main inventory --data-dir data/ --output-dir docs/
```

**What it does:**
- Trains demand forecasting models on order-level data
- Analyzes customer behavior patterns
- Calculates optimal inventory levels
- Generates executive dashboards and alerts

### Option 2: Run Menu Engineering Analysis

```bash
# Run traditional BCG matrix analysis
python -m src.main analyze

# Or run with custom paths
python -m src.main analyze --data-dir data/ --output-dir docs/
```

### Option 3: Start API Server

```bash
# Start the FastAPI server
python -m src.main serve --port 8000

# With auto-reload for development
python -m src.main serve --reload
```

Then visit: http://localhost:8000/docs for interactive API documentation.

### Option 4: Run Jupyter Notebook

1. Open `main.ipynb` in VS Code or Jupyter
2. Select your Python environment (`venv`)
3. Run all cells

---

## 📡 API Documentation

Once the server is running, access the interactive docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/initialize` | POST | Load data and initialize service |
| `/analyze` | POST | Run full BCG analysis |
| `/items` | GET | Get classified menu items |
| `/recommendations` | GET | Get strategic recommendations |
| `/pricing-suggestions` | GET | Get pricing optimization suggestions |
| `/ask` | POST | Ask business questions (AI) |

### Example API Usage

```python
import requests

# Initialize the service
requests.post("http://localhost:8000/initialize?data_dir=data/")

# Run analysis
response = requests.post("http://localhost:8000/analyze", json={
    "include_predictions": True,
    "include_clustering": True
})
print(response.json())

# Ask a business question
response = requests.post("http://localhost:8000/ask", json={
    "question": "What are my best performing menu items?"
})
print(response.json()["answer"])
```

---

## 🏗️ Project Structure

```
flavorflow-craft/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── main.ipynb                         # Interactive analysis notebook
│
├── data/                              # Source data files
│   ├── dim_items.csv                 # Item catalog (87k items)
│   ├── dim_menu_items.csv            # Menu configurations
│   ├── dim_places.csv                # Restaurant locations (359)
│   ├── fct_orders.csv                # Order transactions (400k)
│   ├── fct_order_items.csv           # Order line items (2M+)
│   ├── fct_campaigns.csv             # Marketing campaigns
│   └── most_ordered.csv              # Aggregated orders
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── main.py                       # CLI entry point
│   ├── config.py                     # Configuration
│   │
│   ├── models/                       # ML model classes
│   │   ├── __init__.py
│   │   ├── demand_predictor.py      # Legacy demand forecasting
│   │   ├── demand_forecaster.py     # NEW: Order-level forecasting
│   │   ├── item_clusterer.py        # K-Means clustering
│   │   ├── menu_classifier.py       # BCG Matrix classifier
│   │   ├── inventory_optimizer.py   # NEW: Safety stock, EOQ
│   │   └── customer_analyzer.py     # NEW: Behavior patterns
│   │
│   ├── services/                     # Business logic services
│   │   ├── __init__.py
│   │   ├── menu_analysis_service.py # Legacy BCG analysis
│   │   ├── inventory_analysis_service.py # NEW: Full inventory pipeline
│   │   ├── inventory_visualizations.py  # NEW: Dashboard generation
│   │   └── reporting_service.py     # Report generation
│   │
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   ├── helpers.py               # General helpers
│   │   └── data_loader.py           # Data loading
│   │
│   ├── api/                          # REST API (FastAPI)
│   │   ├── __init__.py
│   │   ├── main.py                  # API endpoints
│   │   └── schemas.py               # Pydantic models
│   │
│   └── (legacy modules)              # Original functional modules
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_helpers.py
│   ├── test_models.py
│   └── test_api.py
│
└── docs/                              # Generated outputs
    ├── executive_dashboard.png       # NEW: 4-panel summary
    ├── temporal_patterns.png         # NEW: Demand patterns
    ├── inventory_status.png          # NEW: Stock distribution
    ├── feature_importance.png        # NEW: Model insights
    ├── menu_engineering_matrix.png   # BCG analysis
    ├── pricing_analysis.png
    ├── clustering_analysis.png
    ├── inventory_alerts.csv          # NEW: Actionable alerts
    ├── inventory_analysis.csv        # NEW: Full analysis
    └── *.csv                         # Analysis results
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_helpers.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Results & Insights

### Inventory Optimization Results (Primary Analysis)

**Training on 2M+ Order Transactions:**
- **Orders Processed**: 399,810 transactions
- **Order Items**: 1,999,341 line items
- **Restaurants**: 359 locations
- **Menu Items**: 87,713 total catalog

**Demand Forecasting Model Performance:**
- **Algorithm**: Gradient Boosting Regressor
- **Training Time**: 64.6 seconds
- **MAE**: 2.23 units (Mean Absolute Error)
- **RMSE**: 6.77 units (Root Mean Square Error)
- **R² Score**: 0.6220 (62.2% variance explained)

**Top Predictive Features:**
1. **14-Day Rolling Average** (46.1%) - Recent demand trends
2. **Historical Daily Average** (17.9%) - Baseline demand
3. **Demand Variability** (17.1%) - Standard deviation
4. **7-Day Rolling Average** (14.3%) - Weekly patterns

**Inventory Alerts Generated:**
- **🔴 Critical Stockouts**: 2,130 items need immediate restocking
- **🟠 Low Stock Warnings**: 3,245 items need reordering soon
- **🔵 Excess Inventory**: 1,604 items are overstocked

### Customer Behavior Insights

**Temporal Patterns:**
- **Peak Hour**: 16:00 (4 PM) - Dinner rush
- **Peak Day**: Friday - Weekend demand
- **Weekend Orders**: 26.9% of total volume
- **Average Daily Orders**: 399.8 across network

**Purchase Patterns:**
- **Average Items per Order**: 1.7 items
- **Average Quantity per Order**: 2.3 units
- **Average Order Value**: 445.44 DKK
- **Median Order Value**: 80.00 DKK

### Menu Engineering Results (Legacy Analysis)

**BCG Matrix Classification:**
- **⭐ Stars** (High Popularity, High Profit): ~25% of items
- **🐴 Plowhorses** (High Popularity, Low Profit): ~30% of items
- **❓ Puzzles** (Low Popularity, High Profit): ~20% of items
- **🐕 Dogs** (Low Popularity, Low Profit): ~25% of items

**Pricing Opportunities:**
- Median price point: ~75 DKK
- 15-20% of items are underpriced by 10-15%
- Price elasticity highest in mid-range items

### Expected Business Impact

| Metric | Expected Improvement |
|--------|---------------------|
| **Inventory Turnover** | +25-35% |
| **Stockout Reduction** | -60-80% |
| **Waste Reduction** | -15-25% |
| **Revenue** | +8-15% |
| **Margin** | +5-10% |
| **Campaign ROI** | 2-3x |

---

## 🚀 Innovative Features

### 1. Dual Analysis Engine
- **Inventory Optimization**: Trains on 2M+ order transactions for demand forecasting
- **Menu Engineering**: Traditional BCG matrix analysis for strategic planning
- **Unified Platform**: Both analyses run from single codebase

### 2. Order-Level Machine Learning
- **Real Transaction Training**: Models learn from actual customer behavior, not aggregates
- **Temporal Features**: Incorporates time-based patterns (hourly, daily, seasonal)
- **Restaurant-Specific**: Location-aware demand forecasting across 359 restaurants

### 3. Smart Inventory Management
- **Safety Stock Calculation**: Statistical models prevent stockouts
- **Economic Order Quantity**: Optimizes ordering frequency and costs
- **Automated Alerts**: Real-time notifications for inventory decisions

### 4. Executive Dashboards
- **4-Panel Summary**: Key metrics at a glance
- **Interactive Visualizations**: Temporal patterns, feature importance, stock status
- **Actionable Insights**: Specific recommendations for each alert type

### 5. AI-Powered Business Assistant
Ask questions about your data in natural language:
- "What are my best performing items?"
- "Which items should I consider removing?"
- "How can I increase revenue?"

### 6. Real-time API Integration
Integrate menu intelligence into your existing systems via REST API.

---

## 👥 Team Members

| Name | Role | Contributions |
|------|------|---------------|
| [Team Member 1] | Data Scientist | ML models, clustering |
| [Team Member 2] | Data Analyst | EDA, visualization |
| [Team Member 3] | Backend Developer | API, architecture |

---

## 📄 License

This project was created for the Deloitte x AUC Hackathon 2024-2025.

---

## 🙏 Acknowledgments

- Deloitte Digital for organizing the hackathon
- AUC for providing the venue and support
- FlavorFlow for the real-world dataset
- AUC for hosting
- Data provided by hackathon organizers

---

*Built with 💚 for the Deloitte x AUC Hackathon 2024-2025*

**Last Updated**: February 2026
