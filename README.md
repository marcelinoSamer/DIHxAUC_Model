# 🍽️ FlavorFlow Craft: AI-Powered Menu Intelligence Platform

> **Deloitte x AUC Hackathon 2024-2025 - Menu Engineering Challenge**

Transform restaurant menu decisions from gut instinct to data-driven insights. FlavorFlow Craft analyzes historical sales data to identify profit opportunities, optimize pricing, and maximize revenue.

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

FlavorFlow restaurants sit on a goldmine of historical sales data—every order, every customer preference—yet they're making menu decisions on hunches. They don't know:
- Which dishes are secretly losing money
- What tweaks could turn underperformers into bestsellers
- How to optimize pricing for maximum profitability

**This isn't just a missed opportunity; it's revenue left on the table.**

---

## 💡 Solution Overview

**FlavorFlow Craft Menu Intelligence** is a data-driven assistant that:

1. **Classifies menu items** using BCG Matrix methodology (Stars, Plowhorses, Puzzles, Dogs)
2. **Predicts demand** using machine learning models
3. **Optimizes pricing** based on elasticity analysis
4. **Segments items** using K-Means clustering
5. **Provides REST API** for integration with existing systems
6. **Answers business questions** in natural language (AI-powered)

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

### 2. Demand Prediction Model
- **Random Forest** and **Gradient Boosting** regression models
- Features: price, rating, votes, menu position
- Predicts item purchase volume for inventory planning

### 3. Pricing Optimization
![Pricing Analysis](pricing_analysis.png)

- Price distribution analysis
- Price elasticity insights
- Optimal price point recommendations

### 4. Campaign Effectiveness
![Campaign Analysis](campaign_analysis.png)

- Discount impact analysis
- Redemption rate tracking
- ROI optimization recommendations

### 5. Restaurant Performance Dashboard
- Location-level performance metrics
- Best practice identification
- Underperformer detection

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Notebook** | Jupyter / VS Code |
| **Dimensionality Reduction** | PCA |
| **Clustering** | KMeans |
| **API Framework** | FastAPI |
| **Testing** | pytest |

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

### Option 1: Run Full Analysis (CLI)

```bash
# Run professional analysis with new architecture
python -m src.main analyze

# Or run with custom paths
python -m src.main analyze --data-dir data/ --output-dir docs/
```

### Option 2: Start API Server

```bash
# Start the FastAPI server
python -m src.main serve --port 8000

# With auto-reload for development
python -m src.main serve --reload
```

Then visit: http://localhost:8000/docs for interactive API documentation.

### Option 3: Run Jupyter Notebook

1. Open `main.ipynb` in VS Code or Jupyter
2. Select your Python environment (`venv`)
3. Run all cells

### Option 4: Run Legacy Analysis

```bash
python -m src.main legacy
```

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
│   ├── dim_items.csv                 # Item catalog
│   ├── dim_menu_items.csv            # Menu configurations
│   ├── dim_places.csv                # Restaurant locations
│   ├── fct_orders.csv                # Transaction history
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
│   │   ├── demand_predictor.py      # Demand forecasting
│   │   ├── item_clusterer.py        # K-Means clustering
│   │   └── menu_classifier.py       # BCG Matrix classifier
│   │
│   ├── services/                     # Business logic services
│   │   ├── __init__.py
│   │   ├── menu_analysis_service.py # Main orchestration
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
    ├── menu_engineering_matrix.png
    ├── pricing_analysis.png
    ├── clustering_analysis.png
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

### Key Findings

1. **Menu Composition**
   - ~25% of items are Stars (high performers)
   - ~30% are Plowhorses (volume drivers with margin opportunity)
   - ~20% are Puzzles (hidden gems needing visibility)
   - ~25% are Dogs (candidates for removal)

2. **Pricing Opportunities**
   - Median price point: ~75 DKK
   - 15-20% of items are underpriced by 10-15%
   - Price elasticity highest in mid-range items

3. **Campaign Insights**
   - 15-20% discounts drive highest redemption
   - "2 for 1" promotions outperform percentage discounts
   - Most campaigns have <10% redemption rate

### Expected Business Impact

| Metric | Expected Improvement |
|--------|---------------------|
| Revenue | +8-15% |
| Margin | +5-10% |
| Waste Reduction | -15-25% |
| Campaign ROI | 2-3x |

---

## 🚀 Innovative Features

### 1. AI-Powered Business Assistant
Ask questions about your data in natural language:
- "What are my best performing items?"
- "Which items should I consider removing?"
- "How can I increase revenue?"

### 2. Real-time API Integration
Integrate menu intelligence into your existing systems via REST API.

### 3. Automated Pricing Suggestions
ML-powered pricing recommendations with estimated revenue impact.

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

*Built with 💚 for the Deloitte x AUC Hackathon 2026*
