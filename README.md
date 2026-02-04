# 🍽️ FlavorCraft: AI-Powered Menu Intelligence Platform

> **Deloitte x AUC Hackathon 2026 - Menu Engineering Challenge**

Transform restaurant menu decisions from gut instinct to data-driven insights. FlavorCraft analyzes historical sales data to identify profit opportunities, optimize pricing, and maximize revenue.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Results & Insights](#-results--insights)
- [Team Members](#-team-members)

---

## 🎯 Problem Statement

FlavorCraft restaurants sit on a goldmine of historical sales data—every order, every customer preference—yet they're making menu decisions on hunches. They don't know:
- Which dishes are secretly losing money
- What tweaks could turn underperformers into bestsellers
- How to optimize pricing for maximum profitability

**This isn't just a missed opportunity; it's revenue left on the table.**

---

## 💡 Solution Overview

**FlavorCraft Menu Intelligence** is a data-driven assistant that:

1. **Classifies menu items** using BCG Matrix methodology (Stars, Plowhorses, Puzzles, Dogs)
2. **Predicts demand** using machine learning models
3. **Optimizes pricing** based on elasticity analysis
4. **Segments customers** to enable targeted marketing
5. **Evaluates promotions** to maximize campaign ROI

---

## ✨ Features

### 1. Menu Engineering (BCG Matrix)
![Menu Engineering Matrix](menu_engineering_matrix.png)

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
| **Dimensionality Reduction** | PCA, UMAP |
| **Clustering** | KMeans |

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-team/flavorcraft-menu-intelligence.git
cd flavorcraft-menu-intelligence
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Analysis Notebook

1. Open VS Code or Jupyter
2. Open `main.ipynb`
3. Select your Python environment (`.venv`)
4. Run all cells

### Output Files

After running the notebook, you'll find:

| File | Description |
|------|-------------|
| `results_menu_engineering.csv` | BCG matrix classification for all items |
| `results_recommendations.csv` | Action items for each category |
| `results_item_clusters.csv` | ML clustering results |
| `results_restaurant_performance.csv` | Location-level analysis |

---

## 🏗️ Architecture

```
flavorcraft-menu-intelligence/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── main.ipynb                   # Main analysis notebook
├── data/                        # Source data
│   ├── dim_items.csv           # Menu items dimension
│   ├── dim_menu_items.csv      # Menu details
│   ├── dim_places.csv          # Restaurant locations
│   ├── most_ordered.csv        # Order aggregates
│   ├── fct_campaigns.csv       # Campaign data
│   └── ...                     # Other tables
├── results/                     # Generated outputs
│   ├── results_menu_engineering.csv
│   ├── results_recommendations.csv
│   └── ...
└── visualizations/              # Charts and plots
    ├── menu_engineering_matrix.png
    ├── pricing_analysis.png
    └── ...
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

## 👥 Team Members

| Name | Role | Contributions |
|------|------|---------------|
| [Team Member 1] | Data Scientist | ML models, clustering |
| [Team Member 2] | Data Analyst | EDA, visualization |
| [Team Member 3] | Business Analyst | Recommendations, documentation |

---

## 📄 License

This project was created for the Deloitte x AUC Hackathon 2026.

---

## 🙏 Acknowledgments

- Deloitte for organizing the hackathon
- AUC for hosting
- Data provided by hackathon organizers

---

*Built with 💚 for the Deloitte x AUC Hackathon 2026*
