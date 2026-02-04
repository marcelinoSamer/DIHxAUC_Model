"""
Configuration settings for the Menu Intelligence project.

This module centralizes all configurable parameters including:
- File paths
- Analysis thresholds  
- Visualization settings
- Model hyperparameters
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory - use data folder in project
DATA_DIR = PROJECT_ROOT / "data"

# Output directory for results and visualizations
OUTPUT_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA FILES
# =============================================================================

DATA_FILES = {
    "dim_items": DATA_DIR / "dim_items.csv",
    "dim_menu_items": DATA_DIR / "dim_menu_items.csv",
    "dim_places": DATA_DIR / "dim_places.csv",
    "most_ordered": DATA_DIR / "most_ordered.csv",
    "dim_campaigns": DATA_DIR / "dim_campaigns.csv",
    "fct_campaigns": DATA_DIR / "fct_campaigns.csv",
}

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# Menu Engineering (BCG Matrix) thresholds
# Items are classified based on whether they're above/below these percentiles
POPULARITY_PERCENTILE = 50  # Median split for popularity
PRICE_PERCENTILE = 50  # Median split for profitability

# Price analysis settings
PRICE_BINS = [0, 25, 50, 75, 100, 150, 200, float('inf')]
PRICE_LABELS = ['0-25', '26-50', '51-75', '76-100', '101-150', '151-200', '200+']
MAX_PRICE_DISPLAY = 500  # Max price for visualizations
MAX_PRICE_MODEL = 10000  # Max price for ML models (outlier filter)

# =============================================================================
# VISUALIZATION SETTINGS  
# =============================================================================

# Figure sizes
FIG_SIZE_SINGLE = (10, 6)
FIG_SIZE_DOUBLE = (14, 5)
FIG_SIZE_QUAD = (16, 12)

# Color palettes
COLORS = {
    "star": "#2ecc71",      # Green - high performer
    "plowhorse": "#3498db", # Blue - high volume, low margin
    "puzzle": "#f39c12",    # Orange - low volume, high margin
    "dog": "#e74c3c",       # Red - underperformer
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
}

BCG_COLORS = {
    "⭐ Star": COLORS["star"],
    "🐴 Plowhorse": COLORS["plowhorse"],
    "❓ Puzzle": COLORS["puzzle"],
    "🐕 Dog": COLORS["dog"],
}

# Plot style
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
DPI = 150

# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Random Forest parameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
}

# Gradient Boosting parameters
GB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
}

# Clustering settings
CLUSTER_RANGE = range(2, 10)
OPTIMAL_CLUSTERS = 4  # Matching BCG matrix categories
KMEANS_INIT = 10

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

# Pandas display options
PANDAS_MAX_COLUMNS = 50
PANDAS_DISPLAY_WIDTH = 1000

# Number of items to show in top/bottom lists
TOP_N_ITEMS = 10
