"""
Machine Learning Models Package.

This package contains all ML models for demand prediction,
item clustering, customer segmentation, and inventory optimization.
"""

from .demand_predictor import DemandPredictor
from .item_clusterer import ItemClusterer
from .menu_classifier import MenuClassifier
from .demand_forecaster import DemandForecaster
from .inventory_optimizer import InventoryOptimizer, StockStatus, StockAlertGenerator
from .customer_analyzer import CustomerBehaviorAnalyzer

__all__ = [
    # Legacy BCG analysis models
    'DemandPredictor', 
    'ItemClusterer', 
    'MenuClassifier',
    # New inventory optimization models
    'DemandForecaster',
    'InventoryOptimizer',
    'StockStatus',
    'StockAlertGenerator',
    'CustomerBehaviorAnalyzer',
]
