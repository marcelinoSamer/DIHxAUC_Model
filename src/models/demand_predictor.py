"""
File: demand_predictor.py
Description: Machine learning model for predicting menu item demand.
Dependencies: scikit-learn, numpy, pandas
Author: FlavorFlow Team

This module implements demand prediction using ensemble methods
(Random Forest, Gradient Boosting) for forecasting item popularity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path


class DemandPredictor:
    """
    Demand prediction model using ensemble learning.
    
    This class encapsulates the training, evaluation, and prediction
    logic for forecasting menu item demand based on historical data.
    
    Attributes:
        models: Dictionary of trained model instances
        scaler: StandardScaler for feature normalization
        feature_names: List of feature column names
        metrics: Dictionary of model performance metrics
    
    Example:
        >>> predictor = DemandPredictor()
        >>> predictor.fit(X_train, y_train)
        >>> predictions = predictor.predict(X_test)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the demand predictor.
        
        Args:
            config: Optional configuration dictionary with model parameters
        """
        self.config = config or self._default_config()
        self.models: Dict = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        self._is_fitted = False
    
    def _default_config(self) -> Dict:
        """Return default model configuration."""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'rf_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gb_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features from menu items data.
        
        Args:
            df: DataFrame with columns ['price', 'rating', 'votes', 'purchases', 'index']
            
        Returns:
            Tuple of (feature DataFrame, target Series)
        """
        # Select and clean data
        required_cols = ['price', 'rating', 'votes', 'purchases', 'index']
        model_data = df[required_cols].copy()
        
        model_data = model_data.dropna()
        model_data = model_data[model_data['purchases'] > 0]
        model_data = model_data[model_data['price'] > 0]
        model_data = model_data[model_data['price'] < 10000]  # Remove outliers
        
        # Feature engineering
        model_data['price_bucket'] = pd.qcut(
            model_data['price'].clip(1, 1000), 
            q=5, 
            labels=False, 
            duplicates='drop'
        )
        model_data['rating_score'] = model_data['rating'] * model_data['votes']
        model_data['log_price'] = np.log1p(model_data['price'])
        
        # Clean infinite values
        model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Define features and target
        self.feature_names = [
            'price', 'rating', 'votes', 'index', 
            'price_bucket', 'rating_score', 'log_price'
        ]
        X = model_data[self.feature_names]
        y = model_data['purchases']
        
        return X, y
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DemandPredictor':
        """
        Train demand prediction models.
        
        Args:
            X: Feature DataFrame
            y: Target Series (purchase counts)
            
        Returns:
            Self for method chaining
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train models
        self.models = {
            'random_forest': RandomForestRegressor(**self.config['rf_params']),
            'gradient_boosting': GradientBoostingRegressor(**self.config['gb_params'])
        }
        
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            self.metrics[name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        model_name: str = 'random_forest'
    ) -> np.ndarray:
        """
        Predict demand for new items.
        
        Args:
            X: Feature DataFrame with same columns as training data
            model_name: Which model to use for prediction
            
        Returns:
            Array of predicted demand values
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.models[model_name].predict(X_scaled)
    
    def get_feature_importance(
        self, 
        model_name: str = 'random_forest'
    ) -> pd.DataFrame:
        """
        Get feature importance scores from the model.
        
        Args:
            model_name: Which model to get importance from
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.models[model_name].feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: Path) -> 'DemandPredictor':
        """Load model from disk."""
        data = joblib.load(path)
        predictor = cls(config=data['config'])
        predictor.models = data['models']
        predictor.scaler = data['scaler']
        predictor.feature_names = data['feature_names']
        predictor.metrics = data['metrics']
        predictor._is_fitted = True
        return predictor
    
    def get_metrics_summary(self) -> str:
        """Return formatted metrics summary."""
        lines = ["Model Performance Metrics:", "=" * 40]
        for name, metrics in self.metrics.items():
            lines.append(f"\n{name.replace('_', ' ').title()}:")
            lines.append(f"  MAE:  {metrics['mae']:.2f}")
            lines.append(f"  RMSE: {metrics['rmse']:.2f}")
            lines.append(f"  R²:   {metrics['r2']:.4f}")
        return "\n".join(lines)
