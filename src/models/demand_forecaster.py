"""
File: demand_forecaster.py
Description: Demand forecasting model trained on order-level data.
Dependencies: pandas, numpy, scikit-learn
Author: FlavorFlow Team

This module implements demand prediction using historical order data
to forecast future demand for inventory optimization.

Training Data: fct_orders (400k) + fct_order_items (2M rows)
Target: Predict quantity demanded per item/day/restaurant
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """
    Demand forecasting model for inventory optimization.
    
    Trains on historical order data (2M+ transactions) to predict
    future demand at item/day/restaurant granularity.
    
    Features extracted:
    - Temporal: hour, day_of_week, month, is_weekend, is_holiday
    - Item: price, category, historical popularity
    - Restaurant: location, avg daily orders
    - Lag features: demand in past 7/14/30 days
    
    Example:
        >>> forecaster = DemandForecaster()
        >>> forecaster.load_and_prepare_data(orders_df, order_items_df)
        >>> forecaster.fit()
        >>> predictions = forecaster.predict_next_days(7)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the demand forecaster.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._default_config()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        self._is_fitted = False
        
        # Store aggregated data for predictions
        self.daily_demand: Optional[pd.DataFrame] = None
        self.item_stats: Optional[pd.DataFrame] = None
        self.restaurant_stats: Optional[pd.DataFrame] = None
    
    def _default_config(self) -> Dict:
        """Default model configuration."""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'model_type': 'gradient_boosting',
            'gb_params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'min_samples_split': 10,
                'random_state': 42
            },
            'rf_params': {
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            },
            'aggregation_level': 'daily',  # 'daily', 'hourly', 'weekly'
            'forecast_horizon': 14  # days to predict ahead
        }
    
    def load_and_prepare_data(
        self,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        items_df: Optional[pd.DataFrame] = None,
        places_df: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Load and prepare training data from orders.
        
        Args:
            orders_df: Orders table (fct_orders)
            order_items_df: Order items table (fct_order_items)
            items_df: Optional items dimension table
            places_df: Optional restaurants dimension table
            verbose: Print progress
            
        Returns:
            Prepared training DataFrame
        """
        if verbose:
            print(f"📊 Preparing training data from {len(orders_df):,} orders "
                  f"and {len(order_items_df):,} order items...")
        
        # Step 1: Parse timestamps
        orders_df = orders_df.copy()
        orders_df['order_datetime'] = pd.to_datetime(
            orders_df['created'], unit='s', errors='coerce'
        )
        orders_df['order_date'] = orders_df['order_datetime'].dt.date
        orders_df['order_hour'] = orders_df['order_datetime'].dt.hour
        orders_df['day_of_week'] = orders_df['order_datetime'].dt.dayofweek
        orders_df['month'] = orders_df['order_datetime'].dt.month
        orders_df['week_of_year'] = orders_df['order_datetime'].dt.isocalendar().week
        orders_df['is_weekend'] = orders_df['day_of_week'].isin([5, 6]).astype(int)
        
        if verbose:
            date_range = orders_df['order_datetime'].agg(['min', 'max'])
            print(f"   Date range: {date_range['min']} to {date_range['max']}")
        
        # Step 2: Merge order items with orders
        merged = order_items_df.merge(
            orders_df[['id', 'place_id', 'order_datetime', 'order_date', 
                      'order_hour', 'day_of_week', 'month', 'week_of_year', 
                      'is_weekend', 'total_amount', 'type']],
            left_on='order_id',
            right_on='id',
            how='inner',
            suffixes=('', '_order')
        )
        
        if verbose:
            print(f"   Merged dataset: {len(merged):,} order items with timestamps")
        
        # Step 3: Aggregate to daily demand per item per restaurant
        daily_demand = merged.groupby(
            ['place_id', 'item_id', 'order_date']
        ).agg({
            'quantity': 'sum',
            'price': 'mean',
            'cost': 'mean',
            'order_id': 'nunique',
            'day_of_week': 'first',
            'month': 'first',
            'week_of_year': 'first',
            'is_weekend': 'first'
        }).reset_index()
        
        daily_demand.columns = [
            'place_id', 'item_id', 'date', 'quantity_demanded',
            'avg_price', 'avg_cost', 'num_orders', 'day_of_week',
            'month', 'week_of_year', 'is_weekend'
        ]
        
        if verbose:
            print(f"   Daily demand records: {len(daily_demand):,}")
        
        # Step 4: Calculate item-level statistics
        self.item_stats = merged.groupby('item_id').agg({
            'quantity': ['sum', 'mean', 'std'],
            'price': 'mean',
            'order_id': 'nunique'
        }).reset_index()
        self.item_stats.columns = [
            'item_id', 'total_qty', 'avg_daily_qty', 'std_daily_qty',
            'avg_price', 'total_orders'
        ]
        self.item_stats['std_daily_qty'] = self.item_stats['std_daily_qty'].fillna(0)
        
        # Step 5: Calculate restaurant-level statistics
        self.restaurant_stats = merged.groupby('place_id').agg({
            'order_id': 'nunique',
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        self.restaurant_stats.columns = [
            'place_id', 'total_orders', 'total_qty', 'avg_order_value'
        ]
        
        # Step 6: Add item and restaurant features to daily demand
        daily_demand = daily_demand.merge(
            self.item_stats[['item_id', 'avg_daily_qty', 'std_daily_qty', 'total_orders']],
            on='item_id',
            how='left',
            suffixes=('', '_item')
        )
        daily_demand = daily_demand.merge(
            self.restaurant_stats[['place_id', 'total_orders', 'avg_order_value']],
            on='place_id',
            how='left',
            suffixes=('', '_restaurant')
        )
        
        # Step 7: Add lag features (rolling averages)
        daily_demand = daily_demand.sort_values(['item_id', 'place_id', 'date'])
        daily_demand['lag_7d_avg'] = daily_demand.groupby(
            ['item_id', 'place_id']
        )['quantity_demanded'].transform(
            lambda x: x.rolling(7, min_periods=1).mean().shift(1)
        )
        daily_demand['lag_14d_avg'] = daily_demand.groupby(
            ['item_id', 'place_id']
        )['quantity_demanded'].transform(
            lambda x: x.rolling(14, min_periods=1).mean().shift(1)
        )
        
        # Fill NaN lag features
        daily_demand['lag_7d_avg'] = daily_demand['lag_7d_avg'].fillna(
            daily_demand['avg_daily_qty']
        )
        daily_demand['lag_14d_avg'] = daily_demand['lag_14d_avg'].fillna(
            daily_demand['avg_daily_qty']
        )
        
        self.daily_demand = daily_demand
        
        if verbose:
            print(f"✅ Training data prepared: {len(daily_demand):,} records")
            print(f"   Unique items: {daily_demand['item_id'].nunique():,}")
            print(f"   Unique restaurants: {daily_demand['place_id'].nunique():,}")
        
        return daily_demand
    
    def prepare_features(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable.
        
        Args:
            df: DataFrame (uses self.daily_demand if None)
            
        Returns:
            Tuple of (X features, y target)
        """
        if df is None:
            df = self.daily_demand
        
        if df is None:
            raise ValueError("No data available. Call load_and_prepare_data first.")
        
        # Define features
        self.feature_names = [
            'day_of_week', 'month', 'week_of_year', 'is_weekend',
            'avg_price', 'avg_daily_qty', 'std_daily_qty',
            'total_orders_restaurant', 'avg_order_value',
            'lag_7d_avg', 'lag_14d_avg'
        ]
        
        # Ensure all features exist
        available_features = [f for f in self.feature_names if f in df.columns]
        
        X = df[available_features].copy()
        y = df['quantity_demanded'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        self.feature_names = available_features
        
        return X, y
    
    def fit(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'DemandForecaster':
        """
        Train the demand forecasting model.
        
        Args:
            X: Feature matrix (optional, will prepare if None)
            y: Target variable (optional)
            verbose: Print progress
            
        Returns:
            Self for method chaining
        """
        if X is None or y is None:
            X, y = self.prepare_features()
        
        if verbose:
            print(f"\n🤖 Training demand forecasting model on {len(X):,} samples...")
            print(f"   Features: {len(self.feature_names)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if self.config['model_type'] == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self.config['gb_params'])
        else:
            self.model = RandomForestRegressor(**self.config['rf_params'])
        
        # Train
        import time
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'training_time_seconds': train_time
        }
        
        self._is_fitted = True
        
        if verbose:
            print(f"   Training time: {train_time:.1f} seconds")
            print(f"\n📈 Model Performance:")
            print(f"   MAE:  {self.metrics['mae']:.2f} units")
            print(f"   RMSE: {self.metrics['rmse']:.2f} units")
            print(f"   R²:   {self.metrics['r2']:.4f}")
        
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def predict_demand(
        self,
        item_id: int,
        place_id: int,
        future_dates: List[datetime]
    ) -> pd.DataFrame:
        """
        Predict demand for a specific item at a specific restaurant.
        
        Args:
            item_id: Item ID to predict
            place_id: Restaurant ID
            future_dates: List of dates to predict
            
        Returns:
            DataFrame with predictions
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = []
        
        for date in future_dates:
            features = {
                'day_of_week': date.weekday(),
                'month': date.month,
                'week_of_year': date.isocalendar()[1],
                'is_weekend': 1 if date.weekday() >= 5 else 0
            }
            
            # Add item features
            if self.item_stats is not None:
                item_row = self.item_stats[self.item_stats['item_id'] == item_id]
                if not item_row.empty:
                    features['avg_price'] = item_row['avg_price'].values[0]
                    features['avg_daily_qty'] = item_row['avg_daily_qty'].values[0]
                    features['std_daily_qty'] = item_row['std_daily_qty'].values[0]
            
            # Add restaurant features
            if self.restaurant_stats is not None:
                rest_row = self.restaurant_stats[self.restaurant_stats['place_id'] == place_id]
                if not rest_row.empty:
                    features['total_orders_restaurant'] = rest_row['total_orders'].values[0]
                    features['avg_order_value'] = rest_row['avg_order_value'].values[0]
            
            # Add lag features (use historical average as proxy)
            features['lag_7d_avg'] = features.get('avg_daily_qty', 1)
            features['lag_14d_avg'] = features.get('avg_daily_qty', 1)
            
            # Create feature vector
            X = pd.DataFrame([features])[self.feature_names]
            X = X.fillna(X.median())
            X_scaled = self.scaler.transform(X)
            
            pred = max(0, self.model.predict(X_scaled)[0])
            
            predictions.append({
                'date': date,
                'item_id': item_id,
                'place_id': place_id,
                'predicted_demand': round(pred, 1)
            })
        
        return pd.DataFrame(predictions)
    
    def forecast_all_items(
        self,
        days_ahead: int = 7,
        top_n_items: Optional[int] = 100
    ) -> pd.DataFrame:
        """
        Forecast demand for all (or top N) items across all restaurants.
        
        Args:
            days_ahead: Number of days to forecast
            top_n_items: Limit to top N items by volume (None for all)
            
        Returns:
            DataFrame with forecasts
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get items to forecast
        items = self.item_stats.copy()
        if top_n_items:
            items = items.nlargest(top_n_items, 'total_qty')
        
        # Get restaurants
        restaurants = self.restaurant_stats['place_id'].unique()[:50]  # Limit for speed
        
        # Generate future dates
        today = datetime.now()
        future_dates = [today + timedelta(days=i) for i in range(1, days_ahead + 1)]
        
        all_forecasts = []
        
        for item_id in items['item_id'].values:
            for place_id in restaurants:
                forecast = self.predict_demand(item_id, place_id, future_dates)
                all_forecasts.append(forecast)
        
        return pd.concat(all_forecasts, ignore_index=True)
    
    def get_summary(self) -> Dict:
        """Get summary of the forecaster state and metrics."""
        summary = {
            'is_fitted': self._is_fitted,
            'model_type': self.config['model_type'],
            'features': self.feature_names
        }
        
        if self._is_fitted:
            summary['metrics'] = self.metrics
        
        if self.daily_demand is not None:
            summary['training_data'] = {
                'total_records': len(self.daily_demand),
                'unique_items': self.daily_demand['item_id'].nunique(),
                'unique_restaurants': self.daily_demand['place_id'].nunique(),
                'date_range': {
                    'start': str(self.daily_demand['date'].min()),
                    'end': str(self.daily_demand['date'].max())
                }
            }
        
        return summary
