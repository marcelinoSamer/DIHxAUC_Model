"""
Machine Learning Models for Menu Intelligence.

This module provides:
- Demand prediction models (Random Forest, Gradient Boosting)
- Item clustering and segmentation
- Feature engineering utilities
- Model evaluation metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from . import config


# =============================================================================
# DEMAND PREDICTION
# =============================================================================

def prepare_demand_features(dim_menu_items: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for demand prediction model.
    
    Args:
        dim_menu_items: Menu items data with price, rating, votes, purchases
        
    Returns:
        Tuple of (feature DataFrame, target Series)
    """
    # Select and clean data
    model_data = dim_menu_items[
        ['id', 'price', 'rating', 'votes', 'purchases', 'index']
    ].copy()
    
    model_data = model_data.dropna()
    model_data = model_data[model_data['purchases'] > 0]
    model_data = model_data[model_data['price'] > 0]
    model_data = model_data[model_data['price'] < config.MAX_PRICE_MODEL]
    
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
    features = ['price', 'rating', 'votes', 'index', 'price_bucket', 
                'rating_score', 'log_price']
    X = model_data[features]
    y = model_data['purchases']
    
    return X, y


def train_demand_models(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = None,
    random_state: int = None
) -> Dict:
    """
    Train multiple demand prediction models.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with models, scalers, and evaluation metrics
    """
    test_size = test_size or config.TEST_SIZE
    random_state = random_state or config.RANDOM_STATE
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(**config.RF_PARAMS),
        'Gradient Boosting': GradientBoostingRegressor(**config.GB_PARAMS)
    }
    
    # Train and evaluate
    results = {
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'models': {},
        'metrics': {}
    }
    
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results['models'][name] = model
        results['metrics'][name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
    
    # Store test data for diagnostics
    results['X_test'] = X_test
    results['y_test'] = y_test
    results['X_test_scaled'] = X_test_scaled
    
    return results


def print_model_results(results: Dict) -> None:
    """
    Print formatted model training results.
    
    Args:
        results: Output from train_demand_models()
    """
    print("=" * 60)
    print("🤖 DEMAND PREDICTION MODEL RESULTS")
    print("=" * 60)
    print(f"\nTraining samples: {len(results['y_test']) * 5:,}")  # Approx from 20% test
    
    for name, metrics in results['metrics'].items():
        print(f"\n🎯 {name}:")
        print(f"   MAE:  {metrics['MAE']:.2f}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   R²:   {metrics['R2']:.4f}")


def get_feature_importance(results: Dict, model_name: str = 'Random Forest') -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        results: Output from train_demand_models()
        model_name: Name of the model to get importance from
        
    Returns:
        DataFrame with features and their importance scores
    """
    model = results['models'][model_name]
    
    feature_importance = pd.DataFrame({
        'feature': results['feature_names'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance


def print_feature_importance(feature_importance: pd.DataFrame) -> None:
    """
    Print formatted feature importance.
    
    Args:
        feature_importance: DataFrame from get_feature_importance()
    """
    print("\n📈 FEATURE IMPORTANCE (Random Forest):")
    for _, row in feature_importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:>15}: {row['importance']:.3f} {bar}")


# =============================================================================
# ITEM CLUSTERING
# =============================================================================

def perform_item_clustering(
    item_performance: pd.DataFrame,
    n_clusters: int = None,
    features: List[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Cluster items based on performance metrics.
    
    Args:
        item_performance: DataFrame with item metrics
        n_clusters: Number of clusters (uses config default if None)
        features: List of feature columns to use
        
    Returns:
        Tuple of (clustered DataFrame, clustering info dict)
    """
    n_clusters = n_clusters or config.OPTIMAL_CLUSTERS
    features = features or ['order_count', 'price', 'revenue']
    
    # Prepare data
    cluster_data = item_performance[features].copy().dropna()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data)
    
    # Find optimal K using elbow method
    inertias = []
    K_range = config.CLUSTER_RANGE
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, 
                       n_init=config.KMEANS_INIT)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Fit final model
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=config.RANDOM_STATE,
        n_init=config.KMEANS_INIT
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add labels to performance data
    item_performance_clustered = item_performance.copy()
    item_performance_clustered['cluster'] = cluster_labels
    
    # Compile info
    info = {
        'n_clusters': n_clusters,
        'inertias': inertias,
        'K_range': K_range,
        'X_scaled': X_scaled,
        'cluster_labels': cluster_labels,
        'kmeans': kmeans,
        'scaler': scaler
    }
    
    return item_performance_clustered, info


def analyze_clusters(item_performance_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze characteristics of each cluster.
    
    Args:
        item_performance_clustered: DataFrame with cluster labels
        
    Returns:
        Summary DataFrame with cluster statistics
    """
    cluster_summary = item_performance_clustered.groupby('cluster').agg({
        'item_id': 'count',
        'order_count': ['mean', 'sum'],
        'price': 'mean',
        'revenue': ['mean', 'sum']
    }).round(2)
    
    cluster_summary.columns = [
        'items', 'avg_orders', 'total_orders', 
        'avg_price', 'avg_revenue', 'total_revenue'
    ]
    
    return cluster_summary


def print_cluster_analysis(
    cluster_summary: pd.DataFrame, 
    n_clusters: int
) -> None:
    """
    Print formatted cluster analysis.
    
    Args:
        cluster_summary: Output from analyze_clusters()
        n_clusters: Number of clusters
    """
    print("=" * 60)
    print("📊 ITEM SEGMENTATION ANALYSIS")
    print("=" * 60)
    print(f"\n🎯 CLUSTER ANALYSIS (K={n_clusters}):")
    
    for cluster_id in range(n_clusters):
        if cluster_id in cluster_summary.index:
            row = cluster_summary.loc[cluster_id]
            print(f"\n  Cluster {cluster_id}:")
            print(f"    Items: {row['items']:.0f}")
            print(f"    Avg Orders: {row['avg_orders']:.1f}")
            print(f"    Avg Price: {row['avg_price']:.2f}")
            print(f"    Total Revenue: {row['total_revenue']:,.0f}")


# =============================================================================
# PREDICTION UTILITIES
# =============================================================================

def predict_demand(
    model_results: Dict,
    new_data: pd.DataFrame,
    model_name: str = 'Random Forest'
) -> np.ndarray:
    """
    Predict demand for new items.
    
    Args:
        model_results: Output from train_demand_models()
        new_data: DataFrame with same features as training data
        model_name: Which model to use for prediction
        
    Returns:
        Array of predicted demand values
    """
    scaler = model_results['scaler']
    model = model_results['models'][model_name]
    
    # Ensure columns match
    features = model_results['feature_names']
    X_new = new_data[features]
    
    # Scale and predict
    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)
    
    return predictions
