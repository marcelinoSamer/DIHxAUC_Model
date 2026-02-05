"""
File: item_clusterer.py
Description: Clustering model for menu item segmentation.
Dependencies: scikit-learn, numpy, pandas
Author: FlavorFlow Team

This module implements K-Means clustering for segmenting menu items
based on performance metrics (orders, price, revenue).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


class ItemClusterer:
    """
    Item segmentation using K-Means clustering.
    
    Segments menu items into clusters based on performance metrics
    to identify high performers, underperformers, and opportunities.
    
    Attributes:
        n_clusters: Number of clusters to form
        kmeans: Fitted KMeans model
        scaler: StandardScaler for feature normalization
        pca: PCA for visualization
        cluster_labels: Array of cluster assignments
    
    Example:
        >>> clusterer = ItemClusterer(n_clusters=4)
        >>> clusterer.fit(item_performance)
        >>> labels = clusterer.predict(new_items)
    """
    
    def __init__(
        self, 
        n_clusters: int = 4,
        features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize the item clusterer.
        
        Args:
            n_clusters: Number of clusters to create
            features: List of feature columns to use for clustering
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.features = features or ['order_count', 'price', 'revenue']
        self.random_state = random_state
        
        self.kmeans: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.inertias: List[float] = []
        self._is_fitted = False
    
    def find_optimal_k(
        self, 
        df: pd.DataFrame, 
        k_range: range = range(2, 10)
    ) -> Tuple[List[float], int]:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            df: DataFrame with clustering features
            k_range: Range of k values to test
            
        Returns:
            Tuple of (inertia values, suggested optimal k)
        """
        cluster_data = df[self.features].dropna()
        X_scaled = self.scaler.fit_transform(cluster_data)
        
        self.inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)
            self.inertias.append(kmeans.inertia_)
        
        # Simple elbow detection (look for largest decrease)
        deltas = np.diff(self.inertias)
        optimal_k = list(k_range)[np.argmin(deltas) + 1]
        
        return self.inertias, optimal_k
    
    def fit(self, df: pd.DataFrame) -> 'ItemClusterer':
        """
        Fit the clustering model.
        
        Args:
            df: DataFrame with item performance data
            
        Returns:
            Self for method chaining
        """
        cluster_data = df[self.features].dropna()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(cluster_data)
        
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Fit PCA for visualization
        self.pca.fit(X_scaled)
        
        self._is_fitted = True
        return self
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the model and return DataFrame with cluster labels.
        
        Args:
            df: DataFrame with item performance data
            
        Returns:
            DataFrame with 'cluster' column added
        """
        self.fit(df)
        result = df.copy()
        
        # Only add labels to rows that were clustered (had valid data)
        valid_mask = df[self.features].notna().all(axis=1)
        result.loc[valid_mask, 'cluster'] = self.cluster_labels
        
        return result
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new items.
        
        Args:
            df: DataFrame with same features as training data
            
        Returns:
            Array of cluster labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = df[self.features].values
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        
        Args:
            df: DataFrame with cluster labels
            
        Returns:
            Summary DataFrame with cluster statistics
        """
        if 'cluster' not in df.columns:
            raise ValueError("DataFrame must have 'cluster' column")
        
        summary = df.groupby('cluster').agg({
            'item_id': 'count',
            'order_count': ['mean', 'sum'],
            'price': 'mean',
            'revenue': ['mean', 'sum']
        }).round(2)
        
        summary.columns = [
            'items', 'avg_orders', 'total_orders',
            'avg_price', 'avg_revenue', 'total_revenue'
        ]
        
        return summary
    
    def get_cluster_profiles(self) -> Dict[int, str]:
        """
        Generate interpretable profiles for each cluster.
        
        Returns:
            Dictionary mapping cluster ID to profile description
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get cluster centers in original scale
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        profiles = {}
        for i, center in enumerate(centers):
            order_count, price, revenue = center[:3]
            
            # Determine profile based on center values
            if order_count > np.median(centers[:, 0]):
                if price > np.median(centers[:, 1]):
                    profiles[i] = "High Volume, Premium Price (Stars)"
                else:
                    profiles[i] = "High Volume, Value Price (Volume Drivers)"
            else:
                if price > np.median(centers[:, 1]):
                    profiles[i] = "Low Volume, Premium Price (Niche Items)"
                else:
                    profiles[i] = "Low Volume, Low Price (Underperformers)"
        
        return profiles
    
    def get_pca_coordinates(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get PCA-reduced coordinates for visualization.
        
        Args:
            df: DataFrame with clustering features
            
        Returns:
            2D array of PCA coordinates
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        cluster_data = df[self.features].dropna()
        X_scaled = self.scaler.transform(cluster_data)
        return self.pca.transform(X_scaled)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'n_clusters': self.n_clusters,
            'features': self.features,
            'inertias': self.inertias
        }, path)
    
    @classmethod
    def load(cls, path: Path) -> 'ItemClusterer':
        """Load model from disk."""
        data = joblib.load(path)
        clusterer = cls(
            n_clusters=data['n_clusters'],
            features=data['features']
        )
        clusterer.kmeans = data['kmeans']
        clusterer.scaler = data['scaler']
        clusterer.pca = data['pca']
        clusterer.inertias = data['inertias']
        clusterer._is_fitted = True
        return clusterer
