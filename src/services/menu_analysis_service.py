"""
File: menu_analysis_service.py
Description: Business logic service for menu engineering analysis.
Dependencies: pandas, numpy
Author: FlavorFlow Team

This service orchestrates the complete menu analysis pipeline,
coordinating data loading, classification, clustering, and reporting.
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.models.menu_classifier import MenuClassifier
from src.models.item_clusterer import ItemClusterer
from src.models.demand_predictor import DemandPredictor
from src.utils.data_loader import DataLoader
from src.utils.helpers import format_currency, convert_unix_timestamp


class MenuAnalysisService:
    """
    Service for comprehensive menu engineering analysis.
    
    Orchestrates data loading, BCG classification, clustering,
    demand prediction, and recommendation generation.
    
    Attributes:
        data_loader: DataLoader instance for data access
        classifier: MenuClassifier for BCG analysis
        clusterer: ItemClusterer for segmentation
        predictor: DemandPredictor for demand forecasting
    
    Example:
        >>> service = MenuAnalysisService(data_dir='data/')
        >>> results = service.run_full_analysis()
        >>> service.export_results(results, 'docs/')
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the menu analysis service.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_loader = DataLoader(data_dir)
        self.classifier = MenuClassifier()
        self.clusterer = ItemClusterer(n_clusters=4)
        self.predictor = DemandPredictor()
        
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._results: Dict = {}
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets.
        
        Returns:
            Dictionary of dataset name to DataFrame
        """
        self._datasets = self.data_loader.load_all()
        return self._datasets
    
    def prepare_item_performance(self) -> pd.DataFrame:
        """
        Prepare item performance data for analysis.
        
        Returns:
            DataFrame with aggregated item metrics
        """
        if not self._datasets:
            self.load_data()
        
        most_ordered = self._datasets.get('most_ordered', pd.DataFrame())
        dim_items = self._datasets.get('dim_items', pd.DataFrame())
        
        if most_ordered.empty or dim_items.empty:
            raise ValueError("Required datasets not loaded")
        
        # Merge datasets
        menu_analysis = most_ordered.merge(
            dim_items[['id', 'title', 'price', 'description', 'section_id', 
                      'status', 'purchases', 'vat']],
            left_on='item_id',
            right_on='id',
            how='left'
        )
        
        # Clean data
        menu_analysis = menu_analysis.dropna(subset=['price', 'order_count'])
        menu_analysis = menu_analysis[menu_analysis['price'] > 0]
        menu_analysis = menu_analysis[menu_analysis['order_count'] > 0]
        
        # Calculate revenue
        menu_analysis['revenue'] = menu_analysis['price'] * menu_analysis['order_count']
        
        # Aggregate by item
        item_performance = menu_analysis.groupby(['item_id', 'item_name']).agg({
            'order_count': 'sum',
            'price': 'mean',
            'revenue': 'sum'
        }).reset_index()
        
        return item_performance
    
    def run_bcg_analysis(
        self, 
        item_performance: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run BCG Matrix classification.
        
        Args:
            item_performance: Optional pre-prepared data
            
        Returns:
            Tuple of (classified DataFrame, metrics dictionary)
        """
        if item_performance is None:
            item_performance = self.prepare_item_performance()
        
        classified = self.classifier.fit_transform(item_performance)
        
        metrics = {
            'popularity_threshold': self.classifier.thresholds.popularity,
            'price_threshold': self.classifier.thresholds.profitability,
            'category_metrics': self.classifier.category_metrics
        }
        
        self._results['bcg_classification'] = classified
        self._results['bcg_metrics'] = metrics
        
        return classified, metrics
    
    def run_clustering(
        self, 
        item_performance: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run item clustering analysis.
        
        Args:
            item_performance: Optional pre-prepared data
            
        Returns:
            Tuple of (clustered DataFrame, cluster info dictionary)
        """
        if item_performance is None:
            item_performance = self._results.get(
                'bcg_classification',
                self.prepare_item_performance()
            )
        
        # Find optimal K
        inertias, optimal_k = self.clusterer.find_optimal_k(item_performance)
        
        # Fit clustering
        clustered = self.clusterer.fit_transform(item_performance)
        cluster_summary = self.clusterer.get_cluster_summary(clustered)
        cluster_profiles = self.clusterer.get_cluster_profiles()
        
        cluster_info = {
            'n_clusters': self.clusterer.n_clusters,
            'inertias': inertias,
            'cluster_summary': cluster_summary,
            'cluster_profiles': cluster_profiles
        }
        
        self._results['clustered_items'] = clustered
        self._results['cluster_info'] = cluster_info
        
        return clustered, cluster_info
    
    def run_demand_prediction(self) -> Dict:
        """
        Train and evaluate demand prediction models.
        
        Returns:
            Dictionary with model results and metrics
        """
        dim_menu_items = self._datasets.get('dim_menu_items', pd.DataFrame())
        
        if dim_menu_items.empty:
            raise ValueError("dim_menu_items dataset not loaded")
        
        # Prepare features and train
        X, y = self.predictor.prepare_features(dim_menu_items)
        self.predictor.fit(X, y)
        
        # Get results
        feature_importance = self.predictor.get_feature_importance()
        
        prediction_results = {
            'metrics': self.predictor.metrics,
            'feature_importance': feature_importance,
            'training_samples': len(X)
        }
        
        self._results['prediction'] = prediction_results
        
        return prediction_results
    
    def generate_recommendations(self) -> pd.DataFrame:
        """
        Generate actionable recommendations for all items.
        
        Returns:
            DataFrame with prioritized recommendations
        """
        if 'bcg_classification' not in self._results:
            self.run_bcg_analysis()
        
        classified = self._results['bcg_classification']
        recommendations = self.classifier.get_recommendations(classified)
        
        self._results['recommendations'] = recommendations
        
        return recommendations
    
    def generate_pricing_suggestions(self) -> pd.DataFrame:
        """
        Generate pricing optimization suggestions.
        
        Returns:
            DataFrame with pricing suggestions for plowhorses
        """
        if 'bcg_classification' not in self._results:
            self.run_bcg_analysis()
        
        classified = self._results['bcg_classification']
        pricing = self.classifier.get_pricing_suggestions(classified)
        
        self._results['pricing_suggestions'] = pricing
        
        return pricing
    
    def run_full_analysis(self) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Returns:
            Dictionary with all analysis results
        """
        # Load data
        self.load_data()
        
        # Run all analyses
        item_performance = self.prepare_item_performance()
        
        self.run_bcg_analysis(item_performance)
        self.run_clustering()
        
        try:
            self.run_demand_prediction()
        except Exception as e:
            print(f"Warning: Demand prediction failed: {e}")
        
        self.generate_recommendations()
        self.generate_pricing_suggestions()
        
        return self._results
    
    def get_executive_summary(self) -> Dict:
        """
        Generate executive summary of key findings.
        
        Returns:
            Dictionary with summary metrics and insights
        """
        summary = {
            'data_overview': {
                'total_items': len(self._datasets.get('dim_items', [])),
                'total_restaurants': len(self._datasets.get('dim_places', [])),
                'total_orders': self._datasets.get('most_ordered', pd.DataFrame()).get('order_count', pd.Series()).sum(),
                'total_campaigns': len(self._datasets.get('fct_campaigns', []))
            }
        }
        
        if 'bcg_classification' in self._results:
            classified = self._results['bcg_classification']
            summary['bcg_breakdown'] = {
                'stars': len(classified[classified['category'] == '⭐ Star']),
                'plowhorses': len(classified[classified['category'] == '🐴 Plowhorse']),
                'puzzles': len(classified[classified['category'] == '❓ Puzzle']),
                'dogs': len(classified[classified['category'] == '🐕 Dog'])
            }
        
        if 'pricing_suggestions' in self._results:
            pricing = self._results['pricing_suggestions']
            if not pricing.empty:
                summary['pricing_opportunity'] = {
                    'total_revenue_gain': pricing['revenue_gain'].sum(),
                    'items_to_reprice': len(pricing)
                }
        
        return summary
    
    def export_results(self, output_dir: Path) -> Dict[str, Path]:
        """
        Export all results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping result names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        paths = {}
        
        if 'bcg_classification' in self._results:
            path = output_dir / 'results_menu_engineering.csv'
            self._results['bcg_classification'].to_csv(path, index=False)
            paths['bcg_classification'] = path
        
        if 'clustered_items' in self._results:
            path = output_dir / 'results_item_clusters.csv'
            self._results['clustered_items'].to_csv(path, index=False)
            paths['clusters'] = path
        
        if 'recommendations' in self._results:
            path = output_dir / 'results_recommendations.csv'
            self._results['recommendations'].to_csv(path, index=False)
            paths['recommendations'] = path
        
        if 'pricing_suggestions' in self._results:
            path = output_dir / 'results_pricing_suggestions.csv'
            self._results['pricing_suggestions'].to_csv(path, index=False)
            paths['pricing'] = path
        
        return paths
