"""
File: menu_analysis_service.py
Description: Business logic service for menu engineering analysis.
Dependencies: pandas, numpy
Author: FlavorFlow Team

This service orchestrates the complete menu analysis pipeline,
coordinating data loading, classification, clustering, and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
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
        
        # Merge with DB data
        self.load_data_from_db()
        
        return self._datasets
    
    def load_data_from_db(self):
        """
        Load data from SQLite database and merge with CSV data.
        """
        from src.database import SessionLocal
        from src.models.db_models import MenuItem, OrderItem, Restaurant
        
        db = SessionLocal()
        try:
            # Load Menu Items from DB
            db_items = db.query(MenuItem).all()
            if db_items:
                items_data = [{
                    'item_id': item.id,
                    'item_name': item.name,
                    'price': item.price,
                    'category': item.category,
                    'description': item.description,
                    'restaurant_id': item.restaurant_id
                } for item in db_items]
                
                # Convert to DataFrame
                df_items = pd.DataFrame(items_data)
                
                # If we have existing CSV data, we might need to reconcile or append
                # For now, let's assume we want to analyze what's in the DB if present, 
                # or mix it.
                # A simple approach is to append or override 'most_ordered' or create a combined view.
                
                # Let's create a synthetic 'most_ordered' from DB orders/items
                # This will allow the rest of the pipeline to work seamlessly
                
                # Fetch order items
                db_order_items = db.query(OrderItem).all()
                if db_order_items:
                    order_data = [{
                        'item_id': oi.menu_item_id,
                        'order_count': oi.quantity,
                        'revenue': oi.quantity * oi.price_at_time
                    } for oi in db_order_items]
                    
                    df_orders = pd.DataFrame(order_data)
                    
                    # Group by item to get total orders/revenue
                    item_performance = df_orders.groupby('item_id').agg({
                        'order_count': 'sum',
                        'revenue': 'sum'
                    }).reset_index()
                    
                    # Merge with item details
                    full_data = item_performance.merge(df_items, on='item_id', how='left')
                    
                    # Store in a special key or override
                    self._results['db_item_performance'] = full_data
            
        finally:
            db.close()
    
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
        
        item_performance = menu_analysis.groupby(['item_id', 'item_name']).agg({
            'order_count': 'sum',
            'price': 'mean',
            'revenue': 'sum'
        }).reset_index()
        
        # Check if we have DB data and merge/append
        if 'db_item_performance' in self._results:
            db_perf = self._results['db_item_performance']
            # Align columns
            db_perf = db_perf[['item_id', 'item_name', 'order_count', 'price', 'revenue']]
            
            # For simplicity in this hybrid mode, we'll append. 
            # In a real scenario, we'd handle ID collisions.
            item_performance = pd.concat([item_performance, db_perf], ignore_index=True)
            
            # Re-aggregate in case of duplicates (though IDs should ideally be distinct)
            item_performance = item_performance.groupby(['item_id', 'item_name']).agg({
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
        
        # Save the trained classifier
        try:
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            classifier_path = model_dir / "bcg_classifier.joblib"
            self.classifier.save(classifier_path)
            print(f"   💾 BCG classifier saved to {classifier_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save BCG classifier: {e}")

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
        
        # Save the trained model for later use
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "demand_model.joblib"
        self.predictor.save(model_path)
        print(f"   💾 Demand model saved to {model_path}")
        
        # Get results
        feature_importance = self.predictor.get_feature_importance()
        
        prediction_results = {
            'metrics': self.predictor.metrics,
            'feature_importance': feature_importance,
            'training_samples': len(X),
            'model_path': str(model_path)
        }
        
        self._results['prediction'] = prediction_results
        
        return prediction_results
    
    def predict_demand_for_items(self, items: List[Dict]) -> Dict[int, float]:
        """
        Predict demand for a list of items using the trained model.
        
        Args:
            items: List of dictionaries with keys ['id', 'price']
            
        Returns:
            Dictionary mapping item_id to predicted daily demand
        """
        # Load model if not already loaded
        if not self.predictor._is_fitted:
            model_path = Path("models/demand_model.joblib")
            if model_path.exists():
                print(f"   🔄 Loading saved demand model from {model_path}...")
                self.predictor = DemandPredictor.load(model_path)
            else:
                print("   ⚠️ No trained model found. Using fallback logic.")
                return {} # Cannot predict without model

        # Prepare features for prediction
        # We need to construct a DataFrame with the same features as training data
        # Features: ['price', 'rating', 'votes', 'index', 'price_bucket', 'rating_score', 'log_price']
        
        predictions = {}
        items_to_predict = []
        item_indices = []
        
        # Get reference data for defaults/lookups
        dim_menu = self._datasets.get('dim_menu_items', pd.DataFrame())
        avg_rating = 4.5
        avg_votes = 50
        
        if not dim_menu.empty and 'rating' in dim_menu.columns:
             avg_rating = dim_menu['rating'].mean()
             avg_votes = dim_menu['votes'].mean()

        for item in items:
            item_id = item.get('id')
            price = item.get('price', 0)
            
            # Try to find item in historical data to get real rating/votes
            # This connects the database item to the training data "knowledge"
            hist_data = None
            if not dim_menu.empty and 'id' in dim_menu.columns:
                hist_data = dim_menu[dim_menu['id'] == item_id]
            
            if hist_data is not None and not hist_data.empty:
                rating = hist_data.iloc[0]['rating']
                votes = hist_data.iloc[0]['votes']
            else:
                # For new items not in training set, use averages (optimistic start)
                rating = avg_rating
                votes = avg_votes
            
            # Construct row for prediction
            # Note: 'purchases' column is target, so we don't need it for prediction input
            # But prepare_features might expect it to exist in the dataframe structure
            # We'll adapt by manually creating the feature set expected by predict()
            
            row = {
                'price': price,
                'rating': rating,
                'votes': votes,
                'index': item_id,
                # Engineered features will be handled by predictor if we pass raw df?
                # Actually predictor.predict expects specific feature columns.
                # Let's reconstruct them manually to match training logic.
            }
            items_to_predict.append(row)
            item_indices.append(item_id)
            
        if not items_to_predict:
            return {}
            
        # Create DataFrame
        df_pred = pd.DataFrame(items_to_predict)
        
        # Apply feature engineering manually
        # Robust handling for price buckets (qcut fails on small datasets)
        try:
            if len(df_pred) >= 5:
                df_pred['price_bucket'] = pd.qcut(
                    df_pred['price'].clip(1, 1000), 
                    q=5, 
                    labels=False, 
                    duplicates='drop'
                ).fillna(2)
            else:
                # Fallback for small batches: simple heuristic buckets
                # 0-50: 0, 50-100: 1, 100-200: 2, 200-500: 3, 500+: 4
                df_pred['price_bucket'] = pd.cut(
                    df_pred['price'],
                    bins=[0, 50, 100, 200, 500, 10000],
                    labels=[0, 1, 2, 3, 4]
                ).fillna(2)
        except Exception:
            df_pred['price_bucket'] = 2 # Default middle bucket
        
        df_pred['rating_score'] = df_pred['rating'] * df_pred['votes']
        df_pred['log_price'] = np.log1p(df_pred['price'])
        
        try:
            # Run prediction
            if not predictions: # Only run if not already filled (logic check)
                predicted_values = self.predictor.predict(df_pred)
                
                # Map back to item IDs
                for idx, item_id in enumerate(item_indices):
                    # Model predicts total purchases (lifetime/long-term). 
                    # Converting to estimated DAILY demand.
                    # Assuming training data covers ~2 years (730 days).
                    daily_demand = max(0.1, predicted_values[idx] / 730.0)
                    predictions[item_id] = daily_demand
                
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            
        return predictions

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
    
    
    def classify_database_items(self, items: List[Dict]) -> Dict[int, str]:
        """
        Classify database items using trained BCG thresholds.
        
        Args:
            items: List of dicts with 'id' and 'price'
            
        Returns:
            Dict of item_id -> category string (e.g., "⭐ Star")
        """
        # 1. Load classifier
        classifier_path = Path("models/bcg_classifier.joblib")
        if not classifier_path.exists():
            return {}
        
        try:
            classifier = MenuClassifier.load(classifier_path)
        except Exception:
            return {}
            
        # 2. Get demand predictions (daily)
        # We need estimated TOTAL orders to match training scale (approx 2 years / 730 days)
        daily_demand = self.predict_demand_for_items(items)
        
        results = {}
        for item in items:
            item_id = item.get('id')
            price = item.get('price', 0)
            
            # Scale daily demand to match the timeframe used for training thresholds
            estimated_orders = daily_demand.get(item_id, 0) * 730 
            
            # 3. Classify
            if classifier.thresholds:
                high_pop = estimated_orders >= classifier.thresholds.popularity
                high_profit = price >= classifier.thresholds.profitability
                
                if high_pop and high_profit:
                    category = "⭐ Star"
                elif high_pop and not high_profit:
                    category = "🐴 Plowhorse"
                elif not high_pop and high_profit:
                    category = "❓ Puzzle"
                else:
                    category = "🐕 Dog"
            else:
                category = "Unknown"
                
            results[item_id] = category
            
        return results

    def get_executive_summary(self) -> Dict:
        """
        Generate executive summary of key findings.
        
        Returns:
            Dictionary with summary metrics and insights
        """
        # Compute total_orders from actual orders table, NOT from most_ordered sum
        fct_orders = self._datasets.get('fct_orders', pd.DataFrame())
        fct_order_items = self._datasets.get('fct_order_items', pd.DataFrame())
        
        total_orders = len(fct_orders) if not fct_orders.empty else 0
        total_order_items = len(fct_order_items) if not fct_order_items.empty else 0
        
        # Active restaurants = those that actually have orders
        if not fct_orders.empty and 'place_id' in fct_orders.columns:
            active_restaurants = int(fct_orders['place_id'].nunique())
        else:
            active_restaurants = len(self._datasets.get('dim_places', []))
        
        summary = {
            'data_overview': {
                'total_items': len(self._datasets.get('dim_items', [])),
                'total_restaurants': active_restaurants,
                'total_orders': total_orders,
                'total_order_items': total_order_items,
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
