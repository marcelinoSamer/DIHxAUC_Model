"""
File: inventory_analysis_service.py
Description: Orchestrates the full inventory optimization pipeline.
Dependencies: All model classes
Author: FlavorFlow Team

This service runs the complete analysis:
1. Load order data (2M+ transactions)
2. Analyze customer behavior patterns
3. Train demand forecasting model
4. Calculate optimal inventory levels
5. Generate alerts and visualizations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from src.models.demand_forecaster import DemandForecaster
from src.models.inventory_optimizer import InventoryOptimizer, StockAlertGenerator
from src.models.customer_analyzer import CustomerBehaviorAnalyzer
from src.utils.data_loader import DataLoader


class InventoryAnalysisService:
    """
    Main service for inventory optimization analysis.
    
    Orchestrates the complete pipeline from raw order data
    to actionable inventory recommendations.
    
    Pipeline:
        Orders (2M) → Behavior Analysis → Demand Forecast → Inventory Optimization
    
    Example:
        >>> service = InventoryAnalysisService(data_dir='data/')
        >>> service.run_full_analysis()
        >>> service.export_results('docs/')
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the service.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir) if data_dir else Path('data')
        self.data_loader = DataLoader(self.data_dir)
        
        # Initialize components
        self.forecaster = DemandForecaster()
        self.optimizer = InventoryOptimizer(lead_time_days=2, service_level=0.95)
        self.behavior_analyzer = CustomerBehaviorAnalyzer()
        
        # Data storage
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._results: Dict = {}
    
    def load_data(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets.
        
        Returns:
            Dictionary of loaded DataFrames
        """
        if verbose:
            print("=" * 60)
            print("📂 LOADING ORDER DATA FOR INVENTORY ANALYSIS")
            print("=" * 60)
        
        # Load order tables (these are the big ones)
        tables_to_load = [
            'fct_orders',
            'fct_order_items', 
            'dim_items',
            'dim_menu_items',
            'dim_places',
            'dim_bill_of_materials',
            'fct_inventory_reports'
        ]
        
        for table in tables_to_load:
            try:
                self._datasets[table] = self.data_loader.load_table(table)
                if verbose:
                    print(f"   ✅ {table}: {len(self._datasets[table]):,} rows")
            except FileNotFoundError:
                if verbose:
                    print(f"   ⚠️ {table}: not found")
        
        return self._datasets
    
    def analyze_customer_behavior(self, verbose: bool = True) -> Dict:
        """
        Run customer behavior analysis on order data.
        
        Returns:
            Behavior analysis results
        """
        if verbose:
            print("\n" + "=" * 60)
            print("👥 ANALYZING CUSTOMER BEHAVIOR")
            print("=" * 60)
        
        orders = self._datasets.get('fct_orders')
        order_items = self._datasets.get('fct_order_items')
        
        if orders is None or order_items is None:
            raise ValueError("Order data not loaded")
        
        self.behavior_analyzer.load_orders(orders, order_items, verbose=verbose)
        
        # Run analyses
        temporal = self.behavior_analyzer.analyze_temporal_patterns()
        purchase = self.behavior_analyzer.analyze_purchase_patterns()
        
        if verbose:
            print(self.behavior_analyzer.generate_insights_report())
        
        self._results['behavior'] = {
            'temporal': temporal,
            'purchase': purchase,
            'summary': self.behavior_analyzer.get_behavior_summary()
        }
        
        return self._results['behavior']
    
    def train_demand_model(self, verbose: bool = True) -> Dict:
        """
        Train the demand forecasting model on order history.
        
        Returns:
            Training results and metrics
        """
        if verbose:
            print("\n" + "=" * 60)
            print("🤖 TRAINING DEMAND FORECASTING MODEL")
            print("=" * 60)
        
        orders = self._datasets.get('fct_orders')
        order_items = self._datasets.get('fct_order_items')
        
        if orders is None or order_items is None:
            raise ValueError("Order data not loaded")
        
        # Prepare training data
        self.forecaster.load_and_prepare_data(
            orders, order_items, verbose=verbose
        )
        
        # Train model
        self.forecaster.fit(verbose=verbose)
        
        # Get feature importance
        importance = self.forecaster.get_feature_importance()
        
        if verbose:
            print("\n📊 Feature Importance:")
            for _, row in importance.head(5).iterrows():
                bar = '█' * int(row['importance'] * 50)
                print(f"   {row['feature']:>25}: {row['importance']:.3f} {bar}")
        
        self._results['demand_model'] = {
            'metrics': self.forecaster.metrics,
            'feature_importance': importance.to_dict('records'),
            'summary': self.forecaster.get_summary()
        }
        
        return self._results['demand_model']
    
    def optimize_inventory(
        self,
        current_inventory: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Calculate optimal inventory levels and generate alerts.
        
        Args:
            current_inventory: Current stock levels (simulated if None)
            
        Returns:
            Inventory optimization results
        """
        if verbose:
            print("\n" + "=" * 60)
            print("📦 OPTIMIZING INVENTORY LEVELS")
            print("=" * 60)
        
        # Get demand data
        daily_demand = self.forecaster.daily_demand
        items_df = self._datasets.get('dim_items')
        
        if daily_demand is None:
            raise ValueError("Demand model not trained")
        
        # Load demand statistics
        self.optimizer.load_demand_data(daily_demand, items_df)
        
        if verbose:
            opt_summary = self.optimizer.get_optimization_summary()
            print(f"\n   Items analyzed: {opt_summary['total_items_analyzed']:,}")
            print(f"   Lead time: {opt_summary['parameters']['lead_time_days']} days")
            print(f"   Service level: {opt_summary['parameters']['service_level']}")
        
        # Simulate current inventory if not provided
        if current_inventory is None:
            current_inventory = self._simulate_current_inventory()
        
        # Analyze inventory
        inventory_analysis = self.optimizer.analyze_inventory(current_inventory)
        
        # Get alerts
        alerts = self.optimizer.get_stock_alerts(current_inventory)
        
        if verbose:
            alert_report = StockAlertGenerator.generate_alert_report(alerts)
            print(alert_report)
        
        self._results['inventory'] = {
            'analysis': inventory_analysis,
            'alerts': alerts,
            'summary': self.optimizer.get_optimization_summary()
        }
        
        return self._results['inventory']
    
    def _simulate_current_inventory(self) -> pd.DataFrame:
        """
        Simulate current inventory levels for demonstration.
        
        In production, this would come from actual inventory system.
        """
        if self.optimizer.demand_stats is None:
            raise ValueError("Demand stats not calculated")
        
        np.random.seed(42)
        
        # Simulate: some items understocked, some overstocked
        inventory = self.optimizer.demand_stats[['item_id', 'reorder_point', 'safety_stock']].copy()
        
        # Random stock levels: some critical, some excess
        inventory['current_stock'] = np.random.choice(
            [0, 5, 10, 50, 100, 500],
            size=len(inventory),
            p=[0.05, 0.10, 0.15, 0.40, 0.20, 0.10]
        ) * inventory['safety_stock'].clip(1, None) / 10
        
        return inventory[['item_id', 'current_stock']]
    
    def run_full_analysis(self, verbose: bool = True) -> Dict:
        """
        Run the complete inventory optimization pipeline.
        
        Returns:
            All analysis results
        """
        start_time = datetime.now()
        
        if verbose:
            print("\n" + "=" * 60)
            print("🚀 RUNNING FULL INVENTORY OPTIMIZATION ANALYSIS")
            print("=" * 60)
            print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load data
        self.load_data(verbose=verbose)
        
        # Step 2: Analyze behavior
        self.analyze_customer_behavior(verbose=verbose)
        
        # Step 3: Train demand model
        self.train_demand_model(verbose=verbose)
        
        # Step 4: Optimize inventory
        self.optimize_inventory(verbose=verbose)
        
        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds()
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"   Total time: {total_time:.1f} seconds")
            print(f"   Orders processed: {len(self._datasets.get('fct_orders', [])):,}")
            print(f"   Order items processed: {len(self._datasets.get('fct_order_items', [])):,}")
        
        self._results['meta'] = {
            'completed_at': datetime.now().isoformat(),
            'total_time_seconds': total_time
        }
        
        return self._results
    
    def get_executive_summary(self) -> Dict:
        """
        Get executive summary of all analyses.
        
        Returns:
            Summary dictionary for reporting
        """
        summary = {
            'data_overview': {
                'total_orders': len(self._datasets.get('fct_orders', [])),
                'total_order_items': len(self._datasets.get('fct_order_items', [])),
                'restaurants': self._datasets.get('fct_orders', pd.DataFrame()).get('place_id', pd.Series()).nunique(),
                'menu_items': self._datasets.get('dim_items', pd.DataFrame()).shape[0]
            }
        }
        
        if 'behavior' in self._results:
            summary['customer_behavior'] = self._results['behavior']['summary']['temporal_insights']
            summary['purchase_patterns'] = self._results['behavior']['summary']['purchase_insights']
        
        if 'demand_model' in self._results:
            summary['model_performance'] = self._results['demand_model']['metrics']
        
        if 'inventory' in self._results:
            alerts = self._results['inventory']['alerts']
            summary['inventory_alerts'] = {
                'critical_items': len(alerts[alerts['status'].str.contains('Critical')]) if not alerts.empty else 0,
                'low_stock_items': len(alerts[alerts['status'].str.contains('Low')]) if not alerts.empty else 0,
                'excess_items': len(alerts[alerts['status'].str.contains('Excess')]) if not alerts.empty else 0
            }
        
        return summary
    
    def export_results(self, output_dir: Path) -> Dict[str, Path]:
        """
        Export all results to files.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary of file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        paths = {}
        
        # Export behavior analysis
        if 'behavior' in self._results:
            # Temporal patterns
            if 'temporal' in self._results['behavior']:
                hourly = pd.DataFrame(self._results['behavior']['temporal']['hourly'])
                path = output_dir / 'temporal_patterns_hourly.csv'
                hourly.to_csv(path, index=False)
                paths['temporal_hourly'] = path
        
        # Export demand model results
        if 'demand_model' in self._results:
            importance = pd.DataFrame(self._results['demand_model']['feature_importance'])
            path = output_dir / 'feature_importance.csv'
            importance.to_csv(path, index=False)
            paths['feature_importance'] = path
        
        # Export inventory analysis
        if 'inventory' in self._results:
            analysis = self._results['inventory']['analysis']
            path = output_dir / 'inventory_analysis.csv'
            analysis.to_csv(path, index=False)
            paths['inventory_analysis'] = path
            
            alerts = self._results['inventory']['alerts']
            path = output_dir / 'inventory_alerts.csv'
            alerts.to_csv(path, index=False)
            paths['inventory_alerts'] = path
        
        # Export executive summary
        summary = self.get_executive_summary()
        summary_df = pd.json_normalize(summary, sep='_')
        path = output_dir / 'executive_summary.csv'
        summary_df.to_csv(path, index=False)
        paths['executive_summary'] = path
        
        print(f"\n📁 Results exported to {output_dir}/")
        for name, path in paths.items():
            print(f"   ✅ {path.name}")
        
        return paths
