"""
File: customer_analyzer.py
Description: Customer behavior analysis from order data.
Dependencies: pandas, numpy, scikit-learn
Author: FlavorFlow Team

This module analyzes customer purchasing patterns including:
- Temporal patterns (when they buy)
- Purchase patterns (what and how much)
- Customer segmentation
- Basket analysis (co-purchased items)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from itertools import combinations


class CustomerBehaviorAnalyzer:
    """
    Analyze customer purchasing behavior from order history.
    
    Key analyses:
    - Temporal patterns: peak hours, days, seasons
    - Purchase patterns: basket size, frequency, value
    - Co-purchase analysis: items bought together
    - Customer segmentation: RFM-like analysis
    
    Example:
        >>> analyzer = CustomerBehaviorAnalyzer()
        >>> analyzer.load_orders(orders_df, order_items_df)
        >>> patterns = analyzer.get_temporal_patterns()
        >>> segments = analyzer.segment_customers()
    """
    
    def __init__(self):
        self.orders: Optional[pd.DataFrame] = None
        self.order_items: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        self.customer_features: Optional[pd.DataFrame] = None
        self.temporal_patterns: Dict = {}
        self.basket_associations: List = []
    
    def load_orders(
        self,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        verbose: bool = True
    ) -> None:
        """
        Load and prepare order data for analysis.
        
        Args:
            orders_df: Orders table
            order_items_df: Order items table
            verbose: Print progress
        """
        self.orders = orders_df.copy()
        self.order_items = order_items_df.copy()
        
        # Parse timestamps
        self.orders['order_datetime'] = pd.to_datetime(
            self.orders['created'], unit='s', errors='coerce'
        )
        self.orders['order_date'] = self.orders['order_datetime'].dt.date
        self.orders['hour'] = self.orders['order_datetime'].dt.hour
        self.orders['day_of_week'] = self.orders['order_datetime'].dt.dayofweek
        self.orders['day_name'] = self.orders['order_datetime'].dt.day_name()
        self.orders['month'] = self.orders['order_datetime'].dt.month
        self.orders['is_weekend'] = self.orders['day_of_week'].isin([5, 6])
        
        # Merge for analysis
        self.merged_data = self.order_items.merge(
            self.orders[['id', 'place_id', 'order_datetime', 'hour', 
                        'day_of_week', 'day_name', 'month', 'is_weekend',
                        'total_amount', 'type']],
            left_on='order_id',
            right_on='id',
            how='inner'
        )
        
        if verbose:
            print(f"✅ Loaded {len(self.orders):,} orders with {len(self.order_items):,} items")
            date_range = self.orders['order_datetime'].agg(['min', 'max'])
            print(f"   Date range: {date_range['min']} to {date_range['max']}")
    
    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze when customers make purchases.
        
        Returns:
            Dictionary with temporal pattern analysis
        """
        if self.merged_data is None:
            raise ValueError("Load data first with load_orders()")
        
        # Hourly patterns
        hourly = self.merged_data.groupby('hour').agg({
            'order_id': 'nunique',
            'quantity': 'sum',
            'price': 'sum'
        }).reset_index()
        hourly.columns = ['hour', 'num_orders', 'total_quantity', 'total_revenue']
        
        peak_hour = hourly.loc[hourly['num_orders'].idxmax(), 'hour']
        
        # Daily patterns
        daily = self.merged_data.groupby('day_name').agg({
            'order_id': 'nunique',
            'quantity': 'sum'
        }).reset_index()
        
        # Reorder by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
        daily['day_order'] = daily['day_name'].map({d: i for i, d in enumerate(day_order)})
        daily = daily.sort_values('day_order')
        
        peak_day = daily.loc[daily['order_id'].idxmax(), 'day_name']
        
        # Weekend vs Weekday
        weekend_orders = self.merged_data[self.merged_data['is_weekend']]['order_id'].nunique()
        weekday_orders = self.merged_data[~self.merged_data['is_weekend']]['order_id'].nunique()
        
        # Monthly patterns
        monthly = self.merged_data.groupby('month').agg({
            'order_id': 'nunique',
            'quantity': 'sum'
        }).reset_index()
        
        self.temporal_patterns = {
            'hourly': hourly.to_dict('records'),
            'daily': daily[['day_name', 'order_id', 'quantity']].to_dict('records'),
            'monthly': monthly.to_dict('records'),
            'insights': {
                'peak_hour': int(peak_hour),
                'peak_hour_label': f"{int(peak_hour):02d}:00",
                'peak_day': peak_day,
                'weekend_pct': round(weekend_orders / (weekend_orders + weekday_orders) * 100, 1),
                'avg_orders_per_day': round(self.orders['id'].nunique() / 
                                           self.orders['order_date'].nunique(), 1)
            }
        }
        
        return self.temporal_patterns
    
    def analyze_purchase_patterns(self) -> Dict:
        """
        Analyze what and how much customers buy.
        
        Returns:
            Dictionary with purchase pattern analysis
        """
        if self.merged_data is None:
            raise ValueError("Load data first")
        
        # Basket analysis
        basket_stats = self.orders.groupby('id').agg({
            'total_amount': 'first'
        }).reset_index()
        
        items_per_order = self.order_items.groupby('order_id').agg({
            'item_id': 'nunique',
            'quantity': 'sum'
        }).reset_index()
        items_per_order.columns = ['order_id', 'unique_items', 'total_quantity']
        
        basket_stats = basket_stats.merge(items_per_order, left_on='id', right_on='order_id')
        
        # Top items
        top_items = self.order_items.groupby('item_id').agg({
            'quantity': 'sum',
            'order_id': 'nunique',
            'price': 'mean'
        }).reset_index()
        top_items.columns = ['item_id', 'total_qty', 'num_orders', 'avg_price']
        top_items = top_items.sort_values('total_qty', ascending=False)
        
        # Order type breakdown
        if 'type' in self.orders.columns:
            order_types = self.orders['type'].value_counts().to_dict()
        else:
            order_types = {}
        
        return {
            'basket_stats': {
                'avg_items_per_order': round(basket_stats['unique_items'].mean(), 1),
                'avg_quantity_per_order': round(basket_stats['total_quantity'].mean(), 1),
                'avg_order_value': round(basket_stats['total_amount'].mean(), 2),
                'median_order_value': round(basket_stats['total_amount'].median(), 2)
            },
            'top_items': top_items.head(20).to_dict('records'),
            'order_types': order_types,
            'total_unique_items': self.order_items['item_id'].nunique()
        }
    
    def find_co_purchased_items(
        self,
        min_support: float = 0.01,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Find items frequently bought together (association rules).
        
        Args:
            min_support: Minimum support threshold (0-1)
            top_n: Number of top associations to return
            
        Returns:
            DataFrame with item pairs and their co-purchase frequency
        """
        if self.order_items is None:
            raise ValueError("Load data first")
        
        # Get items per order
        order_baskets = self.order_items.groupby('order_id')['item_id'].apply(list)
        
        # Count pair occurrences
        pair_counts = Counter()
        total_orders = len(order_baskets)
        
        for basket in order_baskets:
            unique_items = list(set(basket))
            if len(unique_items) >= 2:
                for pair in combinations(sorted(unique_items), 2):
                    pair_counts[pair] += 1
        
        # Convert to DataFrame
        associations = pd.DataFrame([
            {
                'item_1': pair[0],
                'item_2': pair[1],
                'co_purchase_count': count,
                'support': count / total_orders
            }
            for pair, count in pair_counts.most_common(top_n * 10)
        ])
        
        # Filter by support
        associations = associations[associations['support'] >= min_support]
        
        # Calculate lift (requires individual item frequencies)
        item_freq = self.order_items.groupby('item_id')['order_id'].nunique() / total_orders
        
        def calc_lift(row):
            freq_1 = item_freq.get(row['item_1'], 0.001)
            freq_2 = item_freq.get(row['item_2'], 0.001)
            return row['support'] / (freq_1 * freq_2)
        
        associations['lift'] = associations.apply(calc_lift, axis=1)
        
        self.basket_associations = associations.head(top_n).to_dict('records')
        
        return associations.head(top_n)
    
    def segment_by_restaurant(self) -> pd.DataFrame:
        """
        Segment analysis by restaurant/location.
        
        Returns:
            DataFrame with per-restaurant metrics
        """
        if self.merged_data is None:
            raise ValueError("Load data first")
        
        restaurant_stats = self.merged_data.groupby('place_id').agg({
            'order_id': 'nunique',
            'quantity': 'sum',
            'price': 'sum',
            'item_id': 'nunique',
            'hour': lambda x: x.mode().iloc[0] if len(x) > 0 else 12
        }).reset_index()
        
        restaurant_stats.columns = [
            'place_id', 'total_orders', 'total_quantity',
            'total_revenue', 'unique_items_sold', 'peak_hour'
        ]
        
        restaurant_stats['avg_order_value'] = (
            restaurant_stats['total_revenue'] / restaurant_stats['total_orders']
        ).round(2)
        
        restaurant_stats['items_per_order'] = (
            restaurant_stats['unique_items_sold'] / restaurant_stats['total_orders']
        ).round(2)
        
        return restaurant_stats.sort_values('total_orders', ascending=False)
    
    def get_behavior_summary(self) -> Dict:
        """Get comprehensive behavior analysis summary."""
        temporal = self.analyze_temporal_patterns()
        purchase = self.analyze_purchase_patterns()
        
        return {
            'temporal_insights': temporal['insights'],
            'purchase_insights': purchase['basket_stats'],
            'order_types': purchase['order_types'],
            'data_coverage': {
                'total_orders': len(self.orders) if self.orders is not None else 0,
                'total_order_items': len(self.order_items) if self.order_items is not None else 0,
                'unique_items': purchase['total_unique_items'],
                'unique_restaurants': self.merged_data['place_id'].nunique() 
                    if self.merged_data is not None else 0
            }
        }
    
    def generate_insights_report(self) -> str:
        """Generate a human-readable insights report."""
        summary = self.get_behavior_summary()
        temporal = summary['temporal_insights']
        purchase = summary['purchase_insights']
        coverage = summary['data_coverage']
        
        report = f"""
{'='*60}
📊 CUSTOMER BEHAVIOR ANALYSIS REPORT
{'='*60}

📈 DATA COVERAGE
   Total Orders Analyzed: {coverage['total_orders']:,}
   Total Order Items: {coverage['total_order_items']:,}
   Unique Menu Items: {coverage['unique_items']:,}
   Restaurants: {coverage['unique_restaurants']:,}

⏰ TEMPORAL PATTERNS
   Peak Hour: {temporal['peak_hour_label']} ({temporal['peak_hour']}:00)
   Peak Day: {temporal['peak_day']}
   Weekend Orders: {temporal['weekend_pct']}% of total
   Avg Daily Orders: {temporal['avg_orders_per_day']:,}

🛒 PURCHASE PATTERNS
   Avg Items per Order: {purchase['avg_items_per_order']}
   Avg Quantity per Order: {purchase['avg_quantity_per_order']}
   Avg Order Value: {purchase['avg_order_value']:.2f}
   Median Order Value: {purchase['median_order_value']:.2f}

{'='*60}
"""
        return report
