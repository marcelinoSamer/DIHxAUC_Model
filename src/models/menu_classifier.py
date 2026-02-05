"""
File: menu_classifier.py
Description: BCG Matrix classifier for menu engineering analysis.
Dependencies: pandas, numpy
Author: FlavorFlow Team

This module implements the classic Menu Engineering BCG Matrix
classification for categorizing menu items into:
- Stars (High Popularity, High Profitability)
- Plowhorses (High Popularity, Low Profitability)
- Puzzles (Low Popularity, High Profitability)
- Dogs (Low Popularity, Low Profitability)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MenuCategory(Enum):
    """Menu item BCG categories."""
    STAR = "⭐ Star"
    PLOWHORSE = "🐴 Plowhorse"
    PUZZLE = "❓ Puzzle"
    DOG = "🐕 Dog"


@dataclass
class BCGThresholds:
    """Thresholds for BCG classification."""
    popularity: float
    profitability: float


@dataclass
class CategoryMetrics:
    """Metrics for a BCG category."""
    item_count: int
    total_orders: int
    total_revenue: float
    pct_items: float
    pct_revenue: float


class MenuClassifier:
    """
    BCG Matrix classifier for menu engineering.
    
    Classifies menu items based on the classic Menu Engineering
    approach using popularity (order volume) and profitability (price).
    
    Attributes:
        thresholds: BCGThresholds with popularity and price thresholds
        category_metrics: Dictionary of CategoryMetrics per category
    
    Example:
        >>> classifier = MenuClassifier()
        >>> classified = classifier.fit_transform(item_performance)
        >>> stars = classifier.get_category_items(classified, 'star')
    """
    
    CATEGORY_ACTIONS = {
        MenuCategory.STAR: {
            'action': 'Promote heavily, protect margins, feature prominently',
            'priority': 1,
            'expected_impact': '+15% visibility'
        },
        MenuCategory.PLOWHORSE: {
            'action': 'Consider 10-15% price increase, bundle with lower-margin items',
            'priority': 2,
            'expected_impact': '+10% margin'
        },
        MenuCategory.PUZZLE: {
            'action': 'Increase visibility, improve descriptions, run promotions',
            'priority': 3,
            'expected_impact': '+20% orders'
        },
        MenuCategory.DOG: {
            'action': 'Re-engineer, bundle with stars, or remove from menu',
            'priority': 4,
            'expected_impact': '-25% waste'
        }
    }
    
    def __init__(
        self,
        popularity_percentile: float = 50,
        price_percentile: float = 50
    ):
        """
        Initialize the menu classifier.
        
        Args:
            popularity_percentile: Percentile threshold for popularity
            price_percentile: Percentile threshold for price/profitability
        """
        self.popularity_percentile = popularity_percentile
        self.price_percentile = price_percentile
        self.thresholds: Optional[BCGThresholds] = None
        self.category_metrics: Dict[MenuCategory, CategoryMetrics] = {}
    
    def _classify_item(
        self, 
        row: pd.Series
    ) -> str:
        """
        Classify a single menu item.
        
        Args:
            row: DataFrame row with 'order_count' and 'price'
            
        Returns:
            Category string with emoji
        """
        high_popularity = row['order_count'] >= self.thresholds.popularity
        high_profit = row['price'] >= self.thresholds.profitability
        
        if high_popularity and high_profit:
            return MenuCategory.STAR.value
        elif high_popularity and not high_profit:
            return MenuCategory.PLOWHORSE.value
        elif not high_popularity and high_profit:
            return MenuCategory.PUZZLE.value
        else:
            return MenuCategory.DOG.value
    
    def fit(self, df: pd.DataFrame) -> 'MenuClassifier':
        """
        Fit the classifier by calculating thresholds.
        
        Args:
            df: DataFrame with 'order_count' and 'price' columns
            
        Returns:
            Self for method chaining
        """
        self.thresholds = BCGThresholds(
            popularity=df['order_count'].quantile(self.popularity_percentile / 100),
            profitability=df['price'].quantile(self.price_percentile / 100)
        )
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by adding category classification.
        
        Args:
            df: DataFrame with 'order_count' and 'price'
            
        Returns:
            DataFrame with 'category' column added
        """
        if self.thresholds is None:
            raise ValueError("Classifier must be fitted before transform")
        
        result = df.copy()
        result['category'] = result.apply(self._classify_item, axis=1)
        
        # Calculate category metrics
        self._calculate_category_metrics(result)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame with item performance data
            
        Returns:
            DataFrame with category classifications
        """
        return self.fit(df).transform(df)
    
    def _calculate_category_metrics(self, df: pd.DataFrame) -> None:
        """Calculate summary metrics for each category."""
        total_items = len(df)
        total_revenue = df['revenue'].sum()
        
        for category in MenuCategory:
            cat_data = df[df['category'] == category.value]
            self.category_metrics[category] = CategoryMetrics(
                item_count=len(cat_data),
                total_orders=cat_data['order_count'].sum(),
                total_revenue=cat_data['revenue'].sum(),
                pct_items=len(cat_data) / total_items * 100,
                pct_revenue=cat_data['revenue'].sum() / total_revenue * 100 if total_revenue > 0 else 0
            )
    
    def get_category_items(
        self, 
        df: pd.DataFrame, 
        category: str,
        top_n: int = 10,
        sort_by: str = 'revenue'
    ) -> pd.DataFrame:
        """
        Get top items in a specific category.
        
        Args:
            df: Classified DataFrame
            category: Category name ('star', 'plowhorse', 'puzzle', 'dog')
            top_n: Number of items to return
            sort_by: Column to sort by
            
        Returns:
            DataFrame with top items in the category
        """
        category_map = {
            'star': MenuCategory.STAR.value,
            'plowhorse': MenuCategory.PLOWHORSE.value,
            'puzzle': MenuCategory.PUZZLE.value,
            'dog': MenuCategory.DOG.value
        }
        
        cat_value = category_map.get(category.lower())
        if cat_value is None:
            raise ValueError(f"Invalid category: {category}")
        
        filtered = df[df['category'] == cat_value]
        
        if category.lower() == 'dog':
            return filtered.nsmallest(top_n, sort_by)
        return filtered.nlargest(top_n, sort_by)
    
    def get_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate actionable recommendations for all items.
        
        Args:
            df: Classified DataFrame
            
        Returns:
            DataFrame with recommendations
        """
        recommendations = []
        
        for _, row in df.iterrows():
            category_str = row['category']
            category = next(
                (c for c in MenuCategory if c.value == category_str),
                None
            )
            
            if category:
                action_info = self.CATEGORY_ACTIONS[category]
                recommendations.append({
                    'item_name': row.get('item_name', 'Unknown'),
                    'category': category_str,
                    'action': action_info['action'],
                    'priority': action_info['priority'],
                    'expected_impact': action_info['expected_impact']
                })
        
        return pd.DataFrame(recommendations).sort_values('priority')
    
    def get_pricing_suggestions(
        self, 
        df: pd.DataFrame,
        price_increase_pct: float = 0.12
    ) -> pd.DataFrame:
        """
        Generate pricing suggestions for plowhorses.
        
        Args:
            df: Classified DataFrame
            price_increase_pct: Suggested price increase percentage
            
        Returns:
            DataFrame with pricing suggestions
        """
        plowhorses = df[df['category'] == MenuCategory.PLOWHORSE.value].copy()
        
        plowhorses['suggested_price'] = (
            plowhorses['price'] * (1 + price_increase_pct)
        ).round(2)
        plowhorses['potential_revenue'] = (
            plowhorses['suggested_price'] * plowhorses['order_count']
        )
        plowhorses['revenue_gain'] = (
            plowhorses['potential_revenue'] - plowhorses['revenue']
        )
        
        return plowhorses[[
            'item_name', 'order_count', 'price', 'suggested_price',
            'revenue', 'potential_revenue', 'revenue_gain'
        ]].sort_values('revenue_gain', ascending=False)
    
    def get_summary_report(self) -> str:
        """
        Generate a text summary report.
        
        Returns:
            Formatted string with classification summary
        """
        if not self.category_metrics:
            return "No classification performed yet"
        
        lines = [
            "=" * 60,
            "📊 MENU ENGINEERING CLASSIFICATION RESULTS",
            "=" * 60,
            f"\nPopularity Threshold: {self.thresholds.popularity:.0f} orders",
            f"Price Threshold: {self.thresholds.profitability:.2f}",
            "\n📈 CATEGORY BREAKDOWN:\n"
        ]
        
        for category, metrics in self.category_metrics.items():
            lines.append(f"{category.value}:")
            lines.append(f"  Items: {metrics.item_count} ({metrics.pct_items:.1f}%)")
            lines.append(f"  Orders: {metrics.total_orders:,}")
            lines.append(f"  Revenue: {metrics.total_revenue:,.2f} ({metrics.pct_revenue:.1f}%)")
            lines.append("")
        
        return "\n".join(lines)
