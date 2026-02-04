"""
Menu Engineering Analysis (BCG Matrix Classification).

This module implements the classic Menu Engineering approach that classifies
menu items into four categories based on popularity and profitability:

    ⭐ Stars      - High popularity, High profitability → Promote heavily
    🐴 Plowhorses - High popularity, Low profitability  → Re-engineer pricing  
    ❓ Puzzles    - Low popularity, High profitability  → Increase visibility
    🐕 Dogs       - Low popularity, Low profitability   → Consider removing
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

from . import config


# =============================================================================
# BCG MATRIX CLASSIFICATION
# =============================================================================

def classify_menu_item(
    row: pd.Series, 
    popularity_threshold: float, 
    price_threshold: float
) -> str:
    """
    Classify a single menu item into a BCG category.
    
    Args:
        row: DataFrame row with 'order_count' and 'price' columns
        popularity_threshold: Threshold for high/low popularity
        price_threshold: Threshold for high/low price
        
    Returns:
        Category string with emoji
    """
    high_popularity = row['order_count'] >= popularity_threshold
    high_profit = row['price'] >= price_threshold
    
    if high_popularity and high_profit:
        return '⭐ Star'
    elif high_popularity and not high_profit:
        return '🐴 Plowhorse'
    elif not high_popularity and high_profit:
        return '❓ Puzzle'
    else:
        return '🐕 Dog'


def perform_bcg_analysis(item_performance: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform full BCG Matrix analysis on item performance data.
    
    Args:
        item_performance: DataFrame with 'item_id', 'item_name', 
                         'order_count', 'price', 'revenue' columns
                         
    Returns:
        Tuple of (classified DataFrame, analysis metrics dict)
    """
    # Calculate thresholds using median split
    popularity_median = item_performance['order_count'].median()
    price_median = item_performance['price'].median()
    
    # Classify items
    item_performance = item_performance.copy()
    item_performance['category'] = item_performance.apply(
        lambda row: classify_menu_item(row, popularity_median, price_median),
        axis=1
    )
    
    # Calculate category summaries
    category_summary = item_performance.groupby('category').agg({
        'item_id': 'count',
        'order_count': 'sum',
        'revenue': 'sum'
    }).rename(columns={'item_id': 'item_count'})
    
    category_summary['pct_items'] = (
        category_summary['item_count'] / category_summary['item_count'].sum() * 100
    ).round(1)
    category_summary['pct_revenue'] = (
        category_summary['revenue'] / category_summary['revenue'].sum() * 100
    ).round(1)
    
    # Compile metrics
    metrics = {
        'popularity_threshold': popularity_median,
        'price_threshold': price_median,
        'total_items': len(item_performance),
        'total_revenue': item_performance['revenue'].sum(),
        'category_summary': category_summary
    }
    
    return item_performance, metrics


def print_bcg_results(item_performance: pd.DataFrame, metrics: Dict) -> None:
    """
    Print formatted BCG analysis results.
    
    Args:
        item_performance: Classified item DataFrame
        metrics: Analysis metrics dictionary
    """
    print("=" * 60)
    print("📊 MENU ENGINEERING CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"\nPopularity Threshold (median orders): {metrics['popularity_threshold']:.0f}")
    print(f"Price Threshold (median price): {metrics['price_threshold']:.2f}")
    
    print("\n📈 CATEGORY BREAKDOWN:")
    print(metrics['category_summary'].to_string())


# =============================================================================
# TOP PERFORMERS ANALYSIS
# =============================================================================

def get_top_stars(
    item_performance: pd.DataFrame, 
    n: int = 10
) -> pd.DataFrame:
    """Get top Star items by revenue."""
    stars = item_performance[item_performance['category'] == '⭐ Star']
    return stars.nlargest(n, 'revenue')[['item_name', 'order_count', 'price', 'revenue']]


def get_top_plowhorses(
    item_performance: pd.DataFrame, 
    n: int = 10
) -> pd.DataFrame:
    """Get top Plowhorse items by order count."""
    plowhorses = item_performance[item_performance['category'] == '🐴 Plowhorse']
    return plowhorses.nlargest(n, 'order_count')[['item_name', 'order_count', 'price', 'revenue']]


def get_top_puzzles(
    item_performance: pd.DataFrame, 
    n: int = 10
) -> pd.DataFrame:
    """Get top Puzzle items by price."""
    puzzles = item_performance[item_performance['category'] == '❓ Puzzle']
    return puzzles.nlargest(n, 'price')[['item_name', 'order_count', 'price', 'revenue']]


def get_bottom_dogs(
    item_performance: pd.DataFrame, 
    n: int = 10
) -> pd.DataFrame:
    """Get bottom Dog items by revenue."""
    dogs = item_performance[item_performance['category'] == '🐕 Dog']
    return dogs.nsmallest(n, 'revenue')[['item_name', 'order_count', 'price', 'revenue']]


def print_top_performers(item_performance: pd.DataFrame, n: int = 10) -> None:
    """
    Print top performers in each BCG category.
    
    Args:
        item_performance: Classified item DataFrame
        n: Number of items to show per category
    """
    print("=" * 60)
    print("🌟 TOP STAR ITEMS (High Popularity + High Price)")
    print("=" * 60)
    print(get_top_stars(item_performance, n).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🐴 TOP PLOWHORSES (High Popularity + Low Price)")
    print("💡 ACTION: Consider price increases to boost margins")
    print("=" * 60)
    print(get_top_plowhorses(item_performance, n).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("❓ TOP PUZZLES (Low Popularity + High Price)")
    print("💡 ACTION: Increase visibility, improve descriptions, promote")
    print("=" * 60)
    print(get_top_puzzles(item_performance, n).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🐕 BOTTOM DOGS (Low Popularity + Low Price)")
    print("💡 ACTION: Re-engineer, bundle with stars, or remove from menu")
    print("=" * 60)
    print(get_bottom_dogs(item_performance, n).to_string(index=False))


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def generate_recommendations(item_performance: pd.DataFrame) -> pd.DataFrame:
    """
    Generate actionable recommendations for each BCG category.
    
    Args:
        item_performance: Classified item DataFrame
        
    Returns:
        DataFrame with category recommendations
    """
    recommendations = pd.DataFrame({
        'Category': ['⭐ Star Items', '🐴 Plowhorse Items', '❓ Puzzle Items', '🐕 Dog Items'],
        'Action': [
            'Promote heavily, protect margins, feature prominently on menu',
            'Consider 10-15% price increase, bundle with lower-margin items',
            'Increase visibility, improve descriptions, run targeted promotions',
            'Re-engineer recipe, bundle with stars, or remove from menu'
        ],
        'Expected_Impact': ['+15% visibility', '+10% margin', '+20% orders', '-25% waste']
    })
    return recommendations


def get_category_action_items(
    item_performance: pd.DataFrame, 
    category: str
) -> pd.DataFrame:
    """
    Get specific action items for a category with item details.
    
    Args:
        item_performance: Classified item DataFrame
        category: Category to filter (e.g., '🐴 Plowhorse')
        
    Returns:
        DataFrame with items and suggested actions
    """
    items = item_performance[item_performance['category'] == category].copy()
    
    if category == '🐴 Plowhorse':
        items['suggested_price'] = (items['price'] * 1.12).round(2)  # 12% increase
        items['potential_revenue_gain'] = items['revenue'] * 0.12
    elif category == '❓ Puzzle':
        items['visibility_priority'] = pd.qcut(
            items['price'], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
    
    return items
