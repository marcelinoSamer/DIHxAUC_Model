"""
Pricing Analysis Module.

This module provides functions for:
- Price distribution analysis
- Price elasticity estimation
- Pricing optimization recommendations
- Price range segmentation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from . import config


def analyze_price_distribution(items: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of prices across menu items.
    
    Args:
        items: DataFrame with 'price' column
        
    Returns:
        Dictionary containing price statistics and distributions
    """
    # Filter valid prices
    items_with_price = items[items['price'].notna() & (items['price'] > 0)].copy()
    
    # Calculate statistics
    price_stats = items_with_price['price'].describe()
    
    # Create price ranges
    items_with_price['price_range'] = pd.cut(
        items_with_price['price'],
        bins=config.PRICE_BINS,
        labels=config.PRICE_LABELS
    )
    
    price_distribution = items_with_price['price_range'].value_counts().sort_index()
    
    return {
        'stats': price_stats,
        'distribution': price_distribution,
        'data': items_with_price,
        'count': len(items_with_price)
    }


def print_price_analysis(analysis: Dict) -> None:
    """
    Print formatted pricing analysis results.
    
    Args:
        analysis: Output from analyze_price_distribution()
    """
    print("=" * 60)
    print("💰 PRICING DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    stats = analysis['stats']
    print("\n📊 Price Statistics:")
    print(f"  Mean Price:   {stats['mean']:.2f}")
    print(f"  Median Price: {stats['50%']:.2f}")
    print(f"  Std Dev:      {stats['std']:.2f}")
    print(f"  Min Price:    {stats['min']:.2f}")
    print(f"  Max Price:    {stats['max']:.2f}")
    
    print("\n📈 Price Distribution:")
    total = analysis['count']
    for range_name, count in analysis['distribution'].items():
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {range_name:>8}: {count:>5} ({pct:>5.1f}%) {bar}")


def calculate_price_percentiles(
    items: pd.DataFrame, 
    percentiles: list = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
) -> pd.Series:
    """
    Calculate price percentiles for pricing strategy.
    
    Args:
        items: DataFrame with 'price' column
        percentiles: List of percentiles to calculate
        
    Returns:
        Series with percentile values
    """
    valid_prices = items.loc[items['price'] > 0, 'price']
    return valid_prices.quantile(percentiles)


def identify_pricing_anomalies(
    items: pd.DataFrame,
    lower_threshold: float = 0.05,
    upper_threshold: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify items with unusually low or high prices.
    
    Args:
        items: DataFrame with 'price' column
        lower_threshold: Percentile below which prices are considered too low
        upper_threshold: Percentile above which prices are considered too high
        
    Returns:
        Tuple of (underpriced items, overpriced items)
    """
    valid_items = items[items['price'] > 0].copy()
    
    lower_bound = valid_items['price'].quantile(lower_threshold)
    upper_bound = valid_items['price'].quantile(upper_threshold)
    
    underpriced = valid_items[valid_items['price'] < lower_bound]
    overpriced = valid_items[valid_items['price'] > upper_bound]
    
    return underpriced, overpriced


def suggest_price_adjustments(
    item_performance: pd.DataFrame,
    adjustment_pct: float = 0.12
) -> pd.DataFrame:
    """
    Suggest price adjustments for plowhorse items.
    
    Args:
        item_performance: DataFrame with BCG categories and prices
        adjustment_pct: Percentage to suggest for price increase
        
    Returns:
        DataFrame with suggested price changes
    """
    # Focus on plowhorses (high volume, low price)
    plowhorses = item_performance[
        item_performance['category'] == '🐴 Plowhorse'
    ].copy()
    
    plowhorses['current_price'] = plowhorses['price']
    plowhorses['suggested_price'] = (plowhorses['price'] * (1 + adjustment_pct)).round(2)
    plowhorses['price_change'] = plowhorses['suggested_price'] - plowhorses['current_price']
    plowhorses['potential_revenue'] = plowhorses['suggested_price'] * plowhorses['order_count']
    plowhorses['revenue_gain'] = plowhorses['potential_revenue'] - plowhorses['revenue']
    
    return plowhorses[[
        'item_name', 'order_count', 'current_price', 'suggested_price', 
        'price_change', 'revenue', 'potential_revenue', 'revenue_gain'
    ]].sort_values('revenue_gain', ascending=False)


def calculate_price_elasticity_estimate(
    items: pd.DataFrame,
    price_col: str = 'price',
    demand_col: str = 'purchases'
) -> float:
    """
    Estimate price elasticity using correlation between price and demand.
    
    Note: This is a simplified estimate. True elasticity requires 
    time-series or experimental data.
    
    Args:
        items: DataFrame with price and demand columns
        price_col: Name of price column
        demand_col: Name of demand/quantity column
        
    Returns:
        Correlation coefficient as elasticity proxy
    """
    valid = items[(items[price_col] > 0) & (items[demand_col] > 0)].copy()
    
    if len(valid) < 10:
        return np.nan
    
    # Log-log correlation as elasticity proxy
    log_price = np.log(valid[price_col])
    log_demand = np.log(valid[demand_col])
    
    elasticity = np.corrcoef(log_price, log_demand)[0, 1]
    
    return elasticity


def segment_by_price_tier(
    items: pd.DataFrame,
    n_tiers: int = 4
) -> pd.DataFrame:
    """
    Segment items into price tiers using quantiles.
    
    Args:
        items: DataFrame with 'price' column
        n_tiers: Number of price tiers to create
        
    Returns:
        DataFrame with 'price_tier' column added
    """
    items = items.copy()
    valid_mask = items['price'] > 0
    
    tier_labels = ['Budget', 'Value', 'Premium', 'Luxury'][:n_tiers]
    
    items.loc[valid_mask, 'price_tier'] = pd.qcut(
        items.loc[valid_mask, 'price'],
        q=n_tiers,
        labels=tier_labels,
        duplicates='drop'
    )
    
    return items
