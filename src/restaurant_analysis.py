"""
Restaurant Performance Analysis Module.

This module provides functions for:
- Restaurant-level performance analysis
- Location comparison
- Active vs inactive restaurant analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def analyze_restaurant_performance(place_analysis: pd.DataFrame) -> Dict:
    """
    Comprehensive restaurant performance analysis.
    
    Args:
        place_analysis: DataFrame with merged place performance data
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Basic stats
    results['total_restaurants'] = len(place_analysis)
    results['total_orders'] = place_analysis['total_orders'].sum()
    
    # Top performers
    results['top_restaurants'] = place_analysis.nlargest(
        10, 'total_orders'
    )[['title', 'total_orders', 'unique_items', 'rating', 'area']]
    
    # Distribution statistics
    results['order_percentiles'] = place_analysis['total_orders'].quantile(
        [0.25, 0.5, 0.75, 0.9]
    )
    
    # Active status analysis (if available)
    if 'active' in place_analysis.columns:
        results['active_summary'] = place_analysis.groupby('active').agg({
            'place_id': 'count',
            'total_orders': 'mean'
        }).round(1)
        results['active_summary'].columns = ['restaurant_count', 'avg_orders']
    
    return results


def print_restaurant_analysis(results: Dict) -> None:
    """
    Print formatted restaurant analysis results.
    
    Args:
        results: Output from analyze_restaurant_performance()
    """
    print("=" * 60)
    print("📍 RESTAURANT PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal Restaurants: {results['total_restaurants']:,}")
    print(f"Total Orders Tracked: {results['total_orders']:,}")
    
    print("\n🏆 TOP 10 RESTAURANTS BY ORDER VOLUME:")
    print(results['top_restaurants'].to_string(index=False))
    
    print("\n📊 ORDER VOLUME DISTRIBUTION:")
    percentiles = results['order_percentiles']
    print(f"  25th percentile: {percentiles[0.25]:.0f} orders")
    print(f"  50th percentile: {percentiles[0.5]:.0f} orders")
    print(f"  75th percentile: {percentiles[0.75]:.0f} orders")
    print(f"  90th percentile: {percentiles[0.9]:.0f} orders")
    
    if 'active_summary' in results:
        print("\n🔄 ACTIVE STATUS BREAKDOWN:")
        print(results['active_summary'])


def identify_underperforming_restaurants(
    place_analysis: pd.DataFrame,
    percentile_threshold: float = 0.25
) -> pd.DataFrame:
    """
    Identify restaurants performing below a given percentile.
    
    Args:
        place_analysis: Restaurant performance data
        percentile_threshold: Percentile below which restaurants are flagged
        
    Returns:
        DataFrame of underperforming restaurants
    """
    threshold = place_analysis['total_orders'].quantile(percentile_threshold)
    underperformers = place_analysis[
        place_analysis['total_orders'] < threshold
    ].copy()
    
    return underperformers.sort_values('total_orders')


def compare_regions(
    place_analysis: pd.DataFrame,
    region_col: str = 'area'
) -> pd.DataFrame:
    """
    Compare performance across different regions/areas.
    
    Args:
        place_analysis: Restaurant performance data
        region_col: Column containing region identifier
        
    Returns:
        DataFrame with regional performance comparison
    """
    if region_col not in place_analysis.columns:
        return pd.DataFrame()
    
    regional = place_analysis.groupby(region_col).agg({
        'place_id': 'count',
        'total_orders': ['sum', 'mean'],
        'unique_items': 'mean'
    }).round(2)
    
    regional.columns = ['restaurants', 'total_orders', 'avg_orders', 'avg_menu_items']
    regional = regional.sort_values('total_orders', ascending=False)
    
    return regional
