"""
Campaign and Promotion Analysis Module.

This module provides functions for:
- Campaign type analysis
- Discount effectiveness measurement
- Redemption rate analysis
- Promotional ROI estimation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def analyze_campaigns(fct_campaigns: pd.DataFrame) -> Dict:
    """
    Perform comprehensive campaign analysis.
    
    Args:
        fct_campaigns: Campaign fact table with campaign details
        
    Returns:
        Dictionary containing campaign analysis results
    """
    results = {}
    
    # Campaign type distribution
    results['campaign_types'] = fct_campaigns['type'].value_counts()
    
    # Discount analysis
    discount_campaigns = fct_campaigns[
        fct_campaigns['discount'].notna() & (fct_campaigns['discount'] > 0)
    ]
    results['discount_campaigns'] = discount_campaigns
    results['total_discount_campaigns'] = len(discount_campaigns)
    results['avg_discount'] = discount_campaigns['discount'].mean()
    results['most_common_discount'] = discount_campaigns['discount'].mode().values[0] if len(discount_campaigns) > 0 else 0
    
    # Redemption analysis
    results['total_redemptions'] = fct_campaigns['used_redemptions'].sum()
    campaigns_with_redemptions = fct_campaigns[fct_campaigns['used_redemptions'] > 0]
    results['campaigns_with_redemptions'] = len(campaigns_with_redemptions)
    results['redemption_rate'] = len(campaigns_with_redemptions) / len(fct_campaigns) * 100
    
    # Top campaigns
    results['top_campaigns'] = fct_campaigns.nlargest(
        10, 'used_redemptions'
    )[['title', 'type', 'discount', 'used_redemptions', 'status']]
    
    return results


def print_campaign_analysis(results: Dict) -> None:
    """
    Print formatted campaign analysis results.
    
    Args:
        results: Output from analyze_campaigns()
    """
    print("=" * 60)
    print("🎯 CAMPAIGN & PROMOTION ANALYSIS")
    print("=" * 60)
    
    # Campaign types
    print("\n📊 CAMPAIGN TYPES:")
    for ctype, count in results['campaign_types'].items():
        print(f"  {ctype}: {count}")
    
    # Discount analysis
    print(f"\n💸 DISCOUNT ANALYSIS:")
    print(f"  Total Discount Campaigns: {results['total_discount_campaigns']}")
    print(f"  Average Discount: {results['avg_discount']:.1f}%")
    print(f"  Most Common Discount: {results['most_common_discount']:.0f}%")
    
    # Redemption analysis
    print(f"\n🎟️ REDEMPTION ANALYSIS:")
    print(f"  Total Redemptions: {results['total_redemptions']:,}")
    print(f"  Campaigns with Redemptions: {results['campaigns_with_redemptions']} "
          f"({results['redemption_rate']:.1f}%)")
    
    # Top campaigns
    print("\n🏆 TOP 10 CAMPAIGNS BY REDEMPTIONS:")
    print(results['top_campaigns'].to_string(index=False))


def analyze_discount_effectiveness(
    fct_campaigns: pd.DataFrame,
    bins: list = [0, 10, 20, 30, 50, 100]
) -> pd.DataFrame:
    """
    Analyze which discount ranges are most effective.
    
    Args:
        fct_campaigns: Campaign fact table
        bins: Discount percentage bins
        
    Returns:
        DataFrame with effectiveness metrics by discount range
    """
    discount_df = fct_campaigns[
        fct_campaigns['discount'].notna() & (fct_campaigns['discount'] > 0)
    ].copy()
    
    labels = [f"{bins[i]}-{bins[i+1]}%" for i in range(len(bins)-1)]
    discount_df['discount_range'] = pd.cut(
        discount_df['discount'], 
        bins=bins, 
        labels=labels
    )
    
    effectiveness = discount_df.groupby('discount_range').agg({
        'used_redemptions': ['count', 'sum', 'mean'],
        'discount': 'mean'
    }).round(2)
    
    effectiveness.columns = ['campaign_count', 'total_redemptions', 
                            'avg_redemptions', 'avg_discount']
    
    return effectiveness


def calculate_campaign_roi(
    fct_campaigns: pd.DataFrame,
    average_order_value: float = 100.0,
    margin_pct: float = 0.30
) -> pd.DataFrame:
    """
    Estimate ROI for campaigns based on redemptions and discounts.
    
    Note: This is an estimate. Actual ROI requires transaction-level data.
    
    Args:
        fct_campaigns: Campaign fact table
        average_order_value: Assumed average order value
        margin_pct: Assumed profit margin before discount
        
    Returns:
        DataFrame with ROI estimates
    """
    campaigns = fct_campaigns[
        (fct_campaigns['used_redemptions'] > 0) & 
        (fct_campaigns['discount'].notna())
    ].copy()
    
    # Estimate revenue
    campaigns['estimated_revenue'] = campaigns['used_redemptions'] * average_order_value
    
    # Estimate discount cost
    campaigns['discount_cost'] = (
        campaigns['estimated_revenue'] * campaigns['discount'] / 100
    )
    
    # Estimate profit
    campaigns['gross_profit'] = campaigns['estimated_revenue'] * margin_pct
    campaigns['net_profit'] = campaigns['gross_profit'] - campaigns['discount_cost']
    campaigns['roi'] = (
        campaigns['net_profit'] / campaigns['discount_cost'] * 100
    ).replace([np.inf, -np.inf], np.nan)
    
    return campaigns[[
        'title', 'type', 'discount', 'used_redemptions',
        'estimated_revenue', 'discount_cost', 'net_profit', 'roi'
    ]].sort_values('net_profit', ascending=False)


def identify_underperforming_campaigns(
    fct_campaigns: pd.DataFrame,
    min_redemption_rate: float = 0.01
) -> pd.DataFrame:
    """
    Identify campaigns with low redemption rates.
    
    Args:
        fct_campaigns: Campaign fact table
        min_redemption_rate: Minimum expected redemption rate
        
    Returns:
        DataFrame of underperforming campaigns
    """
    campaigns = fct_campaigns.copy()
    
    # Calculate redemption rate if max_redemptions available
    if 'max_redemptions' in campaigns.columns:
        campaigns['redemption_rate'] = (
            campaigns['used_redemptions'] / campaigns['max_redemptions']
        ).fillna(0)
        
        underperforming = campaigns[
            campaigns['redemption_rate'] < min_redemption_rate
        ]
    else:
        # Use absolute redemptions as fallback
        avg_redemptions = campaigns['used_redemptions'].mean()
        underperforming = campaigns[
            campaigns['used_redemptions'] < avg_redemptions * 0.1
        ]
    
    return underperforming


def get_campaign_recommendations(results: Dict) -> pd.DataFrame:
    """
    Generate campaign optimization recommendations.
    
    Args:
        results: Output from analyze_campaigns()
        
    Returns:
        DataFrame with recommendations
    """
    recommendations = []
    
    # Based on average discount
    if results['avg_discount'] > 25:
        recommendations.append({
            'Area': 'Discounting',
            'Finding': f"Average discount ({results['avg_discount']:.1f}%) is high",
            'Recommendation': 'Consider smaller discounts with perceived value adds'
        })
    
    # Based on redemption rate
    if results['redemption_rate'] < 50:
        recommendations.append({
            'Area': 'Targeting',
            'Finding': f"Low redemption rate ({results['redemption_rate']:.1f}%)",
            'Recommendation': 'Improve targeting and campaign visibility'
        })
    
    # Based on campaign types
    if 'flash_sale' in results['campaign_types'].index:
        recommendations.append({
            'Area': 'Campaign Mix',
            'Finding': 'Flash sales present in portfolio',
            'Recommendation': 'Track cannibalization effects on regular sales'
        })
    
    return pd.DataFrame(recommendations)
