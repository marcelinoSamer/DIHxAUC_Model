"""
Export utilities for saving analysis results.

This module handles:
- Saving DataFrames to CSV
- Generating summary reports
- Exporting visualizations
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from . import config


def export_menu_engineering_results(
    item_performance: pd.DataFrame,
    output_dir: Path = None
) -> Path:
    """
    Export menu engineering classification results.
    
    Args:
        item_performance: DataFrame with BCG categories
        output_dir: Output directory (uses config default if None)
        
    Returns:
        Path to saved file
    """
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / 'results_menu_engineering.csv'
    item_performance.to_csv(filepath, index=False)
    print(f"✅ Menu engineering results saved: {filepath}")
    
    return filepath


def export_recommendations(output_dir: Path = None) -> Path:
    """
    Export standard recommendations for each BCG category.
    
    Args:
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    recommendations = pd.DataFrame({
        'Category': ['Star Items', 'Plowhorse Items', 'Puzzle Items', 'Dog Items'],
        'Action': [
            'Promote heavily, protect margins, feature prominently',
            'Consider 10-15% price increase, bundle with lower-margin items',
            'Increase visibility, improve descriptions, run promotions',
            'Re-engineer recipe, bundle with stars, or remove from menu'
        ],
        'Expected_Impact': ['+15% visibility', '+10% margin', '+20% orders', '-25% waste']
    })
    
    filepath = output_dir / 'results_recommendations.csv'
    recommendations.to_csv(filepath, index=False)
    print(f"✅ Recommendations saved: {filepath}")
    
    return filepath


def export_cluster_results(
    item_performance_clustered: pd.DataFrame,
    output_dir: Path = None
) -> Path:
    """
    Export item clustering results.
    
    Args:
        item_performance_clustered: DataFrame with cluster labels
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / 'results_item_clusters.csv'
    item_performance_clustered.to_csv(filepath, index=False)
    print(f"✅ Item clusters saved: {filepath}")
    
    return filepath


def export_restaurant_performance(
    place_analysis: pd.DataFrame,
    output_dir: Path = None
) -> Path:
    """
    Export restaurant performance results.
    
    Args:
        place_analysis: DataFrame with restaurant metrics
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / 'results_restaurant_performance.csv'
    place_analysis.to_csv(filepath, index=False)
    print(f"✅ Restaurant performance saved: {filepath}")
    
    return filepath


def export_all_results(
    item_performance: pd.DataFrame,
    item_performance_clustered: pd.DataFrame,
    place_analysis: pd.DataFrame,
    output_dir: Path = None
) -> Dict[str, Path]:
    """
    Export all analysis results.
    
    Args:
        item_performance: Menu engineering results
        item_performance_clustered: Clustering results
        place_analysis: Restaurant performance data
        output_dir: Output directory
        
    Returns:
        Dictionary mapping result names to file paths
    """
    output_dir = output_dir or config.OUTPUT_DIR
    
    print("=" * 60)
    print("💾 EXPORTING RESULTS")
    print("=" * 60)
    
    paths = {
        'menu_engineering': export_menu_engineering_results(item_performance, output_dir),
        'recommendations': export_recommendations(output_dir),
        'clusters': export_cluster_results(item_performance_clustered, output_dir),
        'restaurants': export_restaurant_performance(place_analysis, output_dir)
    }
    
    print("\n" + "=" * 60)
    print("🎉 ALL EXPORTS COMPLETE!")
    print("=" * 60)
    
    return paths


def print_export_summary(paths: Dict[str, Path]) -> None:
    """
    Print summary of exported files.
    
    Args:
        paths: Dictionary from export_all_results()
    """
    print("""
Files created:
  📊 results_menu_engineering.csv     - BCG matrix classification
  📋 results_recommendations.csv      - Action items for each category
  🎯 results_item_clusters.csv        - ML clustering results
  📍 results_restaurant_performance.csv - Location analysis
  
Visualizations:
  📈 menu_engineering_matrix.png      - BCG matrix scatter plot
  💰 pricing_analysis.png             - Price distribution charts
  🎯 campaign_analysis.png            - Campaign effectiveness
  📊 clustering_analysis.png          - Item segmentation
""")
