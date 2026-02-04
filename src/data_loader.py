"""
Data loading and preprocessing utilities.

This module handles:
- Loading CSV files from the data directory
- Basic data cleaning and type conversions
- Merging datasets for analysis
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from . import config


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all required datasets from the configured data directory.
    
    Returns:
        Dictionary mapping dataset names to DataFrames
    """
    print("Loading datasets...")
    
    datasets = {}
    for name, filepath in config.DATA_FILES.items():
        try:
            datasets[name] = pd.read_csv(filepath, low_memory=False)
            print(f"  ✓ {name}: {datasets[name].shape[0]:,} rows × {datasets[name].shape[1]} cols")
        except FileNotFoundError:
            print(f"  ✗ {name}: File not found at {filepath}")
            datasets[name] = pd.DataFrame()
    
    return datasets


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a single dataset by name.
    
    Args:
        name: Dataset name (e.g., 'dim_items', 'most_ordered')
        
    Returns:
        DataFrame containing the dataset
    """
    filepath = config.DATA_FILES.get(name)
    if filepath is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(config.DATA_FILES.keys())}")
    
    return pd.read_csv(filepath, low_memory=False)


def prepare_menu_analysis_data(
    most_ordered: pd.DataFrame, 
    dim_items: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge and prepare data for menu engineering analysis.
    
    Args:
        most_ordered: Order frequency data
        dim_items: Item master data
        
    Returns:
        Merged DataFrame ready for analysis
    """
    # Select relevant columns from dim_items
    item_cols = ['id', 'title', 'price', 'description', 'section_id', 
                 'status', 'purchases', 'vat']
    item_cols = [c for c in item_cols if c in dim_items.columns]
    
    # Merge datasets
    menu_analysis = most_ordered.merge(
        dim_items[item_cols],
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
    
    return menu_analysis


def prepare_place_analysis_data(
    most_ordered: pd.DataFrame,
    dim_places: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare data for restaurant performance analysis.
    
    Args:
        most_ordered: Order data by place
        dim_places: Place/restaurant master data
        
    Returns:
        DataFrame with aggregated place performance metrics
    """
    # Aggregate orders by place
    place_performance = most_ordered.groupby('place_id').agg({
        'order_count': 'sum',
        'item_id': 'nunique'
    }).reset_index()
    place_performance.columns = ['place_id', 'total_orders', 'unique_items']
    
    # Select relevant place columns
    place_cols = ['id', 'title', 'country', 'currency', 'area', 
                  'street_address', 'rating', 'votes', 'activated', 'active']
    place_cols = [c for c in place_cols if c in dim_places.columns]
    
    # Merge with place details
    place_analysis = place_performance.merge(
        dim_places[place_cols],
        left_on='place_id',
        right_on='id',
        how='left'
    )
    
    return place_analysis


def aggregate_item_performance(menu_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate menu data by item across all locations.
    
    Args:
        menu_analysis: Merged menu data
        
    Returns:
        DataFrame with one row per item and aggregated metrics
    """
    item_performance = menu_analysis.groupby(['item_id', 'item_name']).agg({
        'order_count': 'sum',
        'price': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    return item_performance


def print_dataset_summary(datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Print a summary of all loaded datasets.
    
    Args:
        datasets: Dictionary of dataset name to DataFrame
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    total_rows = 0
    for name, df in datasets.items():
        rows = len(df)
        cols = len(df.columns)
        total_rows += rows
        print(f"  {name:20} {rows:>10,} rows × {cols:>3} columns")
    
    print("-" * 60)
    print(f"  {'TOTAL':20} {total_rows:>10,} rows")
