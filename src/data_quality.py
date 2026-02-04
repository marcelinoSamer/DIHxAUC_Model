"""
Data quality assessment utilities.

This module provides functions to:
- Assess missing values and data completeness
- Identify duplicates
- Generate data quality reports
"""

import pandas as pd
from typing import Dict, Tuple, Optional


def assess_data_quality(df: pd.DataFrame, name: str, verbose: bool = True) -> pd.DataFrame:
    """
    Comprehensive data quality assessment for a DataFrame.
    
    Args:
        df: DataFrame to assess
        name: Name for the report header
        verbose: Whether to print the report
        
    Returns:
        DataFrame with missing value statistics
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"📋 DATA QUALITY REPORT: {name}")
        print(f"{'=' * 60}")
        
        # Basic stats
        print(f"Total Records: {len(df):,}")
        print(f"Total Columns: {len(df.columns)}")
    
    # Missing values analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing, 
        'Percent': missing_pct
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values(
        'Percent', ascending=False
    )
    
    if verbose:
        if len(missing_df) > 0:
            print(f"\n⚠️ Columns with Missing Values (Top 10):")
            print(missing_df.head(10).to_string())
        else:
            print("\n✅ No missing values!")
        
        # Duplicates
        dups = df.duplicated().sum()
        print(f"\n🔄 Duplicate Rows: {dups:,} ({dups / len(df) * 100:.2f}%)")
    
    return missing_df


def get_data_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of data types in the DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with column names and their data types
    """
    return pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values,
        'Unique': [df[col].nunique() for col in df.columns]
    })


def check_numeric_ranges(
    df: pd.DataFrame, 
    numeric_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    Check ranges and distributions of numeric columns.
    
    Args:
        df: DataFrame to analyze
        numeric_cols: List of columns to check (auto-detect if None)
        
    Returns:
        DataFrame with statistics for each numeric column
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    stats = []
    for col in numeric_cols:
        col_stats = {
            'Column': col,
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Std': df[col].std(),
            'Zeros': (df[col] == 0).sum(),
            'Negatives': (df[col] < 0).sum()
        }
        stats.append(col_stats)
    
    return pd.DataFrame(stats)


def identify_outliers(
    df: pd.DataFrame, 
    column: str, 
    method: str = 'iqr', 
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Identify outliers in a numeric column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        method: 'iqr' for interquartile range or 'zscore' for z-score
        threshold: IQR multiplier or z-score threshold
        
    Returns:
        DataFrame containing only the outlier rows
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        z_scores = (df[column] - mean) / std
        outliers = df[abs(z_scores) > threshold]
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
    
    return outliers


def generate_quality_report(datasets: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """
    Generate a comprehensive quality report for multiple datasets.
    
    Args:
        datasets: Dictionary mapping names to DataFrames
        
    Returns:
        Dictionary with quality metrics for each dataset
    """
    report = {}
    
    for name, df in datasets.items():
        missing_df = assess_data_quality(df, name, verbose=False)
        
        report[name] = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_columns': len(missing_df),
            'total_missing': df.isnull().sum().sum(),
            'completeness': (1 - df.isnull().sum().sum() / df.size) * 100,
            'duplicates': df.duplicated().sum(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    return report


def print_quality_summary(report: Dict[str, dict]) -> None:
    """
    Print a formatted summary of the quality report.
    
    Args:
        report: Quality report from generate_quality_report()
    """
    print("\n" + "=" * 80)
    print("📊 DATA QUALITY SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<25} {'Rows':>10} {'Complete%':>10} {'Duplicates':>12} {'Memory':>10}")
    print("-" * 80)
    
    for name, metrics in report.items():
        print(f"{name:<25} {metrics['rows']:>10,} {metrics['completeness']:>9.1f}% "
              f"{metrics['duplicates']:>12,} {metrics['memory_mb']:>8.2f} MB")
