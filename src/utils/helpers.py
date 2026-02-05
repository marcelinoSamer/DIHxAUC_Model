"""
File: helpers.py
Description: General utility functions for data processing and formatting.
Dependencies: pandas, datetime
Author: FlavorFlow Team

This module provides helper functions used across the application
for common operations like timestamp conversion, currency formatting,
and performance categorization.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging


def convert_unix_timestamp(
    timestamp: Union[int, float, None],
    format_string: str = '%Y-%m-%d %H:%M:%S'
) -> Optional[str]:
    """
    Convert UNIX timestamp to human-readable datetime string.
    
    Args:
        timestamp: UNIX timestamp (seconds since epoch)
        format_string: Output format string
    
    Returns:
        Formatted datetime string or None if invalid
    
    Example:
        >>> convert_unix_timestamp(1609459200)
        '2021-01-01 00:00:00'
    """
    if pd.isna(timestamp) or timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp)).strftime(format_string)
    except (ValueError, TypeError, OSError):
        return None


def format_currency(
    amount: Union[int, float, None],
    currency: str = 'DKK',
    decimal_places: int = 2
) -> str:
    """
    Format a number as currency string.
    
    Args:
        amount: The monetary amount
        currency: Currency code
        decimal_places: Number of decimal places
    
    Returns:
        Formatted currency string
    
    Example:
        >>> format_currency(1234.5)
        '1,234.50 DKK'
    """
    if pd.isna(amount) or amount is None:
        return f"0.00 {currency}"
    return f"{amount:,.{decimal_places}f} {currency}"


def categorize_performance(
    popularity: float,
    profitability: float,
    pop_threshold: float,
    prof_threshold: float
) -> str:
    """
    Categorize an item based on BCG matrix logic.
    
    Args:
        popularity: Popularity metric (e.g., order count)
        profitability: Profitability metric (e.g., price or margin)
        pop_threshold: Threshold for popularity
        prof_threshold: Threshold for profitability
    
    Returns:
        Category label (Star, Plowhorse, Puzzle, or Dog)
    
    Example:
        >>> categorize_performance(100, 50, 80, 40)
        '⭐ Star'
    """
    high_pop = popularity >= pop_threshold
    high_prof = profitability >= prof_threshold
    
    if high_pop and high_prof:
        return '⭐ Star'
    elif high_pop and not high_prof:
        return '🐴 Plowhorse'
    elif not high_pop and high_prof:
        return '❓ Puzzle'
    else:
        return '🐕 Dog'


def calculate_percentile(
    series: pd.Series,
    percentile: float = 50.0
) -> float:
    """
    Calculate percentile of a pandas Series.
    
    Args:
        series: Numeric series
        percentile: Percentile to calculate (0-100)
    
    Returns:
        Percentile value
    
    Example:
        >>> calculate_percentile(pd.Series([1, 2, 3, 4, 5]), 50)
        3.0
    """
    return series.quantile(percentile / 100)


def safe_division(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: float = 0.0
) -> float:
    """
    Perform division with protection against zero division.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division fails
    
    Returns:
        Result of division or default value
    
    Example:
        >>> safe_division(10, 0)
        0.0
        >>> safe_division(10, 2)
        5.0
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '_'
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between key levels
    
    Returns:
        Flattened dictionary
    
    Example:
        >>> flatten_dict({'a': {'b': 1, 'c': 2}})
        {'a_b': 1, 'a_c': 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level
        log_file: Optional file path for logging
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logging(logging.DEBUG, 'app.log')
        >>> logger.info('Application started')
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    return logging.getLogger('flavorflow')


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> bool:
    """
    Validate that a DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    return True


def clean_string(s: Optional[str]) -> str:
    """
    Clean and normalize a string value.
    
    Args:
        s: Input string
    
    Returns:
        Cleaned string
    """
    if s is None or pd.isna(s):
        return ""
    return str(s).strip().lower()


def batch_iterator(items: List[Any], batch_size: int = 100):
    """
    Iterate over items in batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
    
    Yields:
        Batches of items
    
    Example:
        >>> for batch in batch_iterator(range(10), 3):
        ...     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get human-readable memory usage of a DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Formatted memory usage string
    """
    bytes_used = df.memory_usage(deep=True).sum()
    if bytes_used < 1024:
        return f"{bytes_used} B"
    elif bytes_used < 1024**2:
        return f"{bytes_used/1024:.2f} KB"
    elif bytes_used < 1024**3:
        return f"{bytes_used/1024**2:.2f} MB"
    else:
        return f"{bytes_used/1024**3:.2f} GB"
