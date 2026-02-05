"""
File: test_helpers.py
Description: Unit tests for utility helper functions.
Author: FlavorFlow Team

Run with: pytest tests/test_helpers.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.helpers import (
    convert_unix_timestamp,
    format_currency,
    categorize_performance,
    calculate_percentile,
    safe_division,
    flatten_dict,
    validate_dataframe,
    clean_string,
    get_memory_usage
)


class TestConvertUnixTimestamp:
    """Tests for convert_unix_timestamp function."""
    
    def test_valid_timestamp(self):
        """Test conversion of a valid UNIX timestamp."""
        result = convert_unix_timestamp(1609459200)
        assert result is not None
        assert "2021" in result
    
    def test_none_timestamp(self):
        """Test handling of None input."""
        result = convert_unix_timestamp(None)
        assert result is None
    
    def test_nan_timestamp(self):
        """Test handling of NaN input."""
        result = convert_unix_timestamp(float('nan'))
        assert result is None
    
    def test_custom_format(self):
        """Test custom format string."""
        result = convert_unix_timestamp(1609459200, format_string='%Y-%m-%d')
        assert result is not None
        assert len(result) == 10  # YYYY-MM-DD format
    
    def test_invalid_timestamp(self):
        """Test handling of invalid timestamp."""
        result = convert_unix_timestamp("invalid")
        assert result is None


class TestFormatCurrency:
    """Tests for format_currency function."""
    
    def test_basic_formatting(self):
        """Test basic currency formatting."""
        result = format_currency(1234.5)
        assert result == "1,234.50 DKK"
    
    def test_zero_value(self):
        """Test zero value formatting."""
        result = format_currency(0)
        assert result == "0.00 DKK"
    
    def test_none_value(self):
        """Test None value handling."""
        result = format_currency(None)
        assert result == "0.00 DKK"
    
    def test_custom_currency(self):
        """Test custom currency code."""
        result = format_currency(100, currency='USD')
        assert 'USD' in result
    
    def test_custom_decimals(self):
        """Test custom decimal places."""
        result = format_currency(100.123, decimal_places=3)
        assert '100.123' in result
    
    def test_large_number(self):
        """Test large number formatting with commas."""
        result = format_currency(1000000)
        assert "1,000,000.00" in result


class TestCategorizePerformance:
    """Tests for categorize_performance function."""
    
    def test_star_category(self):
        """Test Star classification (high pop, high prof)."""
        result = categorize_performance(100, 50, 80, 40)
        assert result == '⭐ Star'
    
    def test_plowhorse_category(self):
        """Test Plowhorse classification (high pop, low prof)."""
        result = categorize_performance(100, 30, 80, 40)
        assert result == '🐴 Plowhorse'
    
    def test_puzzle_category(self):
        """Test Puzzle classification (low pop, high prof)."""
        result = categorize_performance(50, 50, 80, 40)
        assert result == '❓ Puzzle'
    
    def test_dog_category(self):
        """Test Dog classification (low pop, low prof)."""
        result = categorize_performance(50, 30, 80, 40)
        assert result == '🐕 Dog'
    
    def test_boundary_values(self):
        """Test exact threshold values (should be included in high)."""
        result = categorize_performance(80, 40, 80, 40)
        assert result == '⭐ Star'


class TestCalculatePercentile:
    """Tests for calculate_percentile function."""
    
    def test_median(self):
        """Test median (50th percentile) calculation."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = calculate_percentile(series, 50)
        assert result == 3.0
    
    def test_25th_percentile(self):
        """Test 25th percentile calculation."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = calculate_percentile(series, 25)
        assert result == 3.25
    
    def test_empty_series(self):
        """Test empty series handling."""
        series = pd.Series([], dtype=float)
        result = calculate_percentile(series, 50)
        assert pd.isna(result)


class TestSafeDivision:
    """Tests for safe_division function."""
    
    def test_normal_division(self):
        """Test normal division."""
        result = safe_division(10, 2)
        assert result == 5.0
    
    def test_zero_denominator(self):
        """Test zero denominator returns default."""
        result = safe_division(10, 0)
        assert result == 0.0
    
    def test_custom_default(self):
        """Test custom default value."""
        result = safe_division(10, 0, default=-1)
        assert result == -1
    
    def test_nan_denominator(self):
        """Test NaN denominator."""
        result = safe_division(10, float('nan'))
        assert result == 0.0


class TestFlattenDict:
    """Tests for flatten_dict function."""
    
    def test_simple_nested(self):
        """Test simple nested dictionary."""
        d = {'a': {'b': 1, 'c': 2}}
        result = flatten_dict(d)
        assert result == {'a_b': 1, 'a_c': 2}
    
    def test_flat_dict(self):
        """Test already flat dictionary."""
        d = {'a': 1, 'b': 2}
        result = flatten_dict(d)
        assert result == {'a': 1, 'b': 2}
    
    def test_deep_nested(self):
        """Test deeply nested dictionary."""
        d = {'a': {'b': {'c': 1}}}
        result = flatten_dict(d)
        assert result == {'a_b_c': 1}
    
    def test_custom_separator(self):
        """Test custom separator."""
        d = {'a': {'b': 1}}
        result = flatten_dict(d, sep='.')
        assert result == {'a.b': 1}


class TestValidateDataframe:
    """Tests for validate_dataframe function."""
    
    def test_valid_dataframe(self):
        """Test DataFrame with all required columns."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        result = validate_dataframe(df, ['a', 'b'])
        assert result is True
    
    def test_missing_columns(self):
        """Test DataFrame missing required columns."""
        df = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError) as excinfo:
            validate_dataframe(df, ['a', 'b'])
        assert 'missing columns' in str(excinfo.value).lower()


class TestCleanString:
    """Tests for clean_string function."""
    
    def test_normal_string(self):
        """Test normal string cleaning."""
        result = clean_string("  Hello World  ")
        assert result == "hello world"
    
    def test_none_value(self):
        """Test None value handling."""
        result = clean_string(None)
        assert result == ""
    
    def test_nan_value(self):
        """Test NaN value handling."""
        result = clean_string(float('nan'))
        assert result == ""


class TestGetMemoryUsage:
    """Tests for get_memory_usage function."""
    
    def test_small_dataframe(self):
        """Test memory usage of small DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = get_memory_usage(df)
        assert any(unit in result for unit in ['B', 'KB', 'MB', 'GB'])
    
    def test_larger_dataframe(self):
        """Test memory usage of larger DataFrame."""
        df = pd.DataFrame({'a': range(10000)})
        result = get_memory_usage(df)
        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestHelpersIntegration:
    """Integration tests combining multiple helpers."""
    
    def test_bcg_categorization_workflow(self):
        """Test full BCG categorization workflow."""
        # Create sample data
        data = {
            'item_id': [1, 2, 3, 4],
            'order_count': [100, 90, 20, 15],
            'price': [50, 30, 60, 25]
        }
        df = pd.DataFrame(data)
        
        # Calculate thresholds
        pop_threshold = calculate_percentile(df['order_count'], 50)
        prof_threshold = calculate_percentile(df['price'], 50)
        
        # Categorize items
        df['category'] = df.apply(
            lambda row: categorize_performance(
                row['order_count'],
                row['price'],
                pop_threshold,
                prof_threshold
            ),
            axis=1
        )
        
        # Verify results
        assert df[df['item_id'] == 1]['category'].values[0] == '⭐ Star'
        assert df[df['item_id'] == 4]['category'].values[0] == '🐕 Dog'
    
    def test_currency_formatting_in_report(self):
        """Test currency formatting for report generation."""
        revenues = [1234.5, 5678.9, None, 0]
        formatted = [format_currency(r) for r in revenues]
        
        assert formatted[0] == "1,234.50 DKK"
        assert formatted[2] == "0.00 DKK"
        assert formatted[3] == "0.00 DKK"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
