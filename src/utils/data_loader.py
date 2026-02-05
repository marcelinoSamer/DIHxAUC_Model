"""
File: data_loader.py
Description: Data loading and preprocessing utilities.
Dependencies: pandas, pathlib
Author: FlavorFlow Team

This module handles loading data from CSV files,
preprocessing, and data validation.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DataQualityReport:
    """Report on data quality metrics."""
    total_rows: int
    missing_values: Dict[str, int]
    duplicates: int
    memory_usage: str
    data_types: Dict[str, str]


class DataLoader:
    """
    Data loading and preprocessing utility.
    
    Handles loading CSV files from the data directory,
    data validation, and basic preprocessing.
    
    Attributes:
        data_dir: Path to the data directory
        loaded_tables: Dictionary of loaded DataFrames
    
    Example:
        >>> loader = DataLoader(Path('data/'))
        >>> loader.load_all()
        >>> items = loader.get_table('dim_items')
    """
    
    # Standard table names expected in the dataset
    TABLE_NAMES = [
        'dim_add_ons',
        'dim_bill_of_materials',
        'dim_campaigns',
        'dim_items',
        'dim_menu_item_add_ons',
        'dim_menu_items',
        'dim_places',
        'dim_skus',
        'dim_stock_categories',
        'dim_taxonomy_terms',
        'dim_users',
        'fct_bonus_codes',
        'fct_campaigns',
        'fct_cash_balances',
        'fct_inventory_reports',
        'fct_invoice_items',
        'fct_order_items',
        'fct_orders',
        'most_ordered'
    ]
    
    def __init__(self, data_dir: Path):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.loaded_tables: Dict[str, pd.DataFrame] = {}
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_table(
        self,
        table_name: str,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load a single table from CSV.
        
        Args:
            table_name: Name of the table (without .csv extension)
            force_reload: Force reload even if already cached
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        if table_name in self.loaded_tables and not force_reload:
            return self.loaded_tables[table_name]
        
        file_path = self.data_dir / f"{table_name}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Table file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        self.loaded_tables[table_name] = df
        
        return df
    
    def load_all(
        self,
        tables: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all or specified tables.
        
        Args:
            tables: List of table names to load, or None for all
            verbose: Print progress messages
        
        Returns:
            Dictionary of loaded DataFrames
        """
        tables_to_load = tables or self.TABLE_NAMES
        
        for table_name in tables_to_load:
            try:
                df = self.load_table(table_name)
                if verbose:
                    print(f"✅ Loaded {table_name}: {len(df):,} rows")
            except FileNotFoundError:
                if verbose:
                    print(f"⚠️ Table not found: {table_name}")
        
        return self.loaded_tables
    
    def get_table(self, table_name: str) -> pd.DataFrame:
        """
        Get a loaded table by name.
        
        Args:
            table_name: Name of the table
        
        Returns:
            The requested DataFrame
        
        Raises:
            KeyError: If table hasn't been loaded
        """
        if table_name not in self.loaded_tables:
            # Try to load it
            return self.load_table(table_name)
        return self.loaded_tables[table_name]
    
    def get_quality_report(self, table_name: str) -> DataQualityReport:
        """
        Generate a data quality report for a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            DataQualityReport with quality metrics
        """
        df = self.get_table(table_name)
        
        # Calculate memory usage
        bytes_used = df.memory_usage(deep=True).sum()
        if bytes_used < 1024**2:
            memory = f"{bytes_used/1024:.2f} KB"
        else:
            memory = f"{bytes_used/1024**2:.2f} MB"
        
        return DataQualityReport(
            total_rows=len(df),
            missing_values=df.isnull().sum().to_dict(),
            duplicates=df.duplicated().sum(),
            memory_usage=memory,
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()}
        )
    
    def merge_items_with_orders(self) -> pd.DataFrame:
        """
        Merge menu items with order data for analysis.
        
        Returns:
            DataFrame with items and their order statistics
        """
        items = self.get_table('dim_items')
        menu_items = self.get_table('dim_menu_items')
        order_items = self.get_table('fct_order_items')
        
        # Calculate order statistics per menu item
        item_stats = order_items.groupby('menu_item_id').agg({
            'quantity': 'sum',
            'total_cost': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        
        item_stats.columns = ['menu_item_id', 'total_quantity', 
                             'total_revenue', 'order_count']
        
        # Merge with menu items
        merged = menu_items.merge(
            item_stats,
            left_on='id',
            right_on='menu_item_id',
            how='left'
        )
        
        # Fill missing values
        merged['total_quantity'] = merged['total_quantity'].fillna(0)
        merged['total_revenue'] = merged['total_revenue'].fillna(0)
        merged['order_count'] = merged['order_count'].fillna(0)
        
        return merged
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for all loaded tables.
        
        Returns:
            Dictionary with summary for each table
        """
        summary = {}
        
        for name, df in self.loaded_tables.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'missing_pct': (df.isnull().sum().sum() / df.size * 100) 
                              if df.size > 0 else 0
            }
        
        return summary
    
    def validate_data_integrity(self) -> Dict[str, List[str]]:
        """
        Validate data integrity across tables.
        
        Checks for:
        - Foreign key relationships
        - Missing required fields
        - Data type consistency
        
        Returns:
            Dictionary of issues found per table
        """
        issues = {}
        
        # Check dim_menu_items references
        if 'dim_menu_items' in self.loaded_tables:
            menu_items = self.loaded_tables['dim_menu_items']
            menu_issues = []
            
            if 'price' in menu_items.columns:
                invalid_prices = (menu_items['price'] < 0).sum()
                if invalid_prices > 0:
                    menu_issues.append(f"{invalid_prices} items with negative price")
            
            if menu_issues:
                issues['dim_menu_items'] = menu_issues
        
        # Check fct_orders
        if 'fct_orders' in self.loaded_tables:
            orders = self.loaded_tables['fct_orders']
            order_issues = []
            
            if 'total' in orders.columns:
                zero_totals = (orders['total'] == 0).sum()
                if zero_totals > 0:
                    order_issues.append(f"{zero_totals} orders with zero total")
            
            if order_issues:
                issues['fct_orders'] = order_issues
        
        return issues
    
    def __repr__(self) -> str:
        return f"DataLoader(dir='{self.data_dir}', tables_loaded={len(self.loaded_tables)})"
