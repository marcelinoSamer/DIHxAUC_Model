"""
File: inventory_optimizer.py
Description: Inventory optimization based on demand forecasts.
Dependencies: pandas, numpy
Author: FlavorFlow Team

This module calculates optimal inventory levels including:
- Safety stock
- Reorder points
- Stock alerts (over/under)
- Ingredient requirements via Bill of Materials
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class StockStatus(Enum):
    """Stock status classification."""
    CRITICAL = "🔴 Critical - Stockout Risk"
    LOW = "🟠 Low - Reorder Soon"
    OPTIMAL = "🟢 Optimal"
    HIGH = "🟡 High - Monitor"
    EXCESS = "🔵 Excess - Overstock Risk"


@dataclass
class InventoryMetrics:
    """Metrics for a single item's inventory."""
    item_id: int
    item_name: Optional[str]
    current_stock: float
    avg_daily_demand: float
    demand_std: float
    safety_stock: float
    reorder_point: float
    days_of_stock: float
    status: StockStatus
    recommended_order_qty: float


class InventoryOptimizer:
    """
    Inventory optimization engine.
    
    Uses demand forecasts to calculate optimal inventory levels
    and generate stock alerts to prevent over/understocking.
    
    Key calculations:
    - Safety Stock = Z * σ * √(Lead Time)
    - Reorder Point = (Avg Daily Demand * Lead Time) + Safety Stock
    - Economic Order Quantity = √(2DS/H)
    
    Example:
        >>> optimizer = InventoryOptimizer(lead_time_days=3)
        >>> optimizer.load_demand_data(demand_df)
        >>> alerts = optimizer.get_stock_alerts(current_inventory_df)
    """
    
    def __init__(
        self,
        lead_time_days: float = 2.0,
        service_level: float = 0.95,
        holding_cost_pct: float = 0.20,
        order_cost: float = 50.0
    ):
        """
        Initialize the inventory optimizer.
        
        Args:
            lead_time_days: Average lead time for restocking (days)
            service_level: Target service level (0-1), e.g., 0.95 = 95%
            holding_cost_pct: Annual holding cost as % of item value
            order_cost: Fixed cost per order placement
        """
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        self.holding_cost_pct = holding_cost_pct
        self.order_cost = order_cost
        
        # Z-score for service level (95% = 1.65, 99% = 2.33)
        from scipy import stats
        self.z_score = stats.norm.ppf(service_level)
        
        self.demand_stats: Optional[pd.DataFrame] = None
        self.inventory_metrics: Dict[int, InventoryMetrics] = {}
    
    def load_demand_data(
        self,
        daily_demand: pd.DataFrame,
        items_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Load demand data and calculate demand statistics.
        
        Args:
            daily_demand: DataFrame with daily demand per item
            items_df: Optional items master data for names/prices
            
        Returns:
            DataFrame with demand statistics per item
        """
        # Calculate demand statistics per item
        self.demand_stats = daily_demand.groupby('item_id').agg({
            'quantity_demanded': ['mean', 'std', 'sum', 'count'],
            'avg_price': 'mean'
        }).reset_index()
        
        self.demand_stats.columns = [
            'item_id', 'avg_daily_demand', 'demand_std', 
            'total_demand', 'days_with_demand', 'avg_price'
        ]
        
        # Fill NaN std with small value
        self.demand_stats['demand_std'] = self.demand_stats['demand_std'].fillna(
            self.demand_stats['avg_daily_demand'] * 0.2
        )
        
        # Calculate safety stock and reorder point
        self.demand_stats['safety_stock'] = (
            self.z_score * 
            self.demand_stats['demand_std'] * 
            np.sqrt(self.lead_time_days)
        ).round(0)
        
        self.demand_stats['reorder_point'] = (
            (self.demand_stats['avg_daily_demand'] * self.lead_time_days) +
            self.demand_stats['safety_stock']
        ).round(0)
        
        # Calculate Economic Order Quantity (EOQ)
        annual_demand = self.demand_stats['avg_daily_demand'] * 365
        holding_cost = self.demand_stats['avg_price'] * self.holding_cost_pct
        
        self.demand_stats['eoq'] = np.sqrt(
            (2 * annual_demand * self.order_cost) / 
            holding_cost.replace(0, 1)
        ).round(0)
        
        # Add item names if provided
        if items_df is not None and 'title' in items_df.columns:
            self.demand_stats = self.demand_stats.merge(
                items_df[['id', 'title']].rename(columns={'id': 'item_id', 'title': 'item_name'}),
                on='item_id',
                how='left'
            )
        
        return self.demand_stats
    
    def calculate_stock_status(
        self,
        current_stock: float,
        avg_daily_demand: float,
        safety_stock: float,
        reorder_point: float
    ) -> Tuple[StockStatus, float]:
        """
        Determine stock status and days of stock remaining.
        
        Args:
            current_stock: Current inventory level
            avg_daily_demand: Average daily demand
            safety_stock: Calculated safety stock
            reorder_point: Reorder point threshold
            
        Returns:
            Tuple of (StockStatus, days_of_stock)
        """
        if avg_daily_demand <= 0:
            days_of_stock = float('inf')
        else:
            days_of_stock = current_stock / avg_daily_demand
        
        # Determine status
        if current_stock <= 0:
            status = StockStatus.CRITICAL
        elif current_stock < safety_stock:
            status = StockStatus.CRITICAL
        elif current_stock < reorder_point:
            status = StockStatus.LOW
        elif days_of_stock > 30:  # More than 30 days of stock
            status = StockStatus.EXCESS
        elif days_of_stock > 14:  # More than 2 weeks
            status = StockStatus.HIGH
        else:
            status = StockStatus.OPTIMAL
        
        return status, days_of_stock
    
    def analyze_inventory(
        self,
        current_inventory: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze current inventory against demand forecasts.
        
        Args:
            current_inventory: DataFrame with 'item_id' and 'current_stock' columns
            
        Returns:
            DataFrame with full inventory analysis
        """
        if self.demand_stats is None:
            raise ValueError("Demand data not loaded. Call load_demand_data first.")
        
        # Merge current inventory with demand stats
        analysis = self.demand_stats.merge(
            current_inventory[['item_id', 'current_stock']],
            on='item_id',
            how='left'
        )
        analysis['current_stock'] = analysis['current_stock'].fillna(0)
        
        # Calculate status for each item
        statuses = []
        days_of_stock = []
        recommended_orders = []
        
        for _, row in analysis.iterrows():
            status, dos = self.calculate_stock_status(
                row['current_stock'],
                row['avg_daily_demand'],
                row['safety_stock'],
                row['reorder_point']
            )
            statuses.append(status.value)
            days_of_stock.append(round(dos, 1) if dos != float('inf') else 999)
            
            # Calculate recommended order quantity
            if row['current_stock'] < row['reorder_point']:
                # Order up to EOQ + safety stock
                recommended = max(0, row['eoq'] - row['current_stock'] + row['safety_stock'])
            else:
                recommended = 0
            recommended_orders.append(round(recommended, 0))
        
        analysis['status'] = statuses
        analysis['days_of_stock'] = days_of_stock
        analysis['recommended_order_qty'] = recommended_orders
        
        return analysis
    
    def get_stock_alerts(
        self,
        current_inventory: pd.DataFrame,
        alert_levels: List[StockStatus] = None
    ) -> pd.DataFrame:
        """
        Get items that need attention (stockout risk or overstock).
        
        Args:
            current_inventory: Current inventory levels
            alert_levels: Which status levels to include in alerts
            
        Returns:
            DataFrame with items needing attention
        """
        if alert_levels is None:
            alert_levels = [StockStatus.CRITICAL, StockStatus.LOW, StockStatus.EXCESS]
        
        analysis = self.analyze_inventory(current_inventory)
        
        alert_values = [s.value for s in alert_levels]
        alerts = analysis[analysis['status'].isin(alert_values)].copy()
        
        # Sort by urgency (critical first)
        status_order = {
            StockStatus.CRITICAL.value: 0,
            StockStatus.LOW.value: 1,
            StockStatus.EXCESS.value: 2,
            StockStatus.HIGH.value: 3,
            StockStatus.OPTIMAL.value: 4
        }
        alerts['sort_order'] = alerts['status'].map(status_order)
        alerts = alerts.sort_values('sort_order')
        
        return alerts.drop(columns=['sort_order'])
    
    def calculate_ingredient_requirements(
        self,
        demand_forecast: pd.DataFrame,
        bill_of_materials: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate ingredient requirements based on demand forecast and BOM.
        
        Args:
            demand_forecast: Forecasted demand per item
            bill_of_materials: Bill of materials (parent_sku_id, sku_id, quantity)
            
        Returns:
            DataFrame with ingredient requirements
        """
        # Merge forecast with BOM
        requirements = demand_forecast.merge(
            bill_of_materials,
            left_on='item_id',
            right_on='parent_sku_id',
            how='inner'
        )
        
        # Calculate required ingredient quantity
        requirements['ingredient_qty_needed'] = (
            requirements['predicted_demand'] * requirements['quantity']
        )
        
        # Aggregate by ingredient (sku_id)
        ingredient_needs = requirements.groupby('sku_id').agg({
            'ingredient_qty_needed': 'sum',
            'parent_sku_id': 'nunique'
        }).reset_index()
        
        ingredient_needs.columns = [
            'ingredient_id', 'total_qty_needed', 'used_in_n_items'
        ]
        
        return ingredient_needs.sort_values('total_qty_needed', ascending=False)
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of inventory optimization metrics."""
        if self.demand_stats is None:
            return {'error': 'No data loaded'}
        
        return {
            'total_items_analyzed': len(self.demand_stats),
            'parameters': {
                'lead_time_days': self.lead_time_days,
                'service_level': f"{self.service_level * 100:.0f}%",
                'z_score': round(self.z_score, 2)
            },
            'demand_overview': {
                'avg_daily_demand_all_items': round(
                    self.demand_stats['avg_daily_demand'].sum(), 0
                ),
                'total_safety_stock_needed': round(
                    self.demand_stats['safety_stock'].sum(), 0
                ),
                'avg_reorder_point': round(
                    self.demand_stats['reorder_point'].mean(), 1
                )
            }
        }


class StockAlertGenerator:
    """Generate actionable stock alerts and reports."""
    
    @staticmethod
    def generate_alert_report(alerts_df: pd.DataFrame) -> str:
        """Generate a formatted alert report."""
        if alerts_df.empty:
            return "✅ No stock alerts - all items at optimal levels!"
        
        lines = [
            "=" * 60,
            "🚨 INVENTORY ALERT REPORT",
            "=" * 60,
            ""
        ]
        
        # Group by status
        critical = alerts_df[alerts_df['status'] == StockStatus.CRITICAL.value]
        low = alerts_df[alerts_df['status'] == StockStatus.LOW.value]
        excess = alerts_df[alerts_df['status'] == StockStatus.EXCESS.value]
        
        if not critical.empty:
            lines.append("🔴 CRITICAL - Immediate Action Required:")
            lines.append("-" * 40)
            for _, row in critical.head(10).iterrows():
                name = row.get('item_name', f"Item {row['item_id']}")
                lines.append(
                    f"  • {name}: {row['current_stock']:.0f} units "
                    f"(need {row['recommended_order_qty']:.0f})"
                )
            lines.append("")
        
        if not low.empty:
            lines.append("🟠 LOW STOCK - Reorder Soon:")
            lines.append("-" * 40)
            for _, row in low.head(10).iterrows():
                name = row.get('item_name', f"Item {row['item_id']}")
                lines.append(
                    f"  • {name}: {row['days_of_stock']:.1f} days remaining"
                )
            lines.append("")
        
        if not excess.empty:
            lines.append("🔵 EXCESS STOCK - Overstock Risk:")
            lines.append("-" * 40)
            for _, row in excess.head(10).iterrows():
                name = row.get('item_name', f"Item {row['item_id']}")
                lines.append(
                    f"  • {name}: {row['days_of_stock']:.0f} days of stock"
                )
            lines.append("")
        
        # Summary
        lines.append("=" * 60)
        lines.append("SUMMARY:")
        lines.append(f"  Critical items: {len(critical)}")
        lines.append(f"  Low stock items: {len(low)}")
        lines.append(f"  Excess stock items: {len(excess)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
