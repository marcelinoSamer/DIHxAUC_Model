"""
File: inventory_visualizations.py
Description: Visualization dashboards for inventory optimization.
Dependencies: matplotlib, seaborn, pandas
Author: FlavorFlow Team

Generates publication-ready visualizations:
1. Demand forecast time series
2. Inventory health heatmap
3. Customer behavior patterns
4. Stock alert dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime


class InventoryVisualizationService:
    """
    Generate visualizations for inventory optimization analysis.
    
    All charts are designed for executive presentations
    and hackathon deliverables.
    """
    
    # Color palette
    COLORS = {
        'primary': '#2C3E50',
        'secondary': '#3498DB',
        'success': '#27AE60',
        'warning': '#F39C12',
        'danger': '#E74C3C',
        'info': '#9B59B6',
        'light': '#ECF0F1'
    }
    
    STATUS_COLORS = {
        'Critical': '#E74C3C',
        'Low': '#F39C12',
        'Optimal': '#27AE60',
        'High': '#F1C40F',
        'Excess': '#3498DB'
    }
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: str = 'seaborn-v0_8-whitegrid',
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Initialize visualization service.
        
        Args:
            output_dir: Directory to save figures
            style: Matplotlib style
            dpi: Figure resolution
            figsize: Default figure size
        """
        self.output_dir = Path(output_dir) if output_dir else Path('docs')
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        
        plt.style.use(style)
        plt.rcParams['figure.dpi'] = dpi
    
    def plot_temporal_patterns(
        self,
        hourly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot customer ordering patterns by hour and day.
        
        Args:
            hourly_data: DataFrame with hour, num_orders columns
            daily_data: DataFrame with day_name, order counts
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Hourly pattern
        ax1 = axes[0]
        bars = ax1.bar(
            hourly_data['hour'],
            hourly_data['num_orders'],
            color=self.COLORS['secondary'],
            alpha=0.8,
            edgecolor='white'
        )
        
        # Highlight peak hours
        peak_idx = hourly_data['num_orders'].idxmax()
        bars[peak_idx].set_color(self.COLORS['danger'])
        
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Number of Orders', fontsize=12)
        ax1.set_title('⏰ Orders by Hour of Day', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 24, 2))
        ax1.axhline(y=hourly_data['num_orders'].mean(), color='red', 
                   linestyle='--', alpha=0.5, label='Average')
        ax1.legend()
        
        # Daily pattern
        ax2 = axes[1]
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
        daily_sorted = daily_data.set_index('day_name').reindex(day_order)
        
        colors = [self.COLORS['info'] if day in ['Saturday', 'Sunday'] 
                  else self.COLORS['secondary'] for day in day_order]
        
        ax2.bar(
            range(7),
            daily_sorted['order_id'],
            color=colors,
            alpha=0.8,
            edgecolor='white'
        )
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Number of Orders', fontsize=12)
        ax2.set_title('📅 Orders by Day of Week', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'temporal_patterns.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_demand_forecast(
        self,
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        item_name: str = "All Items",
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot demand forecast time series.
        
        Args:
            historical: Historical demand data
            forecast: Forecasted demand data
            item_name: Name of item for title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Historical
        if 'date' in historical.columns:
            ax.plot(
                pd.to_datetime(historical['date']),
                historical['quantity_demanded'],
                color=self.COLORS['secondary'],
                alpha=0.7,
                label='Historical Demand'
            )
        
        # Forecast
        if 'date' in forecast.columns:
            ax.plot(
                pd.to_datetime(forecast['date']),
                forecast['predicted_demand'],
                color=self.COLORS['danger'],
                linestyle='--',
                linewidth=2,
                label='Forecast'
            )
            
            # Confidence interval (simulated)
            ax.fill_between(
                pd.to_datetime(forecast['date']),
                forecast['predicted_demand'] * 0.8,
                forecast['predicted_demand'] * 1.2,
                color=self.COLORS['danger'],
                alpha=0.2,
                label='Confidence Interval'
            )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Demand (Units)', fontsize=12)
        ax.set_title(f'📈 Demand Forecast: {item_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'demand_forecast.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_inventory_status(
        self,
        inventory_analysis: pd.DataFrame,
        top_n: int = 20,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot inventory status overview.
        
        Args:
            inventory_analysis: DataFrame with stock analysis
            top_n: Number of items to show
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Status distribution pie chart
        ax1 = axes[0]
        status_counts = inventory_analysis['status'].apply(
            lambda x: x.split(' - ')[0].replace('🔴 ', '').replace('🟠 ', '')
                      .replace('🟢 ', '').replace('🟡 ', '').replace('🔵 ', '')
        ).value_counts()
        
        colors = [self.STATUS_COLORS.get(s, '#95A5A6') for s in status_counts.index]
        
        wedges, texts, autotexts = ax1.pie(
            status_counts,
            labels=status_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.05 if 'Critical' in s else 0 for s in status_counts.index]
        )
        ax1.set_title('📊 Inventory Status Distribution', fontsize=14, fontweight='bold')
        
        # Days of stock bar chart
        ax2 = axes[1]
        
        # Sort by days of stock and get critical/low items
        critical_items = inventory_analysis[
            inventory_analysis['status'].str.contains('Critical|Low')
        ].nsmallest(top_n, 'days_of_stock')
        
        if not critical_items.empty:
            y_pos = range(len(critical_items))
            
            colors = [self.COLORS['danger'] if 'Critical' in s 
                      else self.COLORS['warning'] for s in critical_items['status']]
            
            ax2.barh(
                y_pos,
                critical_items['days_of_stock'],
                color=colors,
                alpha=0.8
            )
            
            ax2.set_yticks(y_pos)
            labels = critical_items['item_id'].astype(str).tolist()
            if 'item_name' in critical_items.columns:
                labels = critical_items['item_name'].fillna(
                    critical_items['item_id'].astype(str)
                ).tolist()
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.set_xlabel('Days of Stock Remaining', fontsize=12)
            ax2.set_title('⚠️ Items with Low Stock', fontsize=14, fontweight='bold')
            ax2.axvline(x=7, color='red', linestyle='--', alpha=0.5, label='1 Week')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No Critical Items', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'inventory_status.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot demand model feature importance.
        
        Args:
            importance_df: DataFrame with feature, importance columns
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort and plot
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(importance_df)))
        
        ax.barh(
            importance_df['feature'],
            importance_df['importance'],
            color=colors,
            edgecolor='white'
        )
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('🎯 Demand Prediction Feature Importance', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (_, row) in enumerate(importance_df.iterrows()):
            ax.text(row['importance'] + 0.01, i, f"{row['importance']:.3f}",
                   va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'feature_importance.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_model_performance(
        self,
        metrics: Dict,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot model performance metrics.
        
        Args:
            metrics: Dictionary with MAE, RMSE, R2
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        metric_names = ['MAE', 'RMSE', 'R²']
        metric_values = [
            metrics.get('mae', 0),
            metrics.get('rmse', 0),
            metrics.get('r2', 0)
        ]
        colors = [self.COLORS['secondary'], self.COLORS['warning'], self.COLORS['success']]
        
        for ax, name, value, color in zip(axes, metric_names, metric_values, colors):
            # Create a gauge-like visualization
            ax.barh([0], [value if name != 'R²' else value], color=color, height=0.5)
            
            if name == 'R²':
                ax.set_xlim(0, 1)
                ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
            
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.text(value/2 if name != 'R²' else value/2, 0, 
                   f'{value:.3f}', ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='white')
            ax.set_yticks([])
        
        plt.suptitle('🤖 Demand Model Performance', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'model_performance.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_executive_dashboard(
        self,
        results: Dict,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Create a comprehensive executive dashboard.
        
        Args:
            results: Full analysis results dictionary
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(
            '📊 INVENTORY OPTIMIZATION EXECUTIVE DASHBOARD',
            fontsize=18, fontweight='bold', y=0.98
        )
        
        # 1. Temporal patterns (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'behavior' in results and 'temporal' in results['behavior']:
            hourly = pd.DataFrame(results['behavior']['temporal']['hourly'])
            ax1.bar(hourly['hour'], hourly['num_orders'], 
                   color=self.COLORS['secondary'], alpha=0.8)
            ax1.set_title('⏰ Orders by Hour', fontweight='bold')
            ax1.set_xlabel('Hour')
            ax1.set_ylabel('Orders')
        
        # 2. Model metrics (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'demand_model' in results and 'metrics' in results['demand_model']:
            metrics = results['demand_model']['metrics']
            metric_names = ['MAE', 'RMSE', 'R²']
            metric_values = [metrics.get('mae', 0), metrics.get('rmse', 0), metrics.get('r2', 0)]
            colors = [self.COLORS['secondary'], self.COLORS['warning'], self.COLORS['success']]
            ax2.bar(metric_names, metric_values, color=colors, alpha=0.8)
            ax2.set_title('🤖 Model Performance', fontweight='bold')
            for i, v in enumerate(metric_values):
                ax2.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
        
        # 3. Key stats (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        if 'behavior' in results:
            summary = results['behavior'].get('summary', {})
            temporal = summary.get('temporal_insights', {})
            purchase = summary.get('purchase_insights', {})
            
            stats_text = f"""
📈 KEY METRICS

Peak Hour: {temporal.get('peak_hour_label', 'N/A')}
Peak Day: {temporal.get('peak_day', 'N/A')}
Weekend %: {temporal.get('weekend_pct', 0):.1f}%

Avg Order Value: {purchase.get('avg_order_value', 0):.2f}
Avg Items/Order: {purchase.get('avg_items_per_order', 0):.1f}
"""
            ax3.text(0.1, 0.5, stats_text, fontsize=12, va='center',
                    family='monospace', transform=ax3.transAxes)
            ax3.set_title('📋 Summary', fontweight='bold')
        
        # 4. Inventory status (middle, spanning 2 cols)
        ax4 = fig.add_subplot(gs[1, :2])
        if 'inventory' in results and 'analysis' in results['inventory']:
            analysis = results['inventory']['analysis']
            status_counts = analysis['status'].apply(
                lambda x: x.split(' - ')[0].replace('🔴 ', '').replace('🟠 ', '')
                          .replace('🟢 ', '').replace('🟡 ', '').replace('🔵 ', '')
            ).value_counts()
            colors = [self.STATUS_COLORS.get(s, '#95A5A6') for s in status_counts.index]
            ax4.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax4.set_title('📦 Inventory Status', fontweight='bold')
        
        # 5. Feature importance (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'demand_model' in results and 'feature_importance' in results['demand_model']:
            importance = pd.DataFrame(results['demand_model']['feature_importance'])
            importance = importance.nlargest(5, 'importance')
            ax5.barh(importance['feature'], importance['importance'],
                    color=self.COLORS['info'], alpha=0.8)
            ax5.set_title('🎯 Top Features', fontweight='bold')
        
        # 6. Alerts summary (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        if 'inventory' in results and 'alerts' in results['inventory']:
            alerts = results['inventory']['alerts']
            if not alerts.empty:
                critical = len(alerts[alerts['status'].str.contains('Critical')])
                low = len(alerts[alerts['status'].str.contains('Low')])
                excess = len(alerts[alerts['status'].str.contains('Excess')])
                
                alert_text = f"""
{'='*80}
🚨 INVENTORY ALERTS SUMMARY

🔴 CRITICAL (Stockout Risk): {critical} items - IMMEDIATE ACTION REQUIRED
🟠 LOW STOCK (Reorder Soon): {low} items - Place orders this week
🔵 EXCESS (Overstock Risk): {excess} items - Monitor and reduce orders

💡 RECOMMENDATION: Focus on critical items first. Review reorder points weekly.
{'='*80}
"""
                ax6.text(0.5, 0.5, alert_text, fontsize=11, va='center', ha='center',
                        family='monospace', transform=ax6.transAxes)
        
        if save:
            path = self.output_dir / 'executive_dashboard.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def close_all(self):
        """Close all open figures."""
        plt.close('all')
