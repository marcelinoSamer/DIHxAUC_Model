"""
File: reporting_service.py
Description: Service for generating reports and visualizations.
Dependencies: pandas, matplotlib, seaborn
Author: FlavorFlow Team

This service handles all report generation including
visualizations, executive summaries, and data exports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class ReportingService:
    """
    Service for generating business reports and visualizations.
    
    Generates publication-ready charts, executive summaries,
    and formatted reports for stakeholder presentations.
    
    Attributes:
        output_dir: Directory for saving reports
        style: Matplotlib style to use
        dpi: Resolution for saved figures
    
    Example:
        >>> reporter = ReportingService(output_dir='docs/')
        >>> reporter.generate_bcg_report(classified_data, metrics)
        >>> reporter.generate_executive_presentation(all_results)
    """
    
    # Color palette for consistency
    COLORS = {
        'star': '#2ecc71',
        'plowhorse': '#3498db',
        'puzzle': '#f39c12',
        'dog': '#e74c3c',
        'primary': '#3498db',
        'secondary': '#9b59b6',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c'
    }
    
    BCG_COLORS = {
        '⭐ Star': '#2ecc71',
        '🐴 Plowhorse': '#3498db',
        '❓ Puzzle': '#f39c12',
        '🐕 Dog': '#e74c3c'
    }
    
    def __init__(
        self, 
        output_dir: Optional[Path] = None,
        style: str = 'seaborn-v0_8-whitegrid',
        dpi: int = 150
    ):
        """
        Initialize the reporting service.
        
        Args:
            output_dir: Directory for output files
            style: Matplotlib style
            dpi: Figure resolution
        """
        self.output_dir = Path(output_dir) if output_dir else Path('docs')
        self.output_dir.mkdir(exist_ok=True)
        self.style = style
        self.dpi = dpi
        
        plt.style.use(style)
    
    def generate_bcg_matrix_plot(
        self,
        item_performance: pd.DataFrame,
        popularity_threshold: float,
        price_threshold: float,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Generate BCG Matrix scatter plot.
        
        Args:
            item_performance: DataFrame with classified items
            popularity_threshold: Horizontal threshold
            price_threshold: Vertical threshold
            save: Whether to save the figure
            show: Whether to display the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1 = axes[0]
        for category, color in self.BCG_COLORS.items():
            mask = item_performance['category'] == category
            if mask.any():
                sizes = (item_performance.loc[mask, 'revenue'] / 
                        item_performance['revenue'].max() * 500 + 20)
                ax1.scatter(
                    item_performance.loc[mask, 'order_count'],
                    item_performance.loc[mask, 'price'],
                    c=color,
                    label=category,
                    alpha=0.6,
                    s=sizes
                )
        
        ax1.axhline(y=price_threshold, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=popularity_threshold, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Order Count (Popularity)', fontsize=12)
        ax1.set_ylabel('Price (Profitability)', fontsize=12)
        ax1.set_title('🎯 Menu Engineering BCG Matrix', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_xscale('log')
        
        # Pie chart
        ax2 = axes[1]
        cat_counts = item_performance['category'].value_counts()
        colors = [self.BCG_COLORS[c] for c in cat_counts.index]
        ax2.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('📊 Menu Item Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'menu_engineering_matrix.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        
        return fig
    
    def generate_pricing_analysis_plot(
        self,
        items_with_price: pd.DataFrame,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Generate pricing distribution visualization.
        
        Args:
            items_with_price: DataFrame with price data
            save: Whether to save
            show: Whether to display
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        price_stats = items_with_price['price'].describe()
        
        # Histogram
        axes[0].hist(items_with_price['price'], bins=50,
                    color=self.COLORS['primary'], edgecolor='white', alpha=0.7)
        axes[0].axvline(price_stats['50%'], color='red', linestyle='--',
                       label=f"Median: {price_stats['50%']:.0f}")
        axes[0].axvline(price_stats['mean'], color='orange', linestyle='--',
                       label=f"Mean: {price_stats['mean']:.0f}")
        axes[0].set_xlabel('Price')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('💰 Price Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].set_xlim(0, 500)
        
        # Price vs purchases
        if 'purchases' in items_with_price.columns:
            mask = items_with_price['purchases'] > 0
            axes[1].scatter(items_with_price.loc[mask, 'price'],
                          items_with_price.loc[mask, 'purchases'],
                          alpha=0.3, c=self.COLORS['success'])
            axes[1].set_xlabel('Price')
            axes[1].set_ylabel('Total Purchases')
            axes[1].set_title('📈 Price vs. Purchase Volume', fontweight='bold')
            axes[1].set_xlim(0, 500)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'pricing_analysis.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        
        return fig
    
    def generate_cluster_plot(
        self,
        X_pca: pd.DataFrame,
        cluster_labels: List[int],
        inertias: List[float],
        k_range: range,
        optimal_k: int,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Generate clustering visualization.
        
        Args:
            X_pca: PCA-transformed coordinates
            cluster_labels: Cluster assignments
            inertias: Inertia values for elbow plot
            k_range: Range of K values tested
            optimal_k: Selected K
            save: Whether to save
            show: Whether to display
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(list(k_range), inertias, 'bo-')
        axes[0].axvline(x=optimal_k, color='r', linestyle='--',
                       label=f'Optimal K={optimal_k}')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('📈 Elbow Method', fontweight='bold')
        axes[0].legend()
        
        # Cluster scatter
        scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                                 c=cluster_labels, cmap='viridis', alpha=0.6)
        axes[1].set_xlabel('PCA Component 1')
        axes[1].set_ylabel('PCA Component 2')
        axes[1].set_title('🎯 Item Clusters', fontweight='bold')
        plt.colorbar(scatter, ax=axes[1], label='Cluster')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'clustering_analysis.png'
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        if show:
            plt.show()
        
        return fig
    
    def generate_executive_summary_text(self, summary: Dict) -> str:
        """
        Generate formatted executive summary text.
        
        Args:
            summary: Dictionary with summary data
            
        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 80)
        report.append("📋 FLAVORFLOW CRAFT - EXECUTIVE SUMMARY")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 80)
        
        if 'data_overview' in summary:
            overview = summary['data_overview']
            report.append("\n📊 DATA OVERVIEW")
            report.append("-" * 40)
            report.append(f"  Menu Items Analyzed: {overview.get('total_items', 0):,}")
            report.append(f"  Restaurant Locations: {overview.get('total_restaurants', 0):,}")
            report.append(f"  Total Orders Tracked: {overview.get('total_orders', 0):,}")
            report.append(f"  Campaigns Analyzed: {overview.get('total_campaigns', 0):,}")
        
        if 'bcg_breakdown' in summary:
            bcg = summary['bcg_breakdown']
            report.append("\n🎯 BCG MATRIX CLASSIFICATION")
            report.append("-" * 40)
            report.append(f"  ⭐ Stars: {bcg.get('stars', 0)} items")
            report.append(f"  🐴 Plowhorses: {bcg.get('plowhorses', 0)} items")
            report.append(f"  ❓ Puzzles: {bcg.get('puzzles', 0)} items")
            report.append(f"  🐕 Dogs: {bcg.get('dogs', 0)} items")
        
        if 'pricing_opportunity' in summary:
            pricing = summary['pricing_opportunity']
            report.append("\n💰 PRICING OPTIMIZATION OPPORTUNITY")
            report.append("-" * 40)
            report.append(f"  Items to Reprice: {pricing.get('items_to_reprice', 0)}")
            report.append(f"  Potential Revenue Gain: {pricing.get('total_revenue_gain', 0):,.2f} DKK")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_executive_summary(self, summary: Dict) -> Path:
        """
        Save executive summary to file.
        
        Args:
            summary: Summary dictionary
            
        Returns:
            Path to saved file
        """
        text = self.generate_executive_summary_text(summary)
        path = self.output_dir / 'executive_summary.txt'
        path.write_text(text)
        print(f"✅ Saved: {path}")
        return path
    
    def close_all(self) -> None:
        """Close all open figures."""
        plt.close('all')
