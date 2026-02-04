"""
Visualization utilities for menu intelligence analysis.

This module provides consistent, publication-ready visualizations for:
- BCG Matrix scatter plots
- Price distributions
- Campaign effectiveness
- Clustering results
- Restaurant performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List

from . import config


# =============================================================================
# SETUP
# =============================================================================

def setup_plotting_style():
    """Configure matplotlib with project defaults."""
    plt.style.use(config.PLOT_STYLE)
    plt.rcParams['figure.dpi'] = config.DPI
    plt.rcParams['savefig.dpi'] = config.DPI
    plt.rcParams['font.size'] = 10


# =============================================================================
# BCG MATRIX VISUALIZATIONS
# =============================================================================

def plot_bcg_matrix(
    item_performance: pd.DataFrame,
    popularity_threshold: float,
    price_threshold: float,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create BCG Matrix scatter plot with category distribution pie chart.
    
    Args:
        item_performance: DataFrame with 'category', 'order_count', 'price', 'revenue'
        popularity_threshold: Vertical threshold line position
        price_threshold: Horizontal threshold line position
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_SIZE_DOUBLE)
    
    # Plot 1: BCG Matrix Scatter
    ax1 = axes[0]
    for category, color in config.BCG_COLORS.items():
        mask = item_performance['category'] == category
        if mask.any():
            # Size points by revenue
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
    
    # Add threshold lines
    ax1.axhline(y=price_threshold, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=popularity_threshold, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and formatting
    ax1.set_xlabel('Order Count (Popularity)', fontsize=12)
    ax1.set_ylabel('Price (Profitability)', fontsize=12)
    ax1.set_title('🎯 Menu Engineering BCG Matrix', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xscale('log')
    
    # Plot 2: Category Distribution Pie
    ax2 = axes[1]
    cat_counts = item_performance['category'].value_counts()
    colors_ordered = [config.BCG_COLORS[c] for c in cat_counts.index]
    ax2.pie(
        cat_counts, 
        labels=cat_counts.index, 
        autopct='%1.1f%%',
        colors=colors_ordered, 
        startangle=90
    )
    ax2.set_title('📊 Menu Item Distribution by Category', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# PRICING VISUALIZATIONS
# =============================================================================

def plot_price_distribution(
    items_with_price: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create price distribution histogram and price vs. purchases scatter plot.
    
    Args:
        items_with_price: DataFrame with 'price' and optionally 'purchases' columns
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_SIZE_DOUBLE)
    
    # Price statistics
    price_stats = items_with_price['price'].describe()
    
    # Plot 1: Price histogram
    axes[0].hist(
        items_with_price['price'], 
        bins=50, 
        color=config.COLORS['primary'], 
        edgecolor='white', 
        alpha=0.7
    )
    axes[0].axvline(
        price_stats['50%'], 
        color='red', 
        linestyle='--', 
        label=f"Median: {price_stats['50%']:.0f}"
    )
    axes[0].axvline(
        price_stats['mean'], 
        color='orange', 
        linestyle='--', 
        label=f"Mean: {price_stats['mean']:.0f}"
    )
    axes[0].set_xlabel('Price')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('💰 Price Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim(0, config.MAX_PRICE_DISPLAY)
    
    # Plot 2: Price vs Orders scatter
    if 'purchases' in items_with_price.columns:
        mask = items_with_price['purchases'] > 0
        axes[1].scatter(
            items_with_price.loc[mask, 'price'],
            items_with_price.loc[mask, 'purchases'],
            alpha=0.3, 
            c=config.COLORS['success']
        )
        axes[1].set_xlabel('Price')
        axes[1].set_ylabel('Total Purchases')
        axes[1].set_title('📈 Price vs. Purchase Volume', fontweight='bold')
        axes[1].set_xlim(0, config.MAX_PRICE_DISPLAY)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# CAMPAIGN VISUALIZATIONS
# =============================================================================

def plot_campaign_analysis(
    discount_campaigns: pd.DataFrame,
    campaign_types: pd.Series,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create campaign effectiveness visualizations.
    
    Args:
        discount_campaigns: DataFrame with 'discount' column
        campaign_types: Series with campaign type counts
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_SIZE_DOUBLE)
    
    # Plot 1: Discount distribution
    axes[0].hist(
        discount_campaigns['discount'], 
        bins=20, 
        color=config.COLORS['secondary'], 
        edgecolor='white', 
        alpha=0.7
    )
    axes[0].set_xlabel('Discount Percentage')
    axes[0].set_ylabel('Number of Campaigns')
    axes[0].set_title('📊 Discount Distribution', fontweight='bold')
    
    # Plot 2: Campaign types pie
    axes[1].pie(
        campaign_types, 
        labels=campaign_types.index, 
        autopct='%1.1f%%', 
        startangle=90
    )
    axes[1].set_title('🎯 Campaign Type Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# CLUSTERING VISUALIZATIONS
# =============================================================================

def plot_clustering_results(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    inertias: List[float],
    K_range: range,
    optimal_k: int,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create clustering analysis visualizations (elbow plot and PCA scatter).
    
    Args:
        X_scaled: Scaled feature matrix
        cluster_labels: Cluster assignments
        inertias: List of inertia values for elbow plot
        K_range: Range of K values tested
        optimal_k: Selected number of clusters
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.decomposition import PCA
    
    setup_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_SIZE_DOUBLE)
    
    # Plot 1: Elbow plot
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('📈 Elbow Method for Optimal K', fontweight='bold')
    axes[0].legend()
    
    # Plot 2: Cluster visualization (PCA reduced)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    scatter = axes[1].scatter(
        X_pca[:, 0], 
        X_pca[:, 1], 
        c=cluster_labels, 
        cmap='viridis', 
        alpha=0.6
    )
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    axes[1].set_title('🎯 Item Clusters (PCA Visualization)', fontweight='bold')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create horizontal bar chart of feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_SINGLE)
    
    # Sort by importance
    indices = np.argsort(importances)
    
    ax.barh(range(len(indices)), importances[indices], color=config.COLORS['primary'])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'📊 {title}', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_all_figures(figures: dict, output_dir: Path) -> None:
    """
    Save multiple figures to the output directory.
    
    Args:
        figures: Dictionary mapping filenames to Figure objects
        output_dir: Directory to save figures
    """
    output_dir.mkdir(exist_ok=True)
    
    for filename, fig in figures.items():
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"✅ Saved: {filepath}")


def close_all_figures() -> None:
    """Close all open matplotlib figures."""
    plt.close('all')
