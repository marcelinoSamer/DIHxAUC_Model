#!/usr/bin/env python3
"""
Main runner script for Menu Intelligence Analysis.

This script orchestrates the full analysis pipeline:
1. Load and validate data
2. Assess data quality
3. Perform menu engineering (BCG Matrix) analysis
4. Analyze pricing distribution
5. Analyze restaurant performance
6. Analyze campaign effectiveness
7. Train demand prediction models
8. Perform item clustering
9. Generate executive summary
10. Export results

Usage:
    python -m src.main

Or run individual analyses:
    from src.main import run_menu_engineering, run_pricing_analysis
    results = run_menu_engineering()
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# Configure pandas display
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# Import project modules
from src import config
from src.data_loader import (
    load_all_datasets, 
    prepare_menu_analysis_data,
    prepare_place_analysis_data,
    aggregate_item_performance,
    print_dataset_summary
)
from src.data_quality import assess_data_quality, generate_quality_report
from src.menu_engineering import (
    perform_bcg_analysis,
    print_bcg_results,
    print_top_performers,
    generate_recommendations
)
from src.pricing_analysis import analyze_price_distribution, print_price_analysis
from src.restaurant_analysis import (
    analyze_restaurant_performance,
    print_restaurant_analysis
)
from src.campaign_analysis import analyze_campaigns, print_campaign_analysis
from src.ml_models import (
    prepare_demand_features,
    train_demand_models,
    print_model_results,
    get_feature_importance,
    print_feature_importance,
    perform_item_clustering,
    analyze_clusters,
    print_cluster_analysis
)
from src.visualizations import (
    setup_plotting_style,
    plot_bcg_matrix,
    plot_price_distribution,
    plot_campaign_analysis,
    plot_clustering_results
)
from src.export import export_all_results, print_export_summary
from src.summary import print_executive_summary


def run_full_analysis(show_plots: bool = True, save_outputs: bool = True):
    """
    Run the complete menu intelligence analysis pipeline.
    
    Args:
        show_plots: Whether to display plots interactively
        save_outputs: Whether to save results to files
        
    Returns:
        Dictionary containing all analysis results
    """
    setup_plotting_style()
    results = {}
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("📥 LOADING DATA")
    print("=" * 60)
    
    datasets = load_all_datasets()
    print_dataset_summary(datasets)
    results['datasets'] = datasets
    
    # =========================================================================
    # STEP 2: Data Quality Assessment
    # =========================================================================
    print("\n" + "=" * 60)
    print("🔍 DATA QUALITY ASSESSMENT")
    print("=" * 60)
    
    for name in ['dim_items', 'dim_menu_items', 'most_ordered']:
        if name in datasets and len(datasets[name]) > 0:
            assess_data_quality(datasets[name], name)
    
    # =========================================================================
    # STEP 3: Menu Engineering Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("🍽️ MENU ENGINEERING ANALYSIS")
    print("=" * 60)
    
    menu_analysis = prepare_menu_analysis_data(
        datasets['most_ordered'], 
        datasets['dim_items']
    )
    item_performance = aggregate_item_performance(menu_analysis)
    item_performance, bcg_metrics = perform_bcg_analysis(item_performance)
    
    print_bcg_results(item_performance, bcg_metrics)
    print_top_performers(item_performance, n=config.TOP_N_ITEMS)
    
    results['item_performance'] = item_performance
    results['bcg_metrics'] = bcg_metrics
    
    # BCG Matrix Visualization
    if save_outputs:
        plot_bcg_matrix(
            item_performance,
            bcg_metrics['popularity_threshold'],
            bcg_metrics['price_threshold'],
            save_path=config.OUTPUT_DIR / 'menu_engineering_matrix.png',
            show=show_plots
        )
    
    # =========================================================================
    # STEP 4: Pricing Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("💰 PRICING ANALYSIS")
    print("=" * 60)
    
    price_analysis = analyze_price_distribution(datasets['dim_items'])
    print_price_analysis(price_analysis)
    results['price_analysis'] = price_analysis
    
    if save_outputs:
        plot_price_distribution(
            price_analysis['data'],
            save_path=config.OUTPUT_DIR / 'pricing_analysis.png',
            show=show_plots
        )
    
    # =========================================================================
    # STEP 5: Restaurant Performance
    # =========================================================================
    print("\n" + "=" * 60)
    print("📍 RESTAURANT ANALYSIS")
    print("=" * 60)
    
    place_analysis = prepare_place_analysis_data(
        datasets['most_ordered'],
        datasets['dim_places']
    )
    restaurant_results = analyze_restaurant_performance(place_analysis)
    print_restaurant_analysis(restaurant_results)
    
    results['place_analysis'] = place_analysis
    results['restaurant_results'] = restaurant_results
    
    # =========================================================================
    # STEP 6: Campaign Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("🎯 CAMPAIGN ANALYSIS")
    print("=" * 60)
    
    campaign_results = analyze_campaigns(datasets['fct_campaigns'])
    print_campaign_analysis(campaign_results)
    results['campaign_results'] = campaign_results
    
    if save_outputs:
        plot_campaign_analysis(
            campaign_results['discount_campaigns'],
            campaign_results['campaign_types'],
            save_path=config.OUTPUT_DIR / 'campaign_analysis.png',
            show=show_plots
        )
    
    # =========================================================================
    # STEP 7: Demand Prediction Models
    # =========================================================================
    print("\n" + "=" * 60)
    print("🤖 MACHINE LEARNING MODELS")
    print("=" * 60)
    
    try:
        X, y = prepare_demand_features(datasets['dim_menu_items'])
        print(f"\nTraining samples: {len(X):,}")
        
        model_results = train_demand_models(X, y)
        print_model_results(model_results)
        
        feature_importance = get_feature_importance(model_results)
        print_feature_importance(feature_importance)
        
        results['model_results'] = model_results
        results['feature_importance'] = feature_importance
    except Exception as e:
        print(f"⚠️ Could not train models: {e}")
    
    # =========================================================================
    # STEP 8: Item Clustering
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 ITEM CLUSTERING")
    print("=" * 60)
    
    item_performance_clustered, cluster_info = perform_item_clustering(item_performance)
    cluster_summary = analyze_clusters(item_performance_clustered)
    print_cluster_analysis(cluster_summary, cluster_info['n_clusters'])
    
    results['item_performance_clustered'] = item_performance_clustered
    results['cluster_info'] = cluster_info
    
    if save_outputs:
        plot_clustering_results(
            cluster_info['X_scaled'],
            cluster_info['cluster_labels'],
            cluster_info['inertias'],
            cluster_info['K_range'],
            cluster_info['n_clusters'],
            save_path=config.OUTPUT_DIR / 'clustering_analysis.png',
            show=show_plots
        )
    
    # =========================================================================
    # STEP 9: Executive Summary
    # =========================================================================
    print_executive_summary(
        num_items=len(datasets['dim_items']),
        num_places=len(datasets['dim_places']),
        total_orders=datasets['most_ordered']['order_count'].sum(),
        num_campaigns=len(datasets['fct_campaigns']),
        avg_price=datasets['dim_items']['price'].mean() if 'price' in datasets['dim_items'].columns else 0,
        med_price=datasets['dim_items']['price'].median() if 'price' in datasets['dim_items'].columns else 0
    )
    
    # =========================================================================
    # STEP 10: Export Results
    # =========================================================================
    if save_outputs:
        export_paths = export_all_results(
            item_performance,
            item_performance_clustered,
            place_analysis,
            config.OUTPUT_DIR
        )
        print_export_summary(export_paths)
        results['export_paths'] = export_paths
    
    return results


# =============================================================================
# INDIVIDUAL ANALYSIS RUNNERS
# =============================================================================

def run_menu_engineering():
    """Run only menu engineering analysis."""
    datasets = load_all_datasets()
    menu_analysis = prepare_menu_analysis_data(
        datasets['most_ordered'], 
        datasets['dim_items']
    )
    item_performance = aggregate_item_performance(menu_analysis)
    item_performance, metrics = perform_bcg_analysis(item_performance)
    print_bcg_results(item_performance, metrics)
    print_top_performers(item_performance)
    return item_performance, metrics


def run_pricing_analysis():
    """Run only pricing analysis."""
    datasets = load_all_datasets()
    analysis = analyze_price_distribution(datasets['dim_items'])
    print_price_analysis(analysis)
    return analysis


def run_campaign_analysis():
    """Run only campaign analysis."""
    datasets = load_all_datasets()
    results = analyze_campaigns(datasets['fct_campaigns'])
    print_campaign_analysis(results)
    return results


def run_ml_models():
    """Run only ML model training."""
    datasets = load_all_datasets()
    X, y = prepare_demand_features(datasets['dim_menu_items'])
    model_results = train_demand_models(X, y)
    print_model_results(model_results)
    return model_results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_full_analysis(show_plots=True, save_outputs=True)
