#!/usr/bin/env python3
"""
FlavorFlow Craft - Menu Engineering Solution
=============================================

Main entry point for the FlavorFlow Craft application.
This script provides CLI commands for running analysis,
starting the API server, and generating reports.

Usage:
    python -m src.main analyze          # Run full analysis
    python -m src.main serve            # Start API server
    python -m src.main --help           # Show help

Author: FlavorFlow Team
Version: 1.0.0
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

from src.database import engine, Base
from src.models import db_models

# Create database tables
Base.metadata.create_all(bind=engine)


import pandas as pd

# Configure pandas display
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🍽️  FlavorFlow Craft - Menu Engineering Solution                            ║
║                                                                              ║
║   Deloitte x AUC Hackathon 2024-2025                                         ║
║   Empowering restaurants with data-driven menu optimization                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_analysis(data_dir: Path, output_dir: Path, verbose: bool = True):
    """
    Run the menu engineering analysis pipeline.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path for output files
        verbose: Print progress messages
    
    Returns:
        Analysis results dictionary
    """
    from services.menu_analysis_service import MenuAnalysisService
    from services.reporting_service import ReportingService
    
    print("\n🔄 Starting Menu Engineering Analysis...")
    print(f"   Data directory: {data_dir}")
    print(f"   Output directory: {output_dir}")
    print("-" * 60)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize service
    service = MenuAnalysisService(data_dir=data_dir)
    
    # Load data
    print("\n📊 Loading data...")
    service.load_data()
    
    # Run full analysis
    print("\n🔬 Running analysis pipeline...")
    results = service.run_full_analysis()
    
    # Generate executive summary
    print("\n📋 Generating executive summary...")
    summary = service.get_executive_summary()
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 ANALYSIS RESULTS")
    print("=" * 60)
    
    if 'data_overview' in summary:
        overview = summary['data_overview']
        print(f"\n📦 Data Overview:")
        print(f"   • Menu Items: {overview.get('total_items', 0):,}")
        print(f"   • Restaurants: {overview.get('total_restaurants', 0):,}")
        print(f"   • Orders: {overview.get('total_orders', 0):,}")
        print(f"   • Campaigns: {overview.get('total_campaigns', 0):,}")
    
    if 'bcg_breakdown' in summary:
        bcg = summary['bcg_breakdown']
        print(f"\n🎯 BCG Matrix Classification:")
        print(f"   ⭐ Stars:      {bcg.get('stars', 0):,} items")
        print(f"   🐴 Plowhorses: {bcg.get('plowhorses', 0):,} items")
        print(f"   ❓ Puzzles:    {bcg.get('puzzles', 0):,} items")
        print(f"   🐕 Dogs:       {bcg.get('dogs', 0):,} items")
    
    if 'pricing_opportunity' in summary:
        pricing = summary['pricing_opportunity']
        print(f"\n💰 Pricing Optimization Opportunity:")
        print(f"   • Items to reprice: {pricing.get('items_to_reprice', 0)}")
        print(f"   • Potential revenue gain: {pricing.get('total_revenue_gain', 0):,.2f} DKK")
    
    # Export results
    print("\n💾 Exporting results...")
    service.export_results(output_dir)
    
    print("\n✅ Analysis complete!")
    print(f"   Results saved to: {output_dir}")
    
    return results


def run_inventory_analysis(data_dir: Path, output_dir: Path, verbose: bool = True):
    """
    Run the inventory optimization analysis pipeline.
    
    This trains on order-level data (2M+ transactions) to:
    1. Analyze customer behavior patterns
    2. Train demand forecasting model
    3. Calculate optimal inventory levels
    4. Generate stock alerts
    
    Args:
        data_dir: Path to data directory
        output_dir: Path for output files
        verbose: Print progress messages
    
    Returns:
        Analysis results dictionary
    """
    from services.inventory_analysis_service import InventoryAnalysisService
    from services.inventory_visualizations import InventoryVisualizationService
    
    print("\n🔄 Starting Inventory Optimization Analysis...")
    print(f"   Data directory: {data_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Training on: Order-level data (2M+ transactions)")
    print("-" * 60)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize service
    service = InventoryAnalysisService(data_dir=data_dir)
    
    # Run full analysis pipeline
    results = service.run_full_analysis(verbose=verbose)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    viz_service = InventoryVisualizationService(output_dir=output_dir)
    
    # Create dashboard
    viz_service.create_executive_dashboard(results)
    
    # Plot temporal patterns
    if 'behavior' in results and 'temporal' in results['behavior']:
        hourly = pd.DataFrame(results['behavior']['temporal']['hourly'])
        daily = pd.DataFrame(results['behavior']['temporal']['daily'])
        viz_service.plot_temporal_patterns(hourly, daily)
    
    # Plot feature importance
    if 'demand_model' in results and 'feature_importance' in results['demand_model']:
        importance = pd.DataFrame(results['demand_model']['feature_importance'])
        viz_service.plot_feature_importance(importance)
    
    # Plot model performance
    if 'demand_model' in results and 'metrics' in results['demand_model']:
        viz_service.plot_model_performance(results['demand_model']['metrics'])
    
    # Plot inventory status
    if 'inventory' in results and 'analysis' in results['inventory']:
        viz_service.plot_inventory_status(results['inventory']['analysis'])
    
    viz_service.close_all()
    
    # Export data results
    print("\n💾 Exporting results...")
    service.export_results(output_dir)
    
    # Print final summary
    summary = service.get_executive_summary()
    print("\n" + "=" * 60)
    print("📈 INVENTORY OPTIMIZATION RESULTS")
    print("=" * 60)
    
    if 'data_overview' in summary:
        overview = summary['data_overview']
        print(f"\n📦 Data Processed:")
        print(f"   • Orders: {overview.get('total_orders', 0):,}")
        print(f"   • Order Items: {overview.get('total_order_items', 0):,}")
        print(f"   • Restaurants: {overview.get('restaurants', 0):,}")
        print(f"   • Menu Items: {overview.get('menu_items', 0):,}")
    
    if 'model_performance' in summary:
        perf = summary['model_performance']
        print(f"\n🤖 Demand Model Performance:")
        print(f"   • MAE: {perf.get('mae', 0):.2f} units")
        print(f"   • RMSE: {perf.get('rmse', 0):.2f} units")
        print(f"   • R²: {perf.get('r2', 0):.4f}")
        print(f"   • Training time: {perf.get('training_time_seconds', 0):.1f}s")
    
    if 'inventory_alerts' in summary:
        alerts = summary['inventory_alerts']
        print(f"\n🚨 Inventory Alerts:")
        print(f"   🔴 Critical: {alerts.get('critical_items', 0)} items")
        print(f"   🟠 Low Stock: {alerts.get('low_stock_items', 0)} items")
        print(f"   🔵 Excess: {alerts.get('excess_items', 0)} items")
    
    print("\n✅ Inventory analysis complete!")
    print(f"   Results saved to: {output_dir}")
    
    return results


def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.
    
    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload for development
    """
    try:
        import uvicorn
    except ImportError:
        print("❌ Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)
    
    print(f"\n🚀 Starting FlavorFlow Craft API Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Docs: http://{host}:{port}/docs")
    print("-" * 60)
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="FlavorFlow Craft - Menu Engineering & Inventory Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command (legacy BCG analysis)
    analyze_parser = subparsers.add_parser('analyze', help='Run menu engineering analysis (BCG)')
    analyze_parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=PROJECT_ROOT / 'data',
        help='Path to data directory'
    )
    analyze_parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=PROJECT_ROOT / 'docs',
        help='Path for output files'
    )
    analyze_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    # Inventory analysis command (NEW - trains on orders)
    inventory_parser = subparsers.add_parser(
        'inventory', 
        help='Run inventory optimization analysis (trains on 2M+ order records)'
    )
    inventory_parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=PROJECT_ROOT / 'data',
        help='Path to data directory'
    )
    inventory_parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=PROJECT_ROOT / 'docs',
        help='Path for output files'
    )
    inventory_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host address (default: 0.0.0.0)'
    )
    serve_parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port number (default: 8000)'
    )
    serve_parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        run_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
    elif args.command == 'inventory':
        run_inventory_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
    elif args.command == 'serve':
        start_api_server(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    else:
        # Default: show help and run inventory analysis
        print("Available commands:")
        print("  inventory  - Run inventory optimization (trains on 2M+ orders)")
        print("  analyze    - Run menu engineering BCG analysis")
        print("  serve      - Start API server")
        print("\nUse --help for more information.\n")
        print("Running inventory analysis by default...\n")
        run_inventory_analysis(
            data_dir=PROJECT_ROOT / 'data',
            output_dir=PROJECT_ROOT / 'docs'
        )


if __name__ == "__main__":
    main()
