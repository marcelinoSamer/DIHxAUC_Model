# Menu Intelligence Analysis - Output Directory

This directory contains all generated outputs from the analysis:

## CSV Results
- `results_menu_engineering.csv` - BCG matrix classification for all menu items
- `results_recommendations.csv` - Action items for each category
- `results_item_clusters.csv` - ML clustering results with segment labels
- `results_restaurant_performance.csv` - Location-level performance metrics

## Visualizations
- `menu_engineering_matrix.png` - BCG matrix scatter plot with category distribution
- `pricing_analysis.png` - Price distribution histogram and price vs. demand chart
- `campaign_analysis.png` - Discount distribution and campaign type breakdown
- `clustering_analysis.png` - Elbow plot and PCA cluster visualization

## Usage

To regenerate all outputs, run:

```bash
python -m src.main
```

Or from Python:

```python
from src.main import run_full_analysis
results = run_full_analysis(show_plots=True, save_outputs=True)
```
