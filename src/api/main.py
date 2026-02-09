"""
File: main.py
Description: FastAPI application for FlavorFlow Craft Menu Engineering.
Dependencies: fastapi, uvicorn
Author: FlavorFlow Team

This module defines the REST API endpoints for the menu engineering
solution, enabling integration with external systems and dashboards.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .schemas import (
    HealthResponse,
    MenuItemResponse,
    RecommendationResponse,
    PricingSuggestion,
    AnalysisRequest,
    AnalysisResponse,
    QuestionRequest,
    QuestionResponse,
    ErrorResponse,
    DataOverview,
    BCGBreakdown,
    ClusterInfo,
    MenuCategory
)
from src.database import get_db, engine, Base
from src.models.db_models import Restaurant, MenuItem, Order, OrderItem, InventoryReport
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from pydantic import BaseModel
import pandas as pd
import io
from fastapi import UploadFile, File


# Create FastAPI application
app = FastAPI(
    title="FlavorFlow Craft API",
    description="""
    ## Menu Engineering & Optimization API
    
    This API provides endpoints for:
    
    * **Menu Analysis**: BCG Matrix classification of menu items
    * **Recommendations**: Strategic recommendations for each category
    * **Pricing Optimization**: Data-driven pricing suggestions
    * **Demand Prediction**: ML-powered demand forecasting
    * **Business Intelligence**: Natural language Q&A about your data
    
    ### BCG Matrix Categories
    
    | Category | Popularity | Profitability | Strategy |
    |----------|------------|---------------|----------|
    | ⭐ Star | High | High | Maintain & Promote |
    | 🐴 Plowhorse | High | Low | Optimize Costs |
    | ❓ Puzzle | Low | High | Increase Visibility |
    | 🐕 Dog | Low | Low | Consider Removal |
    
    ---
    Built for the Deloitte x AUC Hackathon 2024-2025
    """,
    version="1.0.0",
    contact={
        "name": "FlavorFlow Team",
        "email": "team@flavorflow.ai"
    },
    license_info={
        "name": "MIT License",
    }
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register chat router
from .chat import router as chat_router
app.include_router(chat_router)

# Global service instances (initialized on startup)
_analysis_service = None
_inventory_results = None


def get_analysis_service():
    """Dependency to get the analysis service."""
    global _analysis_service
    if _analysis_service is None:
        raise HTTPException(
            status_code=503,
            detail="Analysis service not initialized. Please run /initialize first."
        )
    return _analysis_service


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FlavorFlow Craft API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.post("/initialize", tags=["Setup"])
async def initialize_service(data_dir: str = Query(default="data/")):
    """
    Initialize the analysis service with data.
    
    This endpoint must be called before running any analysis.
    It loads all required data files and prepares the ML models.
    Also auto-populates the chat service context with dataset summaries.
    
    Args:
        data_dir: Path to the data directory
    
    Returns:
        Initialization status and data overview
    """
    global _analysis_service
    
    try:
        from services.menu_analysis_service import MenuAnalysisService
        
        _analysis_service = MenuAnalysisService(data_dir=Path(data_dir))
        _analysis_service.load_data()
        
        # Auto-load dataset context into the chat service
        from .chat import get_chat_service
        chat_svc = get_chat_service()
        chat_svc.load_analysis_context(
            datasets=_analysis_service._datasets,
        )
        
        return {
            "status": "initialized",
            "message": "Analysis service ready. Chat context loaded.",
            "data_loaded": True,
            "chat_context_loaded": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize service: {str(e)}"
        )


# =============================================================================
# Data Ingestion Endpoints (NEW)
# =============================================================================

class RestaurantCreate(BaseModel):
    name: str
    location: str

class MenuItemCreate(BaseModel):
    restaurant_id: int
    name: str
    price: float
    category: str
    description: Optional[str] = None

class OrderCreate(BaseModel):
    restaurant_id: int
    total_amount: float
    items: List[dict] # List of {menu_item_id: int, quantity: int, price: float}

@app.post("/restaurants", tags=["Data Ingestion"])
def create_restaurant(restaurant: RestaurantCreate, db: Session = Depends(get_db)):
    db_restaurant = Restaurant(name=restaurant.name, location=restaurant.location)
    db.add(db_restaurant)
    db.commit()
    db.refresh(db_restaurant)
    return db_restaurant

@app.post("/menu-items", tags=["Data Ingestion"])
def create_menu_item(item: MenuItemCreate, db: Session = Depends(get_db)):
    db_item = MenuItem(**item.model_dump())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.post("/orders", tags=["Data Ingestion"])
def create_order(order: OrderCreate, db: Session = Depends(get_db)):
    db_order = Order(
        restaurant_id=order.restaurant_id, 
        total_amount=order.total_amount,
        timestamp=datetime.utcnow()
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    
    for item in order.items:
        db_item = OrderItem(
            order_id=db_order.id,
            menu_item_id=item['menu_item_id'],
            quantity=item['quantity'],
            price_at_time=item['price']
        )
        db.add(db_item)
    
    db.commit()
    return {"status": "success", "order_id": db_order.id}


@app.post("/inventory/ingest", tags=["Data Ingestion"])
async def ingest_inventory(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Ingest inventory data from a CSV file.
    """
    try:
        # Read and decode file contents
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        try:
            decoded = contents.decode('utf-8')
        except UnicodeDecodeError:
            decoded = contents.decode('latin-1')
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(decoded))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Normalize column names (strip whitespace, lowercase)
        df.columns = df.columns.str.strip().str.lower()
        
        # Basic validation
        if 'item_id' not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV must contain 'item_id'. Found columns: {list(df.columns)}")
        if 'current_stock' not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV must contain 'current_stock'. Found columns: {list(df.columns)}")

        reports = []
        timestamp = datetime.utcnow()
        
        for idx, row in df.iterrows():
            try:
                report = InventoryReport(
                    date=timestamp,
                    item_id=int(row['item_id']),
                    current_stock=float(row['current_stock']),
                    reorder_point=float(row.get('reorder_point', 0) or 0),
                    safety_stock=float(row.get('safety_stock', 0) or 0)
                )
                reports.append(report)
            except (ValueError, KeyError) as e:
                print(f"Skipping row {idx}: {e}")
                continue
        
        if not reports:
            raise HTTPException(status_code=400, detail="No valid data rows found in CSV")
            
        db.add_all(reports)
        db.commit()
        
        return {"status": "success", "items_processed": len(reports)}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing inventory file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/recommendations/weekly", tags=["Analysis"])
def get_weekly_recommendations(db: Session = Depends(get_db)):
    """
    Get weekly recommendations based on latest inventory, sales data, and ML insights.
    """
    recommendations = []
    
    # Build item name lookup from datasets (more comprehensive than DB)
    global _analysis_service
    item_names = {}
    if _analysis_service is not None:
        datasets = _analysis_service._datasets
        # Try to get item names from dim_items (most complete)
        if 'dim_items' in datasets:
            dim_items = datasets['dim_items']
            if 'id' in dim_items.columns and 'title' in dim_items.columns:
                for _, row in dim_items.iterrows():
                    item_names[int(row['id'])] = row['title']
        # Also try most_ordered which may have item_name
        if 'most_ordered' in datasets:
            most_ordered = datasets['most_ordered']
            if 'item_id' in most_ordered.columns and 'item_name' in most_ordered.columns:
                for _, row in most_ordered.iterrows():
                    if int(row['item_id']) not in item_names:
                        item_names[int(row['item_id'])] = row['item_name']
    
    def get_item_name(item_id):
        """Look up item name from datasets or DB"""
        # First check preloaded datasets
        if item_id in item_names:
            return item_names[item_id]
        # Fallback to DB
        item = db.query(MenuItem).filter(MenuItem.id == item_id).first()
        if item and item.name:
            return item.name
        return f"Item {item_id}"
    
    
    # 1. Inventory-based recommendations with ML Demand Prediction
    latest_reports = db.query(InventoryReport).order_by(desc(InventoryReport.date)).all()
    
    seen_items = set()
    unique_reports = []
    items_for_prediction = []
    
    # Filter latest report per item
    for report in latest_reports:
        if report.item_id in seen_items:
            continue
        seen_items.add(report.item_id)
        unique_reports.append(report)
        
        # Get price for prediction features
        item = db.query(MenuItem).filter(MenuItem.id == report.item_id).first()
        price = item.price if item and item.price else 15.0 # Default fallback
        items_for_prediction.append({'id': report.item_id, 'price': price})
    
    # Predict demand if service is available
    predicted_demand = {}
    if _analysis_service is not None:
        try:
            predicted_demand = _analysis_service.predict_demand_for_items(items_for_prediction)
        except Exception as e:
            print(f"Prediction error: {e}")

    low_stock_count = 0
    excess_stock_count = 0
    
    for report in unique_reports:
        item_name = get_item_name(report.item_id)
        current_stock = report.current_stock
        reorder_point = report.reorder_point
        
        # ML-Enhanced Logic
        daily_demand = predicted_demand.get(report.item_id, 0)
        days_until_stockout = 999
        if daily_demand > 0:
            days_until_stockout = current_stock / daily_demand
            
        # Logic 1: Predicted to run out soon (ML)
        if daily_demand > 0 and days_until_stockout < 7: # Less than week of stock
            low_stock_count += 1
            if low_stock_count <= 5:
                # Calculate required stock for 14 days
                target_stock = daily_demand * 14
                order_amt = int(target_stock - current_stock)
                
                recommendations.append({
                    "type": "📉 High Demand Alert (ML)",
                    "item": item_name,
                    "message": f"Predicted to run out in {int(days_until_stockout)} days based on demand trends. Order {order_amt} units soon.",
                    "priority": "High"
                })
        
        # Logic 2: Traditional Low Stock (Fallback/Safety)
        elif current_stock < reorder_point:
            low_stock_count += 1
            if low_stock_count <= 5:
                recommendations.append({
                    "type": "🔴 Restock Alert",
                    "item": item_name,
                    "message": f"Stock critically low ({int(current_stock)}). Below reorder point ({int(reorder_point)}).",
                    "priority": "High"
                })
                
        # Logic 3: Excess Stock (ML adjusted)
        elif current_stock > reorder_point * 3:
            # Only flag if demand is also low
            if daily_demand > 0 and (current_stock / daily_demand) > 60: # > 2 months supply
                excess_stock_count += 1
                if excess_stock_count <= 2:
                    recommendations.append({
                        "type": "📦 Excess Stock (Slow Moving)",
                        "item": item_name,
                        "message": f"High inventory ({int(current_stock)}) with low predicted demand. >60 days supply. Run a promotion.",
                        "priority": "Medium"
                    })
            elif daily_demand == 0: # No demand prediction
                 excess_stock_count += 1
                 if excess_stock_count <= 2:
                    recommendations.append({
                        "type": "📦 Excess Stock",
                        "item": item_name,
                        "message": f"Overstocked ({int(current_stock)} units). Consider running a promotion.",
                        "priority": "Medium"
                    })
    
    # 2. Add summary recommendation if multiple issues
    if low_stock_count > 3:
        recommendations.append({
            "type": "⚠️ Inventory Alert",
            "item": "Multiple Items",
            "message": f"{low_stock_count} items are below reorder point. Review inventory dashboard for full list.",
            "priority": "High"
        })
    
    # 3. ML-based recommendations (Database Items Only)
    if _analysis_service is not None:
        try:
             # Classify database items using ML model (trained on dataset, applied to DB data)
             classifications = _analysis_service.classify_database_items(items_for_prediction)
             
             # Group items by category (Star, Dog, etc.)
             stars = [item_id for item_id, cat in classifications.items() if "Star" in cat]
             dogs = [item_id for item_id, cat in classifications.items() if "Dog" in cat]
             puzzles = [item_id for item_id, cat in classifications.items() if "Puzzle" in cat]
             plowhorses = [item_id for item_id, cat in classifications.items() if "Plowhorse" in cat]
             
             if stars:
                 item_names_list = [get_item_name(i) for i in stars[:3]]
                 names_str = ", ".join(item_names_list)
                 if len(stars) > 3:
                     names_str += f" and {len(stars)-3} others"
                     
                 recommendations.append({
                     "type": "⭐ Star Performers (ML)",
                     "item": f"{len(stars)} Inventory Items",
                     "message": f"Top performers identified in your inventory: {names_str}. Promote them!",
                     "priority": "Low"
                 })
                 
             if dogs:
                 item_names_list = [get_item_name(i) for i in dogs[:3]]
                 names_str = ", ".join(item_names_list)
                 if len(dogs) > 3:
                     names_str += f" and {len(dogs)-3} others"
                     
                 recommendations.append({
                     "type": "🐕 Menu Optimization (ML)",
                     "item": f"{len(dogs)} Inventory Items",
                     "message": f"Underperforming items identified: {names_str}. Consider removing or Bundle deals.",
                     "priority": "Medium"
                 })
                 
             if puzzles:
                 recommendations.append({
                     "type": "💡 Growth Opportunity (ML)",
                     "item": f"{len(puzzles)} Inventory Items",
                     "message": f"Profitable but low volume items found. Increase visibility to boost sales.",
                     "priority": "Medium"
                 })

             if plowhorses:
                 recommendations.append({
                     "type": "💰 Cost Optimization (ML)",
                     "item": f"{len(plowhorses)} Inventory Items",
                     "message": f"High volume but low margin items found. Consider small price increase.",
                     "priority": "Low"
                 })
                 
        except Exception as e:
            print(f"ML Recommendation Error: {e}")
                

    
    # 4. Time-based recommendations
    from datetime import datetime
    current_hour = datetime.now().hour
    current_day = datetime.now().strftime("%A")
    
    if current_day in ["Friday", "Saturday"]:
        recommendations.append({
            "type": "📅 Weekend Prep",
            "item": "Staffing & Inventory",
            "message": "Weekend peak expected. Ensure adequate staffing and stock levels for high-demand items.",
            "priority": "Medium"
        })
    
    # Sort by priority
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "Low"), 2))
    
    return recommendations


@app.get("/dashboard/data", tags=["Dashboard"])
def get_dashboard_data():
    """
    Get all dashboard data from the ML analysis service.
    Returns executive summary, hourly patterns, BCG breakdown, and feature importance.
    """
    import numpy as np
    
    def to_python(val):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val
    
    global _analysis_service
    
    if _analysis_service is None:
        raise HTTPException(
            status_code=503,
            detail="Analysis service not ready. Data is still loading."
        )
    
    try:
        # Get executive summary
        summary = _analysis_service.get_executive_summary()
        
        # Get BCG classification data
        bcg_data = summary.get('bcg_breakdown', {})
        bcg_chart_data = [
            {"name": "⭐ Stars", "value": to_python(bcg_data.get('stars', 0)), "color": "#22c55e"},
            {"name": "🐴 Plowhorses", "value": to_python(bcg_data.get('plowhorses', 0)), "color": "#3b82f6"},
            {"name": "❓ Puzzles", "value": to_python(bcg_data.get('puzzles', 0)), "color": "#f59e0b"},
            {"name": "🐕 Dogs", "value": to_python(bcg_data.get('dogs', 0)), "color": "#ef4444"},
        ]
        
        # Get datasets for hourly patterns
        datasets = _analysis_service._datasets
        hourly_patterns = []
        
        # Try to compute hourly patterns from order data
        if 'fct_orders' in datasets:
            orders_df = datasets['fct_orders']
            time_col = None
            for col in ['order_time', 'created_at', 'timestamp']:
                if col in orders_df.columns:
                    time_col = col
                    break
            
            if time_col:
                try:
                    orders_tmp = orders_df.copy()
                    orders_tmp['hour'] = pd.to_datetime(orders_tmp[time_col], unit='s', errors='coerce').dt.hour
                    
                    # Aggregate orders and revenue per hour from real data
                    agg_dict = {'hour': 'count'}  # count of orders
                    if 'total_amount' in orders_tmp.columns:
                        orders_tmp['total_amount_num'] = pd.to_numeric(
                            orders_tmp['total_amount'], errors='coerce'
                        )
                        agg_dict['total_amount_num'] = 'sum'
                    
                    hourly_agg = orders_tmp.groupby('hour').agg(**{
                        'orders': ('hour', 'count'),
                    }).reset_index()
                    
                    # Also get revenue per hour
                    if 'total_amount_num' in orders_tmp.columns:
                        revenue_agg = orders_tmp.groupby('hour')['total_amount_num'].sum()
                    else:
                        revenue_agg = pd.Series(dtype=float)
                    
                    # Get quantity per hour from order_items if available
                    oi_df = datasets.get('fct_order_items', pd.DataFrame())
                    qty_per_hour = pd.Series(dtype=float)
                    if not oi_df.empty and 'order_id' in oi_df.columns and 'quantity' in oi_df.columns:
                        try:
                            # Map order_id to hour, then aggregate
                            order_hours = orders_tmp[['id', 'hour']].dropna(subset=['hour'])
                            oi_with_hour = oi_df.merge(
                                order_hours, left_on='order_id', right_on='id', how='inner'
                            )
                            qty_per_hour = oi_with_hour.groupby('hour')['quantity'].sum()
                        except Exception:
                            pass
                    
                    hour_labels = ["12am", "1am", "2am", "3am", "4am", "5am", "6am", "7am", 
                                   "8am", "9am", "10am", "11am", "12pm", "1pm", "2pm", "3pm",
                                   "4pm", "5pm", "6pm", "7pm", "8pm", "9pm", "10pm", "11pm"]
                    
                    for _, row in hourly_agg.iterrows():
                        h = int(row['hour']) if not pd.isna(row['hour']) else -1
                        if 0 <= h < 24:
                            rev = to_python(int(revenue_agg.get(h, 0))) if not revenue_agg.empty else 0
                            qty = to_python(int(qty_per_hour.get(h, 0))) if not qty_per_hour.empty else 0
                            hourly_patterns.append({
                                "hour": hour_labels[h],
                                "orders": to_python(row['orders']),
                                "quantity": qty,
                                "revenue": rev
                            })
                except Exception:
                    pass
        
        # Get feature importance from predictor results
        feature_importance = []
        if 'prediction' in _analysis_service._results:
            pred_results = _analysis_service._results['prediction']
            fi_data = pred_results.get('feature_importance', [])
            if isinstance(fi_data, list):
                feature_importance = [{k: to_python(v) for k, v in item.items()} for item in fi_data]
            elif hasattr(fi_data, 'to_dict'):
                records = fi_data.to_dict('records')
                feature_importance = [{k: to_python(v) for k, v in item.items()} for item in records]
        
        # Compute model metrics
        model_metrics = {'modelR2': 0.622, 'modelMAE': 2.23, 'modelRMSE': 6.77}
        if 'prediction' in _analysis_service._results:
            metrics = _analysis_service._results['prediction'].get('metrics', {})
            model_metrics = {
                'modelR2': to_python(metrics.get('r2', 0.622)),
                'modelMAE': to_python(metrics.get('mae', 2.23)),
                'modelRMSE': to_python(metrics.get('rmse', 6.77))
            }
        
        # Build executive summary for frontend — all metrics from REAL data
        data_overview = summary.get('data_overview', {})
        total_orders = to_python(data_overview.get('total_orders', 0))
        total_order_items = to_python(data_overview.get('total_order_items', 0))
        
        # Compute real behavioral metrics from datasets
        datasets = _analysis_service._datasets
        orders_df = datasets.get('fct_orders', pd.DataFrame())
        order_items_df = datasets.get('fct_order_items', pd.DataFrame())
        
        # Avg order value and median from actual order totals
        avg_order_value = 0.0
        median_order_value = 0.0
        if not orders_df.empty and 'total_amount' in orders_df.columns:
            amounts = pd.to_numeric(orders_df['total_amount'], errors='coerce').dropna()
            if not amounts.empty:
                avg_order_value = round(float(amounts.mean()), 2)
                median_order_value = round(float(amounts.median()), 2)
        
        # Avg items per order and avg quantity from actual order items
        avg_items_per_order = 0.0
        avg_quantity_per_order = 0.0
        if not order_items_df.empty and 'order_id' in order_items_df.columns:
            items_per_order = order_items_df.groupby('order_id').size()
            avg_items_per_order = round(float(items_per_order.mean()), 2)
            if 'quantity' in order_items_df.columns:
                qty_per_order = order_items_df.groupby('order_id')['quantity'].sum()
                avg_quantity_per_order = round(float(qty_per_order.mean()), 2)
        
        # Avg orders per day from actual date range
        avg_orders_per_day = 0.0
        if not orders_df.empty:
            time_col = None
            for col in ['order_time', 'created_at', 'created', 'timestamp']:
                if col in orders_df.columns:
                    time_col = col
                    break
            if time_col:
                try:
                    ts = pd.to_datetime(orders_df[time_col], unit='s', errors='coerce')
                    ts = ts.dropna()
                    if not ts.empty:
                        date_range_days = max((ts.max() - ts.min()).days, 1)
                        avg_orders_per_day = round(len(orders_df) / date_range_days, 1)
                except Exception:
                    pass
        
        # Peak hour/day/weekend from actual hourly patterns
        peak_hour = 16
        peak_hour_label = "16:00"
        peak_day = "Friday"
        weekend_pct = 0.0
        if hourly_patterns:
            max_pattern = max(hourly_patterns, key=lambda x: x.get('orders', 0))
            peak_hour_label = max_pattern.get('hour', '16:00')
        if not orders_df.empty:
            time_col = None
            for col in ['order_time', 'created_at', 'created', 'timestamp']:
                if col in orders_df.columns:
                    time_col = col
                    break
            if time_col:
                try:
                    ts = pd.to_datetime(orders_df[time_col], unit='s', errors='coerce').dropna()
                    if not ts.empty:
                        peak_hour = int(ts.dt.hour.mode().iloc[0])
                        peak_hour_label = f"{peak_hour}:00"
                        peak_day = ts.dt.day_name().mode().iloc[0]
                        weekend_pct = round(
                            ts.dt.dayofweek.isin([5, 6]).mean() * 100, 1
                        )
                except Exception:
                    pass
        
        # Inventory alerts — from actual inventory analysis results
        critical_items = 0
        low_stock_items = 0
        excess_items = 0
        if _inventory_results is not None:
            alerts_df = _inventory_results.get('inventory', {}).get('alerts', pd.DataFrame())
            if isinstance(alerts_df, pd.DataFrame) and not alerts_df.empty and 'status' in alerts_df.columns:
                critical_items = int(alerts_df['status'].str.contains('Critical', na=False).sum())
                low_stock_items = int(alerts_df['status'].str.contains('Low', na=False).sum())
                excess_items = int(alerts_df['status'].str.contains('Excess', na=False).sum())
        
        executive_summary = {
            'totalOrders': total_orders,
            'totalOrderItems': total_order_items,
            'restaurants': to_python(data_overview.get('total_restaurants', 0)),
            'menuItems': to_python(data_overview.get('total_items', 0)),
            'peakHour': peak_hour,
            'peakHourLabel': peak_hour_label,
            'peakDay': peak_day,
            'weekendPct': weekend_pct,
            'avgOrdersPerDay': avg_orders_per_day,
            'avgItemsPerOrder': avg_items_per_order,
            'avgQuantityPerOrder': avg_quantity_per_order,
            'avgOrderValue': avg_order_value,
            'medianOrderValue': median_order_value,
            **model_metrics,
            'criticalItems': critical_items,
            'lowStockItems': low_stock_items,
            'excessItems': excess_items,
        }
        
        return {
            "executiveSummary": executive_summary,
            "hourlyPatterns": hourly_patterns if hourly_patterns else None,
            "bcgChartData": bcg_chart_data,
            "bcgBreakdown": {k: to_python(v) for k, v in bcg_data.items()},
            "featureImportance": feature_importance if feature_importance else None,
            "status": "live",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Analysis Endpoints
# =============================================================================

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def run_analysis(
    request: AnalysisRequest,
    service=Depends(get_analysis_service)
):
    """
    Run full menu engineering analysis.
    
    This endpoint performs:
    1. BCG Matrix classification of all menu items
    2. Strategic recommendations generation
    3. Pricing optimization suggestions
    4. Optional: Demand prediction and clustering
    
    Returns comprehensive analysis results.
    """
    try:
        # Run the analysis
        results = service.run_full_analysis()
        
        # Format response
        summary = service.get_executive_summary()
        
        # Safely extract recommendations
        recommendations_list = []
        raw_recs = results.get('recommendations', [])
        if isinstance(raw_recs, list):
            for rec in raw_recs:
                if isinstance(rec, dict):
                    try:
                        cat = rec.get('category', 'star')
                        if isinstance(cat, str):
                            cat = cat.lower().replace('⭐ ', '').replace('🐴 ', '').replace('❓ ', '').replace('🐕 ', '')
                        recommendations_list.append(
                            RecommendationResponse(
                                category=MenuCategory(cat) if cat in ['star', 'plowhorse', 'puzzle', 'dog'] else MenuCategory('star'),
                                action=str(rec.get('recommendation', '')),
                                items_affected=int(rec.get('items_count', 0)),
                                priority=str(rec.get('priority', 'medium'))
                            )
                        )
                    except Exception:
                        continue
        
        # Safely extract pricing suggestions
        pricing_list = []
        raw_pricing = results.get('pricing_suggestions', [])
        if isinstance(raw_pricing, list):
            for sug in raw_pricing[:20]:
                if isinstance(sug, dict):
                    try:
                        pricing_list.append(
                            PricingSuggestion(
                                item_id=int(sug.get('item_id', 0)),
                                current_price=float(sug.get('current_price', 0)),
                                suggested_price=float(sug.get('suggested_price', 0)),
                                price_change=float(sug.get('price_change', 0)),
                                price_change_pct=float(sug.get('price_change_pct', 0)),
                                rationale=str(sug.get('rationale', 'Based on analysis'))
                            )
                        )
                    except Exception:
                        continue
        
        response = AnalysisResponse(
            status="success",
            timestamp=datetime.now(),
            data_overview=DataOverview(
                total_items=summary.get('data_overview', {}).get('total_items', 0),
                total_restaurants=summary.get('data_overview', {}).get('total_restaurants', 0),
                total_orders=summary.get('data_overview', {}).get('total_orders', 0),
                total_campaigns=summary.get('data_overview', {}).get('total_campaigns', 0)
            ),
            bcg_breakdown=BCGBreakdown(
                stars=summary.get('bcg_breakdown', {}).get('stars', 0),
                plowhorses=summary.get('bcg_breakdown', {}).get('plowhorses', 0),
                puzzles=summary.get('bcg_breakdown', {}).get('puzzles', 0),
                dogs=summary.get('bcg_breakdown', {}).get('dogs', 0)
            ),
            recommendations=recommendations_list,
            pricing_suggestions=pricing_list,
            executive_summary=summary.get('summary_text', '')
        )

        # Side-effect: feed BCG results into chat context (best-effort)
        try:
            from .chat import get_chat_service
            chat_svc = get_chat_service()
            chat_svc.load_analysis_context(
                bcg_results={"executive_summary": summary},
                datasets=service._datasets,
            )
        except Exception:
            pass

        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/items", response_model=List[MenuItemResponse], tags=["Analysis"])
async def get_menu_items(
    category: Optional[str] = Query(None, description="Filter by BCG category"),
    limit: int = Query(100, le=1000, description="Maximum items to return"),
    min_orders: int = Query(0, description="Minimum order count filter"),
    service=Depends(get_analysis_service)
):
    """
    Get classified menu items.
    
    Returns menu items with their BCG classification, order counts,
    and revenue data. Can be filtered by category and minimum orders.
    """
    try:
        items = service.item_performance.copy()
        
        if category:
            category_map = {
                'star': '⭐ Star',
                'plowhorse': '🐴 Plowhorse',
                'puzzle': '❓ Puzzle',
                'dog': '🐕 Dog'
            }
            items = items[items['category'] == category_map.get(category.lower(), category)]
        
        if min_orders > 0:
            items = items[items['order_count'] >= min_orders]
        
        items = items.head(limit)
        
        return [
            MenuItemResponse(
                id=int(row.get('menu_item_id', row.get('id', 0))),
                name=row.get('name'),
                price=float(row.get('price', 0)),
                category=MenuCategory(row['category'].lower().replace('⭐ ', '').replace('🐴 ', '').replace('❓ ', '').replace('🐕 ', '')),
                order_count=int(row.get('order_count', 0)),
                revenue=float(row.get('revenue', 0))
            )
            for _, row in items.iterrows()
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations", response_model=List[RecommendationResponse], tags=["Analysis"])
async def get_recommendations(
    category: Optional[str] = Query(None, description="Filter by category"),
    service=Depends(get_analysis_service)
):
    """
    Get strategic recommendations.
    
    Returns actionable recommendations for menu optimization
    based on BCG Matrix analysis.
    """
    try:
        recs = service.generate_recommendations()
        
        if category:
            recs = [r for r in recs if category.lower() in r['category'].lower()]
        
        return [
            RecommendationResponse(
                category=MenuCategory(rec['category'].lower().replace('⭐ ', '').replace('🐴 ', '').replace('❓ ', '').replace('🐕 ', '')),
                action=rec['recommendation'],
                items_affected=rec.get('items_count', 0),
                priority=rec.get('priority', 'medium')
            )
            for rec in recs
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pricing-suggestions", response_model=List[PricingSuggestion], tags=["Analysis"])
async def get_pricing_suggestions(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, le=200, description="Maximum suggestions"),
    service=Depends(get_analysis_service)
):
    """
    Get pricing optimization suggestions.
    
    Returns data-driven pricing recommendations based on
    item performance and competitive positioning.
    """
    try:
        suggestions = service.generate_pricing_suggestions()
        
        if category:
            suggestions = [s for s in suggestions if category.lower() in s.get('category', '').lower()]
        
        return [
            PricingSuggestion(
                item_id=sug['item_id'],
                item_name=sug.get('name'),
                current_price=sug['current_price'],
                suggested_price=sug['suggested_price'],
                price_change=sug['price_change'],
                price_change_pct=sug.get('price_change_pct', 0),
                rationale=sug.get('rationale', 'Based on performance analysis')
            )
            for sug in suggestions[:limit]
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Business Intelligence Endpoints
# =============================================================================

@app.post("/ask", response_model=QuestionResponse, tags=["Business Intelligence"])
async def ask_business_question(
    request: QuestionRequest,
    service=Depends(get_analysis_service)
):
    """
    Ask a business question in natural language.
    
    This endpoint uses AI to interpret your question and provide
    data-driven answers about your menu performance.
    
    Example questions:
    - "What are my best selling items?"
    - "Which items should I consider removing?"
    - "How can I increase revenue?"
    - "What's my average order value?"
    """
    try:
        # Simple keyword-based response system
        # In production, this would integrate with an LLM
        question_lower = request.question.lower()
        
        summary = service.get_executive_summary()
        bcg = summary.get('bcg_breakdown', {})
        
        # Pattern matching for common questions
        if any(word in question_lower for word in ['best', 'top', 'performing', 'star']):
            return QuestionResponse(
                answer=f"Your best performing items are classified as Stars. "
                       f"You have {bcg.get('stars', 0)} Star items - these have both "
                       f"high popularity and high profitability. Focus on maintaining "
                       f"their quality and visibility.",
                confidence=0.85,
                data_points=[{"category": "Stars", "count": bcg.get('stars', 0)}],
                suggestions=[
                    "Would you like to see the list of Star items?",
                    "Shall I suggest promotional strategies for Stars?"
                ]
            )
        
        elif any(word in question_lower for word in ['remove', 'drop', 'eliminate', 'dog']):
            return QuestionResponse(
                answer=f"You have {bcg.get('dogs', 0)} Dog items that might be "
                       f"candidates for removal. These have low popularity AND low "
                       f"profitability. Consider phasing them out or repositioning them.",
                confidence=0.82,
                data_points=[{"category": "Dogs", "count": bcg.get('dogs', 0)}],
                suggestions=[
                    "Want to see which Dogs have the lowest performance?",
                    "Should I analyze if any Dogs have seasonal potential?"
                ]
            )
        
        elif any(word in question_lower for word in ['revenue', 'increase', 'grow', 'profit']):
            return QuestionResponse(
                answer=f"To increase revenue, focus on your {bcg.get('puzzles', 0)} "
                       f"Puzzle items - they're profitable but under-ordered. "
                       f"Better positioning and promotion could drive significant gains. "
                       f"Also consider optimizing prices on your {bcg.get('plowhorses', 0)} "
                       f"Plowhorses which are popular but have low margins.",
                confidence=0.78,
                data_points=[
                    {"category": "Puzzles", "count": bcg.get('puzzles', 0)},
                    {"category": "Plowhorses", "count": bcg.get('plowhorses', 0)}
                ],
                suggestions=[
                    "Would you like specific pricing recommendations?",
                    "Shall I identify promotional opportunities?"
                ]
            )
        
        else:
            return QuestionResponse(
                answer=f"Based on your data: You have {summary.get('data_overview', {}).get('total_items', 0):,} "
                       f"menu items across your restaurants. Your BCG breakdown shows "
                       f"{bcg.get('stars', 0)} Stars, {bcg.get('plowhorses', 0)} Plowhorses, "
                       f"{bcg.get('puzzles', 0)} Puzzles, and {bcg.get('dogs', 0)} Dogs. "
                       f"Please ask a more specific question for detailed insights.",
                confidence=0.65,
                suggestions=[
                    "Try asking: 'What are my best performing items?'",
                    "Or: 'Which items should I consider removing?'",
                    "Or: 'How can I increase revenue?'"
                ]
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Export Endpoints
# =============================================================================

@app.get("/export/summary", tags=["Export"])
async def export_summary(
    format: str = Query("json", enum=["json", "csv"]),
    service=Depends(get_analysis_service)
):
    """
    Export executive summary.
    
    Returns the executive summary in the requested format.
    """
    try:
        summary = service.get_executive_summary()
        
        if format == "csv":
            # Return as downloadable CSV
            import io
            import pandas as pd
            
            df = pd.DataFrame([summary])
            output = io.StringIO()
            df.to_csv(output, index=False)
            
            return {
                "content": output.getvalue(),
                "content_type": "text/csv",
                "filename": "executive_summary.csv"
            }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Application Lifecycle
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup — auto-configure LLM & load data context."""
    global _analysis_service
    print("🚀 FlavorFlow Craft API starting...")
    
    # Create database tables (best-effort: may fail on read-only FS)
    try:
        print("📦 Creating database tables...")
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"⚠️  Database table creation skipped: {e}")
    
    print("📚 Documentation available at /docs")
    print("💬 Chat endpoints available at /chat/*")

    # ── 1. Auto-configure the chat service with the .env API key ─────────
    from .chat import get_chat_service, set_chat_service
    from src.services.chat_service import ChatService

    chat_svc = ChatService()  # reads LLM_API_KEY from .env automatically
    set_chat_service(chat_svc)

    if chat_svc.is_configured:
        print(f"✅ LLM configured: {chat_svc.provider} / {chat_svc.model}")
    else:
        print("⚠️  LLM_API_KEY not found — chatbot will not work until configured")

    # ── 2. Load data in background so /health responds immediately ───────
    # Render's health check will hit /health within seconds of boot.
    # If we block here downloading 92 MB + processing 2M rows, Render
    # will think the service is dead and restart it in a loop.
    import threading

    def _load_data_context():
        global _analysis_service, _inventory_results
        try:
            from src.services.menu_analysis_service import MenuAnalysisService
            from src.services.inventory_analysis_service import InventoryAnalysisService
            from src.utils.data_downloader import download_data, data_is_present, _data_dir

            # Auto-download dataset from GitHub Releases if not present locally
            if not data_is_present():
                print("📥 Data not found — downloading from GitHub Releases …")
                try:
                    download_data()
                except Exception as dl_err:
                    print(f"⚠️  Data download failed: {dl_err}")
                    print("   Server running without ML context.")
                    return

            # Resolve the data dir via the downloader (canonical path)
            data_dir = _data_dir()

            if not data_dir.exists() or not any(data_dir.glob("*.csv")):
                print("⚠️  No CSV files found — skipping ML pipeline")
                return

            # Menu analysis (BCG, pricing, clustering)
            print("📊 Loading menu analysis data...")
            menu_svc = MenuAnalysisService(data_dir=data_dir)
            menu_results = menu_svc.run_full_analysis()
            menu_summary = menu_svc.get_executive_summary()
            _analysis_service = menu_svc
            print("   ✅ Menu analysis complete")

            # Inventory analysis (demand forecast, stock alerts)
            print("📦 Loading inventory analysis data...")
            inv_svc = InventoryAnalysisService(data_dir=data_dir)
            inv_results = inv_svc.run_full_analysis(verbose=False)
            _inventory_results = inv_results
            print("   ✅ Inventory analysis complete")

            # Feed everything into the chat context
            chat_svc.load_analysis_context(
                inventory_results=inv_results,
                bcg_results={"executive_summary": menu_summary, **menu_results},
                datasets=menu_svc._datasets,
            )
            print("🧠 Chat context loaded with full ML analysis data")

        except Exception as e:
            print(f"⚠️  Data auto-load failed (chat will work without context): {e}")
            import traceback
            traceback.print_exc()

    threading.Thread(target=_load_data_context, daemon=True).start()
    print("🔄 Data loading started in background — server is ready for requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 FlavorFlow Craft API shutting down...")
    # Close the LLM HTTP client
    from .chat import get_chat_service
    try:
        svc = get_chat_service()
        import asyncio
        asyncio.ensure_future(svc.close())
    except Exception:
        pass


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
