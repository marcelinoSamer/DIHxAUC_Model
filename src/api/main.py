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

# Global service instance (initialized on startup)
_analysis_service = None


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
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Basic validation
        if 'item_id' not in df.columns or 'current_stock' not in df.columns:
             raise HTTPException(status_code=400, detail="CSV must contain 'item_id' and 'current_stock'")

        reports = []
        timestamp = datetime.utcnow()
        
        for _, row in df.iterrows():
            report = InventoryReport(
                date=timestamp,
                item_id=int(row['item_id']),
                current_stock=float(row['current_stock']),
                reorder_point=float(row.get('reorder_point', 0)),
                safety_stock=float(row.get('safety_stock', 0))
            )
            reports.append(report)
            
        db.add_all(reports)
        db.commit()
        
        return {"status": "success", "items_processed": len(reports)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/weekly", tags=["Analysis"])
def get_weekly_recommendations(db: Session = Depends(get_db)):
    """
    Get weekly recommendations based on latest inventory and sales data.
    """
    # 1. Get latest inventory snapshot
    latest_reports = db.query(InventoryReport).order_by(desc(InventoryReport.date)).all()
    
    # Simple logic for now: Identify low stock items
    recommendations = []
    
    seen_items = set()
    for report in latest_reports:
        if report.item_id in seen_items:
            continue
        seen_items.add(report.item_id)
        
        item = db.query(MenuItem).filter(MenuItem.id == report.item_id).first()
        item_name = item.name if item else f"Item {report.item_id}"
        
        if report.current_stock < report.reorder_point:
             recommendations.append({
                 "type": "Restock Alert",
                 "item": item_name,
                 "message": f"Stock is low ({report.current_stock}). Reorder point is {report.reorder_point}.",
                 "priority": "High"
             })
        elif report.current_stock > (report.reorder_point * 3): # Arbitrary excess logic
             recommendations.append({
                 "type": "Excess Stock",
                 "item": item_name,
                 "message": f"Excess stock detected ({report.current_stock}). Consider running a promotion.",
                 "priority": "Medium"
             })
             
    return recommendations




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
        results = service.run_full_analysis(
            include_predictions=request.include_predictions,
            include_clustering=request.include_clustering
        )
        
        # Format response
        summary = service.get_executive_summary()
        
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
            recommendations=[
                RecommendationResponse(
                    category=MenuCategory(rec['category'].lower().replace('⭐ ', '').replace('🐴 ', '').replace('❓ ', '').replace('🐕 ', '')),
                    action=rec['recommendation'],
                    items_affected=rec.get('items_count', 0),
                    priority=rec.get('priority', 'medium')
                )
                for rec in results.get('recommendations', [])
            ],
            pricing_suggestions=[
                PricingSuggestion(
                    item_id=sug['item_id'],
                    current_price=sug['current_price'],
                    suggested_price=sug['suggested_price'],
                    price_change=sug['price_change'],
                    price_change_pct=sug.get('price_change_pct', 0),
                    rationale=sug.get('rationale', 'Based on analysis')
                )
                for sug in results.get('pricing_suggestions', [])[:20]  # Limit to top 20
            ],
            executive_summary=summary.get('summary_text', '')
        )

        # Side-effect: feed BCG results into chat context (best-effort)
        try:
            from .chat import get_chat_service
            chat_svc = get_chat_service()
            chat_svc.load_analysis_context(
                bcg_results={"executive_summary": summary, **results},
                datasets=service._datasets,
            )
        except Exception:
            pass

        return response
        
    except Exception as e:
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
        global _analysis_service
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
