"""
File: schemas.py
Description: Pydantic models for API request/response validation.
Dependencies: pydantic
Author: FlavorFlow Team

This module defines all the data models used in the API
for request validation and response serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class MenuCategory(str, Enum):
    """BCG Matrix category classification."""
    STAR = "star"
    PLOWHORSE = "plowhorse"
    PUZZLE = "puzzle"
    DOG = "dog"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0")


class MenuItemResponse(BaseModel):
    """Response model for a single menu item."""
    id: int = Field(..., description="Menu item ID")
    name: Optional[str] = Field(None, description="Item name")
    price: float = Field(..., description="Item price in DKK")
    category: MenuCategory = Field(..., description="BCG Matrix category")
    order_count: int = Field(..., description="Total order count")
    revenue: float = Field(..., description="Total revenue generated")
    popularity_score: Optional[float] = Field(None, description="Popularity percentile")
    profitability_score: Optional[float] = Field(None, description="Profitability percentile")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 12345,
                "name": "Margherita Pizza",
                "price": 89.0,
                "category": "star",
                "order_count": 1523,
                "revenue": 135547.0,
                "popularity_score": 92.5,
                "profitability_score": 78.3
            }
        }


class RecommendationResponse(BaseModel):
    """Response model for a strategic recommendation."""
    category: MenuCategory = Field(..., description="Target category")
    action: str = Field(..., description="Recommended action")
    items_affected: int = Field(..., description="Number of items affected")
    priority: str = Field(..., description="Priority level (high/medium/low)")
    expected_impact: Optional[str] = Field(None, description="Expected business impact")
    details: Optional[str] = Field(None, description="Detailed explanation")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "puzzle",
                "action": "Increase visibility with promotions",
                "items_affected": 156,
                "priority": "high",
                "expected_impact": "15-25% revenue increase",
                "details": "These high-margin items need better positioning"
            }
        }


class PricingSuggestion(BaseModel):
    """Response model for a pricing suggestion."""
    item_id: int = Field(..., description="Menu item ID")
    item_name: Optional[str] = Field(None, description="Item name")
    current_price: float = Field(..., description="Current price in DKK")
    suggested_price: float = Field(..., description="Suggested price in DKK")
    price_change: float = Field(..., description="Price change amount")
    price_change_pct: float = Field(..., description="Price change percentage")
    rationale: str = Field(..., description="Reason for suggestion")
    estimated_revenue_impact: Optional[float] = Field(None, description="Revenue impact")

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": 12345,
                "item_name": "Caesar Salad",
                "current_price": 65.0,
                "suggested_price": 75.0,
                "price_change": 10.0,
                "price_change_pct": 15.38,
                "rationale": "High popularity supports price increase",
                "estimated_revenue_impact": 12500.0
            }
        }


class ClusterInfo(BaseModel):
    """Information about an item cluster."""
    cluster_id: int = Field(..., description="Cluster identifier")
    size: int = Field(..., description="Number of items in cluster")
    avg_price: float = Field(..., description="Average price in cluster")
    avg_orders: float = Field(..., description="Average order count")
    avg_revenue: float = Field(..., description="Average revenue")
    profile: str = Field(..., description="Cluster profile description")


class AnalysisRequest(BaseModel):
    """Request model for running analysis."""
    restaurant_id: Optional[int] = Field(None, description="Filter by restaurant ID")
    include_predictions: bool = Field(True, description="Include demand predictions")
    include_clustering: bool = Field(True, description="Include item clustering")
    prediction_days: int = Field(30, description="Days to predict ahead")
    n_clusters: Optional[int] = Field(None, description="Number of clusters (auto if None)")

    class Config:
        json_schema_extra = {
            "example": {
                "restaurant_id": None,
                "include_predictions": True,
                "include_clustering": True,
                "prediction_days": 30,
                "n_clusters": None
            }
        }


class DataOverview(BaseModel):
    """Overview of the dataset."""
    total_items: int
    total_restaurants: int
    total_orders: int
    total_campaigns: int
    date_range: Optional[Dict[str, str]] = None


class BCGBreakdown(BaseModel):
    """Breakdown by BCG category."""
    stars: int
    plowhorses: int
    puzzles: int
    dogs: int


class AnalysisResponse(BaseModel):
    """Full analysis response."""
    status: str = Field(..., description="Analysis status")
    timestamp: datetime = Field(default_factory=datetime.now)
    data_overview: DataOverview
    bcg_breakdown: BCGBreakdown
    recommendations: List[RecommendationResponse]
    pricing_suggestions: List[PricingSuggestion]
    clusters: Optional[List[ClusterInfo]] = None
    executive_summary: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2024-01-15T10:30:00",
                "data_overview": {
                    "total_items": 87713,
                    "total_restaurants": 1824,
                    "total_orders": 18600000,
                    "total_campaigns": 641
                },
                "bcg_breakdown": {
                    "stars": 5423,
                    "plowhorses": 12890,
                    "puzzles": 8234,
                    "dogs": 61166
                },
                "recommendations": [],
                "pricing_suggestions": []
            }
        }


class QuestionRequest(BaseModel):
    """Request model for business questions (LLM feature)."""
    question: str = Field(..., description="Business question in natural language")
    context: Optional[str] = Field(None, description="Additional context")
    restaurant_id: Optional[int] = Field(None, description="Filter by restaurant")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are my best performing menu items?",
                "context": "Looking at the last 3 months",
                "restaurant_id": None
            }
        }


class QuestionResponse(BaseModel):
    """Response model for business questions."""
    answer: str = Field(..., description="Natural language answer")
    confidence: float = Field(..., description="Confidence score 0-1")
    data_points: Optional[List[Dict[str, Any]]] = Field(
        None, description="Supporting data points"
    )
    suggestions: Optional[List[str]] = Field(
        None, description="Follow-up suggestions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Your top 5 performing items by revenue are...",
                "confidence": 0.92,
                "data_points": [{"item": "Pizza", "revenue": 150000}],
                "suggestions": ["Would you like pricing suggestions for these items?"]
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
