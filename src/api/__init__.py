"""
API Package
===========

FastAPI endpoints for the FlavorFlow Craft Menu Engineering solution.

Modules:
    main: FastAPI application and endpoint definitions
    schemas: Pydantic models for request/response validation
    dependencies: Shared dependencies and middleware
"""

from .main import app, get_analysis_service
from .schemas import (
    MenuItemResponse,
    RecommendationResponse,
    PricingSuggestion,
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse
)

__all__ = [
    'app',
    'get_analysis_service',
    'MenuItemResponse',
    'RecommendationResponse',
    'PricingSuggestion',
    'AnalysisRequest',
    'AnalysisResponse',
    'HealthResponse'
]
