"""
File: test_api.py
Description: Unit tests for API endpoints.
Author: FlavorFlow Team

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "FlavorFlow" in data["name"]
        assert "version" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_items_without_init(self, client):
        """Test items endpoint fails without initialization."""
        response = client.get("/items")
        
        # Should fail because service not initialized
        assert response.status_code == 503
    
    def test_recommendations_without_init(self, client):
        """Test recommendations endpoint fails without initialization."""
        response = client.get("/recommendations")
        
        assert response.status_code == 503
    
    def test_pricing_suggestions_without_init(self, client):
        """Test pricing suggestions endpoint fails without initialization."""
        response = client.get("/pricing-suggestions")
        
        assert response.status_code == 503


class TestAPISchemas:
    """Tests for API Pydantic schemas."""
    
    def test_menu_item_response_schema(self):
        """Test MenuItemResponse schema."""
        from api.schemas import MenuItemResponse, MenuCategory
        
        item = MenuItemResponse(
            id=1,
            name="Test Item",
            price=99.0,
            category=MenuCategory.STAR,
            order_count=100,
            revenue=9900.0
        )
        
        assert item.id == 1
        assert item.category == MenuCategory.STAR
    
    def test_recommendation_response_schema(self):
        """Test RecommendationResponse schema."""
        from api.schemas import RecommendationResponse, MenuCategory
        
        rec = RecommendationResponse(
            category=MenuCategory.PUZZLE,
            action="Increase visibility",
            items_affected=50,
            priority="high"
        )
        
        assert rec.priority == "high"
        assert rec.items_affected == 50
    
    def test_pricing_suggestion_schema(self):
        """Test PricingSuggestion schema."""
        from api.schemas import PricingSuggestion
        
        suggestion = PricingSuggestion(
            item_id=1,
            current_price=50.0,
            suggested_price=60.0,
            price_change=10.0,
            price_change_pct=20.0,
            rationale="High demand"
        )
        
        assert suggestion.price_change == 10.0
        assert suggestion.price_change_pct == 20.0
    
    def test_analysis_request_defaults(self):
        """Test AnalysisRequest default values."""
        from api.schemas import AnalysisRequest
        
        request = AnalysisRequest()
        
        assert request.include_predictions is True
        assert request.include_clustering is True
        assert request.prediction_days == 30
    
    def test_health_response_schema(self):
        """Test HealthResponse schema."""
        from api.schemas import HealthResponse
        
        health = HealthResponse(status="healthy")
        
        assert health.status == "healthy"
        assert health.version == "1.0.0"
    
    def test_question_request_schema(self):
        """Test QuestionRequest schema."""
        from api.schemas import QuestionRequest
        
        question = QuestionRequest(
            question="What are my best items?"
        )
        
        assert "best" in question.question
    
    def test_question_response_schema(self):
        """Test QuestionResponse schema."""
        from api.schemas import QuestionResponse
        
        response = QuestionResponse(
            answer="Your best items are...",
            confidence=0.85
        )
        
        assert response.confidence == 0.85
        assert len(response.answer) > 0


class TestMenuCategory:
    """Tests for MenuCategory enum."""
    
    def test_category_enum_values(self):
        """Test all category enum values."""
        from api.schemas import MenuCategory
        
        assert MenuCategory.STAR.value == "star"
        assert MenuCategory.PLOWHORSE.value == "plowhorse"
        assert MenuCategory.PUZZLE.value == "puzzle"
        assert MenuCategory.DOG.value == "dog"
    
    def test_category_from_string(self):
        """Test creating category from string."""
        from api.schemas import MenuCategory
        
        cat = MenuCategory("star")
        assert cat == MenuCategory.STAR


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
