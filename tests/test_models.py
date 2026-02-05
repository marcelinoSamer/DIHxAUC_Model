"""
File: test_models.py
Description: Unit tests for ML model classes.
Author: FlavorFlow Team

Run with: pytest tests/test_models.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.menu_classifier import MenuClassifier, MenuCategory
from models.item_clusterer import ItemClusterer
from models.demand_predictor import DemandPredictor


class TestMenuClassifier:
    """Tests for MenuClassifier class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'item_id': range(100),
            'order_count': np.random.randint(1, 1000, 100),
            'price': np.random.uniform(20, 200, 100),
            'revenue': np.random.uniform(1000, 50000, 100)
        })
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return MenuClassifier()
    
    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier is not None
        assert classifier.popularity_percentile == 50
        assert classifier.price_percentile == 50
    
    def test_custom_thresholds(self):
        """Test classifier with custom thresholds."""
        classifier = MenuClassifier(
            popularity_percentile=60,
            price_percentile=40
        )
        assert classifier.popularity_percentile == 60
        assert classifier.price_percentile == 40
    
    def test_fit_transform(self, classifier, sample_data):
        """Test fit_transform method."""
        result = classifier.fit_transform(sample_data)
        
        # Check that category column was added
        assert 'category' in result.columns
        
        # Check that all categories are valid
        valid_categories = ['⭐ Star', '🐴 Plowhorse', '❓ Puzzle', '🐕 Dog']
        assert all(cat in valid_categories for cat in result['category'].unique())
    
    def test_thresholds_computed(self, classifier, sample_data):
        """Test that thresholds are computed after fitting."""
        classifier.fit_transform(sample_data)
        
        assert classifier.thresholds is not None
        assert classifier.thresholds.popularity > 0
        assert classifier.thresholds.profitability > 0
    
    def test_get_recommendations(self, classifier, sample_data):
        """Test recommendations generation."""
        classified = classifier.fit_transform(sample_data)
        
        recommendations = classifier.get_recommendations(classified)
        
        assert len(recommendations) == 4  # One for each category
        assert all('recommendation' in rec for rec in recommendations)
        assert all('category' in rec for rec in recommendations)
    
    def test_metrics_computed(self, classifier, sample_data):
        """Test that metrics are computed."""
        classifier.fit_transform(sample_data)
        
        assert classifier.category_metrics is not None
        assert len(classifier.category_metrics) == 4  # One for each category


class TestItemClusterer:
    """Tests for ItemClusterer class."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        return pd.DataFrame({
            'price': np.random.uniform(20, 200, 100),
            'order_count': np.random.randint(1, 1000, 100),
            'revenue': np.random.uniform(1000, 50000, 100)
        })
    
    @pytest.fixture
    def clusterer(self):
        """Create clusterer instance."""
        return ItemClusterer(n_clusters=3, random_state=42)
    
    def test_initialization(self, clusterer):
        """Test clusterer initialization."""
        assert clusterer is not None
        assert clusterer.n_clusters == 3
        assert clusterer.random_state == 42
    
    def test_fit_transform(self, clusterer, sample_features):
        """Test fit_transform method."""
        result = clusterer.fit_transform(sample_features)
        
        # Result should be a DataFrame with cluster column
        assert isinstance(result, pd.DataFrame)
        assert 'cluster' in result.columns
        
        # Check labels are valid
        assert all(0 <= label < 3 for label in result['cluster'])
    
    def test_find_optimal_k(self, clusterer, sample_features):
        """Test optimal K finding."""
        optimal_k, inertias = clusterer.find_optimal_k(
            sample_features,
            k_range=range(2, 6)
        )
        
        # optimal_k could be int or list depending on implementation
        if isinstance(optimal_k, list):
            assert len(optimal_k) > 0
        else:
            assert 2 <= optimal_k <= 5
        assert len(inertias) == 4  # For k=2,3,4,5
    
    def test_scaling_applied(self, clusterer, sample_features):
        """Test that data is scaled."""
        clusterer.fit_transform(sample_features)
        
        assert clusterer.scaler is not None
        assert hasattr(clusterer.scaler, 'mean_')


class TestDemandPredictor:
    """Tests for DemandPredictor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data matching expected columns."""
        np.random.seed(42)
        n_samples = 200
        
        return pd.DataFrame({
            'price': np.random.uniform(20, 200, n_samples),
            'rating': np.random.uniform(3, 5, n_samples),
            'votes': np.random.randint(0, 100, n_samples),
            'purchases': np.random.randint(10, 1000, n_samples),
            'index': np.random.randint(0, 50, n_samples)
        })
    
    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return DemandPredictor()
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor is not None
        assert predictor.config is not None
    
    def test_prepare_features(self, predictor, sample_data):
        """Test feature preparation."""
        X, y = predictor.prepare_features(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert len(y) == len(sample_data)
    
    def test_fit_and_predict(self, predictor, sample_data):
        """Test fit and predict workflow."""
        X, y = predictor.prepare_features(sample_data)
        
        # Fit
        metrics = predictor.fit(X, y)
        
        assert metrics is not None
        
        # Predict
        predictions = predictor.predict(X[:10])
        
        assert len(predictions) == 10


class TestMenuCategoryEnum:
    """Tests for MenuCategory enum."""
    
    def test_category_values(self):
        """Test category enum values."""
        assert MenuCategory.STAR.value == "⭐ Star"
        assert MenuCategory.PLOWHORSE.value == "🐴 Plowhorse"
        assert MenuCategory.PUZZLE.value == "❓ Puzzle"
        assert MenuCategory.DOG.value == "🐕 Dog"
    
    def test_all_categories(self):
        """Test all categories exist."""
        categories = list(MenuCategory)
        assert len(categories) == 4


# =============================================================================
# Integration Tests
# =============================================================================

class TestModelsIntegration:
    """Integration tests for model classes."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic sample data."""
        np.random.seed(42)
        n_items = 500
        
        return pd.DataFrame({
            'item_id': range(n_items),
            'price': np.random.lognormal(4, 0.5, n_items),
            'order_count': np.random.exponential(100, n_items).astype(int),
            'revenue': np.random.lognormal(8, 1, n_items),
            'rating': np.random.uniform(3, 5, n_items),
            'category_id': np.random.randint(1, 20, n_items)
        })
    
    def test_classifier_and_clusterer_pipeline(self, realistic_data):
        """Test classifier and clusterer together."""
        # Step 1: Classify items
        classifier = MenuClassifier()
        classified = classifier.fit_transform(realistic_data)
        
        assert 'category' in classified.columns
        
        # Step 2: Cluster items
        clusterer = ItemClusterer(n_clusters=4)
        clustered = clusterer.fit_transform(
            realistic_data[['price', 'order_count', 'revenue']]
        )
        
        classified['cluster'] = clustered['cluster']
        
        # Step 3: Get recommendations
        recommendations = classifier.get_recommendations(classified)
        
        assert len(recommendations) > 0
        
        # Verify all steps completed
        assert 'category' in classified.columns
        assert 'cluster' in classified.columns
        assert len(recommendations) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
