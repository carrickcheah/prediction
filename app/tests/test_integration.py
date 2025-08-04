"""
Integration tests for the inventory forecasting system.
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd

from src.config.settings import get_settings
from src.data.extractors.job_order_extractor import JobOrderExtractor
from src.data.processors.data_aggregator import DataAggregator
from src.forecasting.trainer import InventoryForecaster


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_full_pipeline(self):
        """Test the complete forecasting pipeline."""
        # This is a placeholder for integration tests
        # In a real implementation, you would:
        # 1. Set up test database or mock data
        # 2. Run the full pipeline
        # 3. Verify outputs
        pass
    
    def test_database_connection(self):
        """Test database connectivity."""
        from src.config.database import db_manager
        
        # Test connection
        with db_manager.get_connection() as conn:
            assert conn.is_connected()
    
    def test_settings_loading(self):
        """Test settings are loaded correctly."""
        settings = get_settings()
        assert settings.MARIADB_DATABASE == "nex_valiant"
        assert settings.FORECAST_HORIZON == 14