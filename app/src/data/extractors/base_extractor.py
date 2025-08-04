"""
Base extractor class for all data extractors.
"""
from abc import ABC, abstractmethod
from datetime import date
from typing import List, Dict, Any
import pandas as pd

from config.database import db_manager
from utils.logger import setup_logger


class BaseExtractor(ABC):
    """Abstract base class for data extractors."""
    
    def __init__(self):
        self.db = db_manager
        self.logger = setup_logger(self.__class__.__name__)
    
    @abstractmethod
    def extract(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Extract data for the given date range.
        
        Args:
            start_date: Start date for extraction
            end_date: End date for extraction
            
        Returns:
            DataFrame with extracted data
        """
        pass
    
    def _execute_query(self, query: str, params: List[Any]) -> List[Dict[str, Any]]:
        """Execute query with error handling and logging."""
        self.logger.debug(f"Executing query with params: {params}")
        try:
            results = self.db.execute_query(query, params)
            self.logger.info(f"Query returned {len(results)} rows")
            return results
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise
    
    def _to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert query results to pandas DataFrame."""
        if not results:
            self.logger.warning("No results to convert to DataFrame")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        self.logger.info(f"Created DataFrame with shape: {df.shape}")
        return df