"""
Report generator for inventory forecasting results.
"""
from datetime import datetime, date
from typing import List, Dict, Any
import pandas as pd
from dataclasses import dataclass

from config.settings import get_settings
from utils.logger import setup_logger


@dataclass
class ProcurementReport:
    """Container for procurement report data."""
    urgent_orders: List[Dict[str, Any]]
    warning_orders: List[Dict[str, Any]]
    summary_stats: Dict[str, Any]
    generated_at: datetime


class ReportGenerator:
    """Generate reports from forecasting results."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = setup_logger(self.__class__.__name__)
    
    def create_procurement_report(self, predictions: pd.DataFrame) -> ProcurementReport:
        """
        Create procurement report from predictions.
        
        Args:
            predictions: DataFrame with columns [item_id, date, predicted_demand, safety_stock]
            
        Returns:
            ProcurementReport object
        """
        self.logger.info("Creating procurement report")
        
        # Identify urgent orders (need within 2 days)
        urgent_date = pd.Timestamp.now() + pd.Timedelta(days=self.settings.URGENT_THRESHOLD_DAYS)
        urgent_orders = self._find_urgent_orders(predictions, urgent_date)
        
        # Identify warning orders (need within 7 days)
        warning_date = pd.Timestamp.now() + pd.Timedelta(days=self.settings.WARNING_THRESHOLD_DAYS)
        warning_orders = self._find_warning_orders(predictions, warning_date)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(predictions)
        
        report = ProcurementReport(
            urgent_orders=urgent_orders,
            warning_orders=warning_orders,
            summary_stats=summary_stats,
            generated_at=datetime.now()
        )
        
        self.logger.info(f"Report created with {len(urgent_orders)} urgent orders")
        
        return report
    
    def _find_urgent_orders(self, predictions: pd.DataFrame, urgent_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Find items that need urgent ordering."""
        urgent_df = predictions[predictions['date'] <= urgent_date].copy()
        
        # Group by item and sum demand
        urgent_summary = urgent_df.groupby('item_id').agg({
            'predicted_demand': 'sum',
            'safety_stock': 'max'
        }).reset_index()
        
        # Filter items with significant demand
        urgent_items = urgent_summary[urgent_summary['predicted_demand'] > 0]
        
        orders = []
        for _, row in urgent_items.iterrows():
            orders.append({
                'item_id': row['item_id'],
                'total_demand': row['predicted_demand'],
                'safety_stock': row['safety_stock'],
                'order_by_date': date.today(),
                'priority': 'URGENT'
            })
        
        return orders
    
    def _find_warning_orders(self, predictions: pd.DataFrame, warning_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Find items that need ordering soon."""
        warning_df = predictions[
            (predictions['date'] > pd.Timestamp.now() + pd.Timedelta(days=self.settings.URGENT_THRESHOLD_DAYS)) &
            (predictions['date'] <= warning_date)
        ].copy()
        
        # Group by item and sum demand
        warning_summary = warning_df.groupby('item_id').agg({
            'predicted_demand': 'sum',
            'safety_stock': 'max'
        }).reset_index()
        
        # Filter items with significant demand
        warning_items = warning_summary[warning_summary['predicted_demand'] > 0]
        
        orders = []
        for _, row in warning_items.iterrows():
            orders.append({
                'item_id': row['item_id'],
                'total_demand': row['predicted_demand'],
                'safety_stock': row['safety_stock'],
                'order_by_date': date.today() + pd.Timedelta(days=self.settings.WARNING_THRESHOLD_DAYS - 2),
                'priority': 'WARNING'
            })
        
        return orders
    
    def _calculate_summary_stats(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the report."""
        return {
            'total_items': predictions['item_id'].nunique(),
            'forecast_horizon': self.settings.FORECAST_HORIZON,
            'total_predicted_demand': predictions['predicted_demand'].sum(),
            'avg_daily_demand': predictions.groupby('date')['predicted_demand'].sum().mean(),
            'peak_demand_date': predictions.groupby('date')['predicted_demand'].sum().idxmax(),
            'items_with_demand': (predictions.groupby('item_id')['predicted_demand'].sum() > 0).sum()
        }