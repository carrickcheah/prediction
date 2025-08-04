"""
Data aggregator to combine data from multiple sources.
"""
import pandas as pd
from typing import Optional

from utils.logger import setup_logger


class DataAggregator:
    """Aggregate data from multiple sources for forecasting."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def aggregate(
        self,
        sales_data: pd.DataFrame,
        purchase_data: pd.DataFrame,
        consumption_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate data from all sources.
        
        Args:
            sales_data: Sales order data
            purchase_data: Purchase order data
            consumption_data: Job order consumption data
            
        Returns:
            Aggregated DataFrame ready for forecasting
        """
        self.logger.info("Starting data aggregation")
        
        # Create date range
        all_dates = pd.concat([
            sales_data['date'] if not sales_data.empty else pd.Series(dtype='datetime64[ns]'),
            consumption_data['date'] if not consumption_data.empty else pd.Series(dtype='datetime64[ns]')
        ]).unique()
        
        if len(all_dates) == 0:
            self.logger.warning("No data to aggregate")
            return pd.DataFrame()
        
        date_range = pd.date_range(
            start=min(all_dates),
            end=max(all_dates),
            freq='D'
        )
        
        # Get unique items
        all_items = set()
        if not sales_data.empty:
            all_items.update(sales_data['item_id'].unique())
        if not consumption_data.empty:
            all_items.update(consumption_data['item_id'].unique())
        if not purchase_data.empty:
            all_items.update(purchase_data['item_id'].unique())
        
        # Create base DataFrame
        base_df = pd.DataFrame(
            [(date, item) for date in date_range for item in all_items],
            columns=['date', 'item_id']
        )
        
        # Aggregate sales
        if not sales_data.empty:
            sales_agg = sales_data.groupby(['date', 'item_id'])['quantity'].sum().reset_index()
            sales_agg.rename(columns={'quantity': 'sales_qty'}, inplace=True)
            base_df = base_df.merge(sales_agg, on=['date', 'item_id'], how='left')
        else:
            base_df['sales_qty'] = 0
            
        # Aggregate consumption
        if not consumption_data.empty:
            consumption_agg = consumption_data.groupby(['date', 'item_id'])['consumption'].sum().reset_index()
            consumption_agg.rename(columns={'consumption': 'consumption_qty'}, inplace=True)
            base_df = base_df.merge(consumption_agg, on=['date', 'item_id'], how='left')
        else:
            base_df['consumption_qty'] = 0
            
        # Fill NaN values with 0
        base_df.fillna(0, inplace=True)
        
        # Calculate total demand
        base_df['total_demand'] = base_df['sales_qty'] + base_df['consumption_qty']
        
        self.logger.info(f"Aggregated data shape: {base_df.shape}")
        return base_df