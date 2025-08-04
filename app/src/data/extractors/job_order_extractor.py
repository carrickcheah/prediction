"""
Job order (manufacturing consumption) data extractor.
"""
from datetime import date, timedelta
from typing import Dict
import pandas as pd

from .base_extractor import BaseExtractor


class JobOrderExtractor(BaseExtractor):
    """Extract job order consumption data from database."""
    
    def extract(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Extract job order consumption data for the given date range.
        
        Args:
            start_date: Start date for extraction
            end_date: End date for extraction
            
        Returns:
            DataFrame with columns: item_id, date, consumption, job_id
        """
        query = """
        SELECT 
            ji.ItemId_i as item_id,
            DATE(jt.TxnDate_dd) as date,
            SUM(ji.Qty_d) as consumption,
            jt.TxnId_i as job_id
        FROM tbl_jo_item ji
        JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
        WHERE 
            ji.InOut_c = 'I'  -- Input/consumption only
            AND jt.TxnDate_dd BETWEEN %s AND %s
            AND jt.Void_c = '0'
            AND ji.Void_c = '0'
        GROUP BY ji.ItemId_i, DATE(jt.TxnDate_dd), jt.TxnId_i
        ORDER BY date, item_id
        """
        
        results = self._execute_query(query, [start_date, end_date])
        df = self._to_dataframe(results)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce').fillna(0)
            
        return df
    
    def extract_all_parts_series(self, days_back: int = 365) -> Dict[str, pd.Series]:
        """
        Extract consumption data in format required by skforecast.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dictionary of {part_id: pd.Series} for multi-series forecasting
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        query = """
        SELECT 
            ji.ItemId_i as part_id,
            DATE(jt.TxnDate_dd) as date,
            SUM(ji.Qty_d) as consumption
        FROM tbl_jo_item ji
        JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
        WHERE 
            ji.InOut_c = 'I'
            AND jt.TxnDate_dd >= %s
            AND jt.Void_c = '0'
            AND ji.Void_c = '0'
        GROUP BY ji.ItemId_i, DATE(jt.TxnDate_dd)
        """
        
        results = self._execute_query(query, [start_date])
        df = self._to_dataframe(results)
        
        if df.empty:
            return {}
        
        # Convert to skforecast format
        series_dict = {}
        for part_id in df['part_id'].unique():
            part_data = df[df['part_id'] == part_id].copy()
            series = pd.Series(
                data=part_data['consumption'].values,
                index=pd.DatetimeIndex(part_data['date']),
                name=str(part_id)
            )
            # Fill missing dates with 0
            series = series.asfreq('D', fill_value=0)
            series_dict[str(part_id)] = series
            
        self.logger.info(f"Extracted series for {len(series_dict)} parts")
        return series_dict