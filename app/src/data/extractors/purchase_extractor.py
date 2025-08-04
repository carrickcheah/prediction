"""
Purchase order data extractor.
"""
from datetime import date
import pandas as pd

from .base_extractor import BaseExtractor


class PurchaseExtractor(BaseExtractor):
    """Extract purchase order data from database."""
    
    def extract(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Extract purchase order data for the given date range.
        
        Args:
            start_date: Start date for extraction
            end_date: End date for extraction
            
        Returns:
            DataFrame with columns: item_id, date, quantity, eta_date, supplier_id
        """
        query = """
        SELECT 
            pi.ItemId_i as item_id,
            DATE(pt.TxnDate_dd) as order_date,
            SUM(pi.Qty_d) as quantity,
            pi.EtaDate_dd as eta_date,
            pt.SuppId_i as supplier_id
        FROM tbl_porder_item pi
        JOIN tbl_porder_txn pt ON pi.TxnId_i = pt.TxnId_i
        WHERE 
            pt.TxnDate_dd BETWEEN %s AND %s
            AND pt.Void_c = '0'
            AND pi.Void_c = '0'
        GROUP BY pi.ItemId_i, DATE(pt.TxnDate_dd), pi.EtaDate_dd, pt.SuppId_i
        ORDER BY order_date, item_id
        """
        
        results = self._execute_query(query, [start_date, end_date])
        df = self._to_dataframe(results)
        
        if not df.empty:
            df['order_date'] = pd.to_datetime(df['order_date'])
            df['eta_date'] = pd.to_datetime(df['eta_date'])
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            
            # Calculate lead time
            df['lead_time_days'] = (df['eta_date'] - df['order_date']).dt.days
            
        return df