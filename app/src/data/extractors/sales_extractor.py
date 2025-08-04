"""
Sales order data extractor.
"""
from datetime import date
import pandas as pd

from .base_extractor import BaseExtractor


class SalesExtractor(BaseExtractor):
    """Extract sales order data from database."""
    
    def extract(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Extract sales order data for the given date range.
        
        Args:
            start_date: Start date for extraction
            end_date: End date for extraction
            
        Returns:
            DataFrame with columns: item_id, date, quantity, customer_id
        """
        query = """
        SELECT 
            si.ItemId_i as item_id,
            DATE(st.TxnDate_dd) as date,
            SUM(si.Qty_d) as quantity,
            st.CustId_i as customer_id
        FROM tbl_sorder_item si
        JOIN tbl_sorder_txn st ON si.TxnId_i = st.TxnId_i
        WHERE 
            st.TxnDate_dd BETWEEN %s AND %s
            AND st.Void_c = '0'
            AND si.Void_c = '0'
        GROUP BY si.ItemId_i, DATE(st.TxnDate_dd), st.CustId_i
        ORDER BY date, item_id
        """
        
        results = self._execute_query(query, [start_date, end_date])
        df = self._to_dataframe(results)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            
        return df