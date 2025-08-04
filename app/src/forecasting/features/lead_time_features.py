"""
Lead time feature engineering from purchase order history.
Extracts supplier performance metrics and lead time patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from utils.logger import setup_logger

logger = setup_logger("lead_time_features")


@dataclass
class LeadTimeStats:
    """Statistics for lead times."""
    mean: float
    median: float
    std: float
    min: float
    max: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    reliability_score: float  # Percentage of on-time deliveries
    variability_score: float  # Coefficient of variation


class LeadTimeFeatures:
    """
    Extract and engineer lead time features from purchase order history.
    """
    
    def __init__(self, lookback_days: int = 365):
        """
        Initialize lead time feature generator.
        
        Args:
            lookback_days: Number of days to look back for historical data
        """
        self.lookback_days = lookback_days
        self.lead_time_cache = {}
        self.supplier_performance = {}
        self.logger = logger
        
    def extract_lead_times_from_purchases(
        self,
        purchase_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract actual lead times from purchase order data.
        
        Args:
            purchase_df: DataFrame with columns:
                - item_id: Part/item identifier
                - order_date: Date order was placed
                - eta_date: Expected arrival date
                - actual_receive_date: Actual receipt date (if available)
                - supplier_id: Supplier identifier
                - quantity: Order quantity
                
        Returns:
            DataFrame with calculated lead times
        """
        self.logger.info(f"Extracting lead times from {len(purchase_df)} purchase orders")
        
        df = purchase_df.copy()
        
        # Calculate promised lead time (order to ETA)
        if 'eta_date' in df.columns and 'order_date' in df.columns:
            df['promised_lead_days'] = (
                pd.to_datetime(df['eta_date']) - pd.to_datetime(df['order_date'])
            ).dt.days
        else:
            df['promised_lead_days'] = np.nan
            
        # Calculate actual lead time if receipt date available
        if 'actual_receive_date' in df.columns:
            df['actual_lead_days'] = (
                pd.to_datetime(df['actual_receive_date']) - pd.to_datetime(df['order_date'])
            ).dt.days
            
            # Calculate delay (positive = late, negative = early)
            df['delivery_delay'] = df['actual_lead_days'] - df['promised_lead_days']
            
            # On-time delivery flag (within +/- 2 days)
            df['on_time'] = df['delivery_delay'].abs() <= 2
        else:
            # If no actual receipt date, use ETA as estimate
            df['actual_lead_days'] = df['promised_lead_days']
            df['delivery_delay'] = 0
            df['on_time'] = True
            
        # Remove invalid lead times
        df = df[df['promised_lead_days'] > 0]
        df = df[df['promised_lead_days'] < 365]  # Remove unrealistic lead times
        
        self.logger.info(f"Extracted lead times for {len(df)} valid orders")
        
        return df
        
    def calculate_lead_time_stats(
        self,
        lead_time_df: pd.DataFrame,
        group_by: List[str] = ['item_id', 'supplier_id']
    ) -> pd.DataFrame:
        """
        Calculate lead time statistics by grouping.
        
        Args:
            lead_time_df: DataFrame with lead time data
            group_by: Columns to group by
            
        Returns:
            DataFrame with lead time statistics
        """
        self.logger.info(f"Calculating lead time statistics grouped by {group_by}")
        
        stats_list = []
        
        for group_vals, group_df in lead_time_df.groupby(group_by):
            if len(group_df) < 2:
                continue  # Need at least 2 orders for statistics
                
            lead_times = group_df['actual_lead_days'].dropna()
            
            if len(lead_times) == 0:
                continue
                
            stats = LeadTimeStats(
                mean=lead_times.mean(),
                median=lead_times.median(),
                std=lead_times.std(),
                min=lead_times.min(),
                max=lead_times.max(),
                percentile_25=lead_times.quantile(0.25),
                percentile_75=lead_times.quantile(0.75),
                percentile_95=lead_times.quantile(0.95),
                reliability_score=group_df['on_time'].mean() if 'on_time' in group_df else 1.0,
                variability_score=lead_times.std() / lead_times.mean() if lead_times.mean() > 0 else 0
            )
            
            # Create result dict
            result = {col: val for col, val in zip(group_by, group_vals)}
            result.update({
                'lead_time_mean': stats.mean,
                'lead_time_median': stats.median,
                'lead_time_std': stats.std,
                'lead_time_min': stats.min,
                'lead_time_max': stats.max,
                'lead_time_p25': stats.percentile_25,
                'lead_time_p75': stats.percentile_75,
                'lead_time_p95': stats.percentile_95,
                'supplier_reliability': stats.reliability_score,
                'lead_time_cv': stats.variability_score,
                'order_count': len(group_df)
            })
            
            stats_list.append(result)
            
        stats_df = pd.DataFrame(stats_list)
        self.logger.info(f"Calculated statistics for {len(stats_df)} item-supplier combinations")
        
        return stats_df
        
    def create_lead_time_features(
        self,
        demand_df: pd.DataFrame,
        lead_time_stats: pd.DataFrame,
        item_col: str = 'item_id'
    ) -> pd.DataFrame:
        """
        Add lead time features to demand forecast data.
        
        Args:
            demand_df: DataFrame with demand/consumption data
            lead_time_stats: DataFrame with lead time statistics
            item_col: Name of item/part column
            
        Returns:
            DataFrame with added lead time features
        """
        self.logger.info("Creating lead time features for demand data")
        
        df = demand_df.copy()
        
        # Get unique lead time stats per item (across all suppliers)
        item_stats = lead_time_stats.groupby(item_col).agg({
            'lead_time_mean': 'mean',
            'lead_time_median': 'median',
            'lead_time_std': 'mean',
            'lead_time_min': 'min',
            'lead_time_max': 'max',
            'lead_time_p95': 'max',
            'supplier_reliability': 'mean',
            'lead_time_cv': 'mean',
            'order_count': 'sum'
        }).reset_index()
        
        # Rename columns to avoid confusion
        item_stats.columns = [item_col] + [f'{col}_all_suppliers' for col in item_stats.columns[1:]]
        
        # Merge with demand data
        df = df.merge(item_stats, on=item_col, how='left')
        
        # Fill missing values with defaults
        default_lead_time = 14  # Default 2 weeks if no history
        
        for col in item_stats.columns[1:]:
            if col.endswith('_all_suppliers'):
                if 'mean' in col or 'median' in col or 'max' in col or 'p95' in col:
                    df[col] = df[col].fillna(default_lead_time)
                elif 'min' in col:
                    df[col] = df[col].fillna(7)  # Minimum 1 week
                elif 'reliability' in col:
                    df[col] = df[col].fillna(0.8)  # 80% default reliability
                elif 'count' in col:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(0)
                    
        # Create derived features
        df['lead_time_buffer_days'] = df['lead_time_p95_all_suppliers'] - df['lead_time_mean_all_suppliers']
        df['lead_time_range'] = df['lead_time_max_all_suppliers'] - df['lead_time_min_all_suppliers']
        df['has_purchase_history'] = (df['order_count_all_suppliers'] > 0).astype(int)
        
        # Calculate reorder point indicator (simplified)
        if 'consumption' in df.columns:
            # Days of supply needed = lead time + safety buffer
            df['reorder_horizon'] = df['lead_time_p95_all_suppliers'] + 3  # 3 days safety
            
        self.logger.info(f"Added {len([c for c in df.columns if 'lead_time' in c or 'supplier' in c])} lead time features")
        
        return df
        
    def calculate_dynamic_lead_times(
        self,
        lead_time_df: pd.DataFrame,
        reference_date: datetime,
        windows: List[int] = [30, 60, 90, 180]
    ) -> pd.DataFrame:
        """
        Calculate rolling lead time statistics over different time windows.
        
        Args:
            lead_time_df: DataFrame with lead time history
            reference_date: Date to calculate backwards from
            windows: List of window sizes in days
            
        Returns:
            DataFrame with dynamic lead time features
        """
        self.logger.info(f"Calculating dynamic lead times for windows: {windows}")
        
        df = lead_time_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        results = []
        
        for item_id in df['item_id'].unique():
            item_df = df[df['item_id'] == item_id].sort_values('order_date')
            
            result = {'item_id': item_id}
            
            for window in windows:
                window_start = reference_date - timedelta(days=window)
                window_data = item_df[item_df['order_date'] >= window_start]
                
                if len(window_data) > 0:
                    lead_times = window_data['actual_lead_days'].dropna()
                    
                    if len(lead_times) > 0:
                        result[f'lead_time_mean_{window}d'] = lead_times.mean()
                        result[f'lead_time_std_{window}d'] = lead_times.std()
                        result[f'lead_time_trend_{window}d'] = self._calculate_trend(
                            window_data[['order_date', 'actual_lead_days']].dropna()
                        )
                        result[f'order_count_{window}d'] = len(window_data)
                    else:
                        result[f'lead_time_mean_{window}d'] = np.nan
                        result[f'lead_time_std_{window}d'] = np.nan
                        result[f'lead_time_trend_{window}d'] = 0
                        result[f'order_count_{window}d'] = 0
                else:
                    result[f'lead_time_mean_{window}d'] = np.nan
                    result[f'lead_time_std_{window}d'] = np.nan
                    result[f'lead_time_trend_{window}d'] = 0
                    result[f'order_count_{window}d'] = 0
                    
            results.append(result)
            
        dynamic_df = pd.DataFrame(results)
        self.logger.info(f"Calculated dynamic lead times for {len(dynamic_df)} items")
        
        return dynamic_df
        
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """
        Calculate trend in lead times over time.
        
        Args:
            df: DataFrame with order_date and actual_lead_days
            
        Returns:
            Trend coefficient (positive = increasing, negative = decreasing)
        """
        if len(df) < 2:
            return 0.0
            
        # Convert dates to numeric (days since first order)
        df = df.copy()
        df['days_since_start'] = (df['order_date'] - df['order_date'].min()).dt.days
        
        # Simple linear regression
        x = df['days_since_start'].values
        y = df['actual_lead_days'].values
        
        if len(x) > 1 and np.std(x) > 0:
            # Calculate trend using numpy polyfit
            trend = np.polyfit(x, y, 1)[0]  # Slope of linear fit
            return trend
        else:
            return 0.0
            
    def create_procurement_urgency_features(
        self,
        df: pd.DataFrame,
        current_inventory: Optional[Dict[int, float]] = None,
        safety_stock_days: int = 7
    ) -> pd.DataFrame:
        """
        Create features indicating procurement urgency based on lead times.
        
        Args:
            df: DataFrame with consumption and lead time features
            current_inventory: Optional current inventory levels by item
            safety_stock_days: Days of safety stock to maintain
            
        Returns:
            DataFrame with urgency features
        """
        self.logger.info("Creating procurement urgency features")
        
        df = df.copy()
        
        # Calculate days until stockout (simplified)
        if 'consumption' in df.columns and 'lead_time_p95_all_suppliers' in df.columns:
            # Average daily consumption
            daily_consumption = df.groupby('item_id')['consumption'].mean()
            
            # Required lead time for ordering
            df['required_order_lead_days'] = df['lead_time_p95_all_suppliers'] + safety_stock_days
            
            # Urgency levels
            df['order_urgency'] = pd.cut(
                df['required_order_lead_days'],
                bins=[0, 7, 14, 30, np.inf],
                labels=['immediate', 'urgent', 'normal', 'low']
            )
            
            # Binary urgency flags
            df['needs_immediate_order'] = (df['required_order_lead_days'] <= 7).astype(int)
            df['needs_urgent_order'] = (df['required_order_lead_days'] <= 14).astype(int)
            
        return df
        
    def get_supplier_performance_metrics(
        self,
        lead_time_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate supplier performance metrics.
        
        Args:
            lead_time_df: DataFrame with lead time and supplier data
            
        Returns:
            DataFrame with supplier performance metrics
        """
        self.logger.info("Calculating supplier performance metrics")
        
        supplier_stats = lead_time_df.groupby('supplier_id').agg({
            'actual_lead_days': ['mean', 'std', 'min', 'max'],
            'delivery_delay': ['mean', 'std'],
            'on_time': 'mean',
            'order_date': 'count'
        }).reset_index()
        
        # Flatten column names
        supplier_stats.columns = ['supplier_id', 
                                 'avg_lead_time', 'lead_time_std', 'min_lead_time', 'max_lead_time',
                                 'avg_delay', 'delay_std', 'on_time_rate', 'total_orders']
        
        # Calculate performance score (0-100)
        supplier_stats['performance_score'] = (
            supplier_stats['on_time_rate'] * 50 +  # 50% weight on reliability
            (1 - supplier_stats['lead_time_std'] / supplier_stats['avg_lead_time'].clip(lower=1)) * 30 +  # 30% on consistency
            (1 - supplier_stats['avg_delay'].clip(lower=0) / supplier_stats['avg_lead_time'].clip(lower=1)) * 20  # 20% on delays
        ).clip(0, 100)
        
        # Rank suppliers
        supplier_stats['supplier_rank'] = supplier_stats['performance_score'].rank(ascending=False)
        
        self.logger.info(f"Calculated performance for {len(supplier_stats)} suppliers")
        
        return supplier_stats