"""
Feature engineering for intermittent demand patterns.
Specialized features for parts with high zero-demand periods.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger("intermittent_features")


class IntermittentDemandFeatures:
    """Generate features specifically for intermittent demand patterns."""
    
    def __init__(self):
        self.logger = logger
        
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        lags: List[int] = [1, 7, 14, 30],
        target_col: str = 'consumption'
    ) -> pd.DataFrame:
        """
        Create lag features for different time horizons.
        
        Args:
            df: DataFrame with time series data
            lags: List of lag periods to create
            target_col: Column name to create lags from
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
            
            # Also create binary indicator for demand existence
            df[f'had_demand_lag_{lag}'] = (df[f'lag_{lag}'] > 0).astype(int)
            
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [7, 14, 30],
        target_col: str = 'consumption'
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: DataFrame with time series data
            windows: List of window sizes
            target_col: Column to calculate statistics from
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        for window in windows:
            # Basic statistics
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window, min_periods=1).std()
            df[f'rolling_max_{window}'] = df[target_col].rolling(window, min_periods=1).max()
            df[f'rolling_min_{window}'] = df[target_col].rolling(window, min_periods=1).min()
            
            # Intermittent demand specific
            df[f'rolling_zero_count_{window}'] = (
                (df[target_col] == 0).rolling(window, min_periods=1).sum()
            )
            df[f'rolling_nonzero_count_{window}'] = (
                (df[target_col] > 0).rolling(window, min_periods=1).sum()
            )
            
            # Intermittency ratio
            df[f'intermittency_ratio_{window}'] = (
                df[f'rolling_zero_count_{window}'] / window
            )
            
        return df
    
    def create_intermittent_indicators(
        self,
        df: pd.DataFrame,
        target_col: str = 'consumption'
    ) -> pd.DataFrame:
        """
        Create features specific to intermittent demand patterns.
        
        Args:
            df: DataFrame with time series data
            target_col: Column with demand values
            
        Returns:
            DataFrame with intermittent demand features
        """
        df = df.copy()
        
        # Time since last demand
        df['time_since_last_demand'] = self._calculate_time_since_last_demand(df[target_col])
        
        # Time until next demand (useful for training)
        df['time_until_next_demand'] = self._calculate_time_until_next_demand(df[target_col])
        
        # Zero run length (consecutive zeros)
        df['zero_run_length'] = self._calculate_zero_run_length(df[target_col])
        
        # Demand run length (consecutive non-zeros)
        df['demand_run_length'] = self._calculate_demand_run_length(df[target_col])
        
        # Demand concentration (Gini coefficient style)
        df['demand_concentration_7d'] = df[target_col].rolling(7, min_periods=1).apply(
            self._calculate_concentration
        )
        df['demand_concentration_30d'] = df[target_col].rolling(30, min_periods=1).apply(
            self._calculate_concentration
        )
        
        # Average interval between demands
        df['avg_demand_interval_30d'] = df['time_since_last_demand'].rolling(30, min_periods=1).mean()
        
        # Demand probability (historical)
        df['demand_probability_7d'] = (df[target_col] > 0).rolling(7, min_periods=1).mean()
        df['demand_probability_30d'] = (df[target_col] > 0).rolling(30, min_periods=1).mean()
        
        return df
    
    def create_business_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on business day patterns.
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with business day features
        """
        df = df.copy()
        
        # Basic day features
        df['day_of_week'] = df.index.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Business day in month
        df['business_day_of_month'] = df.index.to_series().apply(
            lambda x: len(pd.bdate_range(x.replace(day=1), x))
        )
        
        # Days since/until weekend
        df['days_since_weekend'] = self._calculate_days_from_weekend(df.index, 'since')
        df['days_until_weekend'] = self._calculate_days_from_weekend(df.index, 'until')
        
        # Month and quarter features
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Cyclical encoding for day of week and month
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'consumption'
    ) -> pd.DataFrame:
        """
        Create all intermittent demand features.
        
        Args:
            df: DataFrame with time series data and DatetimeIndex
            target_col: Column with demand values
            
        Returns:
            DataFrame with all features added
        """
        self.logger.info(f"Creating intermittent demand features for {len(df)} records")
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        # Create all feature groups
        df = self.create_lag_features(df, target_col=target_col)
        df = self.create_rolling_features(df, target_col=target_col)
        df = self.create_intermittent_indicators(df, target_col=target_col)
        df = self.create_business_day_features(df)
        
        # Log feature creation summary
        n_features = len([col for col in df.columns if col != target_col])
        self.logger.info(f"Created {n_features} features")
        
        return df
    
    # Private helper methods
    def _calculate_time_since_last_demand(self, series: pd.Series) -> pd.Series:
        """Calculate days since last non-zero demand."""
        time_since = pd.Series(index=series.index, dtype=float)
        last_demand_idx = None
        
        for i, (idx, value) in enumerate(series.items()):
            if value > 0:
                last_demand_idx = i
                time_since.loc[idx] = 0
            elif last_demand_idx is not None:
                time_since.loc[idx] = i - last_demand_idx
            else:
                time_since.loc[idx] = np.nan  # No demand seen yet
                
        return time_since.fillna(method='ffill').fillna(0)
    
    def _calculate_time_until_next_demand(self, series: pd.Series) -> pd.Series:
        """Calculate days until next non-zero demand."""
        time_until = pd.Series(index=series.index, dtype=float)
        
        for i, (idx, value) in enumerate(series.items()):
            # Look forward for next demand
            future_demands = series.iloc[i+1:]
            next_demand_positions = future_demands[future_demands > 0].index
            
            if len(next_demand_positions) > 0:
                # Find position difference
                next_pos = series.index.get_loc(next_demand_positions[0])
                current_pos = i
                time_until.loc[idx] = next_pos - current_pos
            else:
                time_until.loc[idx] = np.nan  # No future demand
                
        return time_until.fillna(method='bfill').fillna(30)  # Default to 30 if no future demand
    
    def _calculate_zero_run_length(self, series: pd.Series) -> pd.Series:
        """Calculate current run length of consecutive zeros."""
        run_length = pd.Series(index=series.index, dtype=int)
        current_run = 0
        
        for idx, value in series.items():
            if value == 0:
                current_run += 1
            else:
                current_run = 0
            run_length.loc[idx] = current_run
            
        return run_length
    
    def _calculate_demand_run_length(self, series: pd.Series) -> pd.Series:
        """Calculate current run length of consecutive non-zero demands."""
        run_length = pd.Series(index=series.index, dtype=int)
        current_run = 0
        
        for idx, value in series.items():
            if value > 0:
                current_run += 1
            else:
                current_run = 0
            run_length.loc[idx] = current_run
            
        return run_length
    
    def _calculate_concentration(self, values: pd.Series) -> float:
        """Calculate concentration of demand (0=uniform, 1=concentrated)."""
        if len(values) == 0 or values.sum() == 0:
            return 0.0
            
        # Sort values and calculate cumulative proportion
        sorted_vals = np.sort(values)
        cumsum = np.cumsum(sorted_vals) / sorted_vals.sum()
        
        # Calculate Gini coefficient
        n = len(values)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
        
        return gini
    
    def _calculate_days_from_weekend(
        self, 
        dates: pd.DatetimeIndex, 
        direction: str = 'since'
    ) -> pd.Series:
        """Calculate days since or until weekend."""
        result = pd.Series(index=dates, dtype=int)
        
        for date in dates:
            dow = date.dayofweek
            
            if direction == 'since':
                if dow == 0:  # Monday
                    result.loc[date] = 1
                elif dow == 6:  # Sunday
                    result.loc[date] = 0
                elif dow == 5:  # Saturday
                    result.loc[date] = 0
                else:
                    result.loc[date] = dow + 1
            else:  # until
                if dow < 5:  # Weekday
                    result.loc[date] = 5 - dow
                else:  # Weekend
                    result.loc[date] = 0
                    
        return result


def create_features_for_part(
    df: pd.DataFrame,
    part_id: int,
    target_col: str = 'consumption'
) -> pd.DataFrame:
    """
    Convenience function to create all features for a single part.
    
    Args:
        df: DataFrame with consumption data
        part_id: ID of the part to process
        target_col: Column with demand values
        
    Returns:
        DataFrame with all features for the specified part
    """
    # Filter for specific part
    part_df = df[df['part_id'] == part_id].copy()
    
    # Ensure datetime index
    if 'date' in part_df.columns:
        part_df.set_index('date', inplace=True)
    
    # Create features
    feature_generator = IntermittentDemandFeatures()
    part_df = feature_generator.create_all_features(part_df, target_col=target_col)
    
    # Add part_id back
    part_df['part_id'] = part_id
    
    return part_df