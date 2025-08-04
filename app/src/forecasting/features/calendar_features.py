"""
Calendar features for time series forecasting.
"""
import pandas as pd
import numpy as np


def create_calendar_features(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create calendar-based features for forecasting.
    
    Args:
        date_range: DatetimeIndex for which to create features
        
    Returns:
        DataFrame with calendar features
    """
    df = pd.DataFrame(index=date_range)
    
    # Basic calendar features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for periodic features
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Holiday features (customize for your country)
    holidays = pd.to_datetime([
        '2025-01-01',  # New Year
        '2025-01-28',  # Chinese New Year (example)
        '2025-01-29',  # Chinese New Year
        '2025-01-30',  # Chinese New Year
        '2025-05-01',  # Labour Day
        '2025-08-09',  # National Day
        '2025-12-25',  # Christmas
    ])
    
    df['is_holiday'] = df.index.isin(holidays).astype(int)
    
    # Days to/from holiday
    df['days_to_holiday'] = 999  # Large default value
    df['days_from_holiday'] = 999
    
    for holiday in holidays:
        days_diff = (df.index - holiday).days
        df.loc[days_diff >= 0, 'days_from_holiday'] = np.minimum(
            df.loc[days_diff >= 0, 'days_from_holiday'], 
            days_diff[days_diff >= 0]
        )
        df.loc[days_diff < 0, 'days_to_holiday'] = np.minimum(
            df.loc[days_diff < 0, 'days_to_holiday'], 
            -days_diff[days_diff < 0]
        )
    
    # Cap values at reasonable limits
    df['days_to_holiday'] = df['days_to_holiday'].clip(upper=30)
    df['days_from_holiday'] = df['days_from_holiday'].clip(upper=30)
    
    return df