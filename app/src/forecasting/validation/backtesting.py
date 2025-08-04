"""
Backtesting and validation for forecasting models.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from skforecast.model_selection import backtesting_forecaster

from utils.logger import setup_logger


logger = setup_logger("backtesting")


def validate_forecaster(
    forecaster,
    series: Dict[str, pd.Series],
    steps: int = 7,
    initial_train_size: Optional[int] = None,
    metric: str = 'mean_absolute_percentage_error'
) -> Dict[str, float]:
    """
    Validate forecaster using time series backtesting.
    
    Args:
        forecaster: Trained forecaster instance
        series: Dictionary of time series
        steps: Forecast horizon for validation
        initial_train_size: Initial training size (default: 80% of data)
        metric: Metric to use for evaluation
        
    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Starting backtesting with {steps}-step horizon")
    
    # Set initial train size if not provided
    if initial_train_size is None:
        min_length = min(len(s) for s in series.values())
        initial_train_size = int(0.8 * min_length)
    
    try:
        # Perform backtesting
        metrics_df = backtesting_forecaster(
            forecaster=forecaster,
            series=series,
            steps=steps,
            metric=metric,
            initial_train_size=initial_train_size,
            refit=True,
            n_jobs=-1,
            verbose=False
        )
        
        # Calculate aggregate metrics
        mape = metrics_df[metric].mean()
        
        metrics = {
            'mape': mape,
            'mape_std': metrics_df[metric].std(),
            'mape_min': metrics_df[metric].min(),
            'mape_max': metrics_df[metric].max(),
            'n_series': len(series)
        }
        
        logger.info(f"Validation complete - MAPE: {mape:.2f}%")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Backtesting failed: {str(e)}")
        return {
            'mape': 999,
            'error': str(e)
        }


def calculate_inventory_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    holding_cost: float = 1.0,
    stockout_cost: float = 3.0
) -> Dict[str, float]:
    """
    Calculate inventory-specific metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        holding_cost: Cost per unit of overstock
        stockout_cost: Cost per unit of stockout
        
    Returns:
        Dictionary with inventory metrics
    """
    errors = y_true - y_pred
    
    # Stockout occurs when prediction < actual (error > 0)
    stockouts = errors[errors > 0]
    overstock = -errors[errors < 0]
    
    metrics = {
        'stockout_rate': len(stockouts) / len(errors),
        'avg_stockout': stockouts.mean() if len(stockouts) > 0 else 0,
        'avg_overstock': overstock.mean() if len(overstock) > 0 else 0,
        'total_cost': (stockouts.sum() * stockout_cost + overstock.sum() * holding_cost),
        'service_level': 1 - (len(stockouts) / len(errors))
    }
    
    return metrics