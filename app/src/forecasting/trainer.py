"""
Main training orchestrator for inventory forecasting.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from skforecast.ForecasterMultiSeries import ForecasterAutoregMultiSeries
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config.settings import get_settings
from utils.logger import setup_logger
from .features.calendar_features import create_calendar_features
from .validation.backtesting import validate_forecaster


class InventoryForecaster:
    """Main forecasting class that orchestrates training and prediction."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = setup_logger(self.__class__.__name__)
        self.forecaster = None
        self.training_metrics = None
        
    def train_and_predict(
        self, 
        data: pd.DataFrame,
        horizon: int = 14
    ) -> pd.DataFrame:
        """
        Train model and generate predictions.
        
        Args:
            data: Aggregated data with columns [date, item_id, total_demand]
            horizon: Forecast horizon in days
            
        Returns:
            DataFrame with predictions
        """
        self.logger.info(f"Training model for {len(data['item_id'].unique())} items")
        
        # Convert to multi-series format
        series_dict = self._prepare_series_data(data)
        
        # Create external features
        exog_features = self._create_external_features(data)
        
        # Initialize forecaster
        self._initialize_forecaster()
        
        # Validate before training
        if self.settings.USE_ENSEMBLE:
            metrics = validate_forecaster(self.forecaster, series_dict, steps=7)
            self.logger.info(f"Validation MAPE: {metrics['mape']:.2f}%")
        
        # Train model
        self.logger.info("Training forecaster")
        self.forecaster.fit(
            series=series_dict,
            exog=exog_features if not exog_features.empty else None
        )
        
        # Generate predictions
        self.logger.info(f"Generating {horizon}-day forecast")
        predictions = self.forecaster.predict(steps=horizon)
        
        # Format predictions
        pred_df = self._format_predictions(predictions)
        
        return pred_df
    
    def _prepare_series_data(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Convert aggregated data to series format for skforecast."""
        series_dict = {}
        
        for item_id in data['item_id'].unique():
            item_data = data[data['item_id'] == item_id].copy()
            item_data.set_index('date', inplace=True)
            
            series = pd.Series(
                data=item_data['total_demand'].values,
                index=item_data.index,
                name=str(item_id)
            )
            
            # Ensure daily frequency
            series = series.asfreq('D', fill_value=0)
            series_dict[str(item_id)] = series
            
        return series_dict
    
    def _create_external_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create external features for forecasting."""
        if data.empty:
            return pd.DataFrame()
            
        date_range = pd.date_range(
            start=data['date'].min(),
            end=data['date'].max() + pd.Timedelta(days=self.settings.FORECAST_HORIZON),
            freq='D'
        )
        
        return create_calendar_features(date_range)
    
    def _initialize_forecaster(self):
        """Initialize the multi-series forecaster."""
        self.forecaster = ForecasterAutoregMultiSeries(
            regressor=xgb.XGBRegressor(
                n_estimators=self.settings.XGB_N_ESTIMATORS,
                max_depth=self.settings.XGB_MAX_DEPTH,
                learning_rate=self.settings.XGB_LEARNING_RATE,
                random_state=self.settings.XGB_RANDOM_STATE,
                n_jobs=self.settings.N_JOBS
            ),
            lags=self.settings.MODEL_LAGS,
            encoding='ordinal',
            transformer_series=StandardScaler(),
            transformer_exog=StandardScaler() if self.settings.USE_ENSEMBLE else None
        )
    
    def _format_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Format predictions for reporting."""
        pred_df = predictions.reset_index()
        pred_df.columns = ['item_id', 'date', 'predicted_demand']
        
        # Add safety factor
        pred_df['safety_stock'] = pred_df['predicted_demand'] * self.settings.SAFETY_FACTOR
        
        # Round predictions
        pred_df['predicted_demand'] = pred_df['predicted_demand'].round(0)
        pred_df['safety_stock'] = pred_df['safety_stock'].round(0)
        
        return pred_df