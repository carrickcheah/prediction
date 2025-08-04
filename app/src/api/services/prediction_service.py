"""
Prediction service for generating forecasts.
"""
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.forecasting.features.intermittent_features import IntermittentDemandFeatures
from src.forecasting.features.lead_time_features import LeadTimeFeatures
from src.api.models.schemas import UrgencyLevel, PredictionResponse, Alert
from src.config.database import db_manager
from src.utils.logger import setup_logger

logger = setup_logger("prediction_service")


class PredictionService:
    """Service for generating predictions."""
    
    def __init__(self, model_cache):
        """
        Initialize prediction service.
        
        Args:
            model_cache: ModelCache instance
        """
        self.model_cache = model_cache
        self.feature_generator = IntermittentDemandFeatures()
        self.lead_time_generator = LeadTimeFeatures()
        self.logger = logger
        
        # Load lead time statistics if available
        self.lead_time_stats = self._load_lead_time_stats()
        
    def _load_lead_time_stats(self) -> Optional[pd.DataFrame]:
        """Load lead time statistics."""
        lead_time_file = Path(__file__).parent.parent.parent.parent / "data" / "lead_time_statistics.csv"
        if lead_time_file.exists():
            return pd.read_csv(lead_time_file)
        return None
        
    def predict_single(
        self,
        part_id: int,
        horizon_days: int = 14,
        include_confidence: bool = False
    ) -> PredictionResponse:
        """
        Generate prediction for a single part.
        
        Args:
            part_id: Part ID to predict
            horizon_days: Forecast horizon
            include_confidence: Include confidence intervals
            
        Returns:
            PredictionResponse with predictions
        """
        # Check cache first
        cache_key = f"pred_{part_id}_{horizon_days}_{include_confidence}"
        cached = self.model_cache.get_cached_prediction(cache_key)
        if cached:
            self.logger.info(f"Returning cached prediction for part {part_id}")
            return cached
            
        # Get model
        model = self.model_cache.get_model(part_id)
        if not model:
            raise ValueError(f"No model found for part {part_id}")
            
        # Get metadata
        metadata = self.model_cache.get_model_metadata(part_id)
        
        # Get recent consumption data for features
        consumption_df = self._get_recent_consumption(part_id)
        
        # Generate features
        if consumption_df is not None and not consumption_df.empty:
            features_df = self._prepare_features(consumption_df, part_id)
            
            # Generate predictions
            predictions = self._generate_predictions(
                model, features_df, horizon_days
            )
        else:
            # No recent data, use zeros
            predictions = [0.0] * horizon_days
            
        # Generate dates
        start_date = date.today() + timedelta(days=1)
        dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        
        # Calculate confidence intervals (simple approach)
        confidence_lower = None
        confidence_upper = None
        if include_confidence:
            # Use +/- 20% as simple confidence interval
            confidence_lower = [max(0, p * 0.8) for p in predictions]
            confidence_upper = [p * 1.2 for p in predictions]
            
        # Determine urgency
        urgency = self._calculate_urgency(predictions, consumption_df)
        
        # Calculate recommended order
        rec_date, rec_qty = self._calculate_order_recommendation(
            predictions, metadata
        )
        
        response = PredictionResponse(
            part_id=part_id,
            stock_code=metadata.get('stock_code'),
            predictions=predictions,
            dates=dates,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            zero_percentage=metadata.get('zero_pct', 0),
            model_mae=metadata.get('mae'),
            urgency=urgency,
            recommended_order_date=rec_date,
            recommended_order_quantity=rec_qty
        )
        
        # Cache the response
        self.model_cache.cache_prediction(cache_key, response)
        
        return response
        
    def predict_batch(
        self,
        part_ids: List[int],
        horizon_days: int = 14
    ) -> List[PredictionResponse]:
        """Generate predictions for multiple parts."""
        predictions = []
        
        for part_id in part_ids:
            try:
                pred = self.predict_single(part_id, horizon_days, False)
                predictions.append(pred)
            except Exception as e:
                self.logger.error(f"Error predicting part {part_id}: {e}")
                # Create error response
                predictions.append(PredictionResponse(
                    part_id=part_id,
                    predictions=[],
                    dates=[],
                    zero_percentage=0,
                    urgency=UrgencyLevel.NORMAL
                ))
                
        return predictions
        
    def get_alerts(self) -> List[Alert]:
        """Get current inventory alerts."""
        alerts = []
        
        # Check all loaded models
        for part_id in self.model_cache.get_loaded_parts():
            try:
                # Get prediction
                pred = self.predict_single(part_id, horizon_days=14)
                
                # Check if urgent
                if pred.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]:
                    # Calculate days until stockout
                    days_until_stockout = None
                    for i, p in enumerate(pred.predictions):
                        if p > 0:  # Simplified - would need inventory levels
                            days_until_stockout = i + 1
                            break
                            
                    alert = Alert(
                        part_id=part_id,
                        stock_code=pred.stock_code,
                        urgency=pred.urgency,
                        message=f"Part {pred.stock_code or part_id} needs attention",
                        predicted_stockout_date=pred.recommended_order_date,
                        recommended_action=f"Order {pred.recommended_order_quantity:.0f} units by {pred.recommended_order_date}" if pred.recommended_order_quantity else "Review inventory",
                        current_consumption_rate=sum(pred.predictions) / len(pred.predictions) if pred.predictions else 0,
                        days_until_stockout=days_until_stockout
                    )
                    alerts.append(alert)
                    
            except Exception as e:
                self.logger.error(f"Error checking alerts for part {part_id}: {e}")
                
        # Sort by urgency
        alerts.sort(key=lambda x: (x.urgency == UrgencyLevel.CRITICAL, x.urgency == UrgencyLevel.HIGH), reverse=True)
        
        return alerts
        
    def _get_recent_consumption(self, part_id: int, days: int = 90) -> Optional[pd.DataFrame]:
        """Get recent consumption data for a part."""
        try:
            query = f"""
            SELECT 
                DATE(jt.TxnDate_dd) as date,
                SUM(ji.Qty_d) as consumption
            FROM tbl_jo_item ji
            JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
            WHERE 
                ji.ItemId_i = {part_id}
                AND ji.InOut_c = 'I'
                AND jt.TxnDate_dd >= DATE_SUB('2025-07-22', INTERVAL {days} DAY)
                AND jt.TxnDate_dd <= '2025-07-22'
                AND jt.Void_c = '0'
            GROUP BY DATE(jt.TxnDate_dd)
            ORDER BY date
            """
            
            results = db_manager.execute_query(query)
            
            if results:
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Fill missing dates
                date_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
                df = df.reindex(date_range, fill_value=0)
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting consumption data: {e}")
            
        return None
        
    def _prepare_features(self, consumption_df: pd.DataFrame, part_id: int) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Generate intermittent features
        features_df = self.feature_generator.create_all_features(
            consumption_df, target_col='consumption'
        )
        
        # Add lead time features if available
        if self.lead_time_stats is not None:
            features_df['item_id'] = part_id
            features_df = self.lead_time_generator.create_lead_time_features(
                features_df, self.lead_time_stats, item_col='item_id'
            )
            
        return features_df
        
    def _generate_predictions(
        self,
        model: Any,
        features_df: pd.DataFrame,
        horizon_days: int
    ) -> List[float]:
        """Generate predictions using model."""
        try:
            # Get feature columns
            feature_cols = [col for col in features_df.columns 
                          if col not in ['consumption', 'part_id', 'item_id']]
            
            # Use last row features as base (simplified)
            last_features = features_df[feature_cols].iloc[-1:].fillna(0)
            
            # Generate predictions for each day
            predictions = []
            for _ in range(horizon_days):
                pred = model.predict(last_features)[0]
                predictions.append(float(max(0, pred)))  # Ensure non-negative
                
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return [0.0] * horizon_days
            
    def _calculate_urgency(
        self,
        predictions: List[float],
        consumption_df: Optional[pd.DataFrame]
    ) -> UrgencyLevel:
        """Calculate urgency level based on predictions."""
        if not predictions or sum(predictions) == 0:
            return UrgencyLevel.NORMAL
            
        # Simple logic: if high consumption predicted in next 7 days
        next_7_days = sum(predictions[:7]) if len(predictions) >= 7 else sum(predictions)
        
        if consumption_df is not None and not consumption_df.empty:
            avg_consumption = consumption_df['consumption'].mean()
            
            if next_7_days > avg_consumption * 10:
                return UrgencyLevel.CRITICAL
            elif next_7_days > avg_consumption * 7:
                return UrgencyLevel.HIGH
            elif next_7_days > avg_consumption * 5:
                return UrgencyLevel.MEDIUM
                
        return UrgencyLevel.NORMAL
        
    def _calculate_order_recommendation(
        self,
        predictions: List[float],
        metadata: Dict
    ) -> Tuple[Optional[date], Optional[float]]:
        """Calculate recommended order date and quantity."""
        if not predictions or sum(predictions) == 0:
            return None, None
            
        # Simple logic: order for next 14 days consumption
        total_predicted = sum(predictions[:14]) if len(predictions) >= 14 else sum(predictions)
        
        # Add safety stock (20%)
        recommended_qty = total_predicted * 1.2
        
        # Order 7 days before needed (simple lead time)
        recommended_date = date.today() + timedelta(days=7)
        
        return recommended_date, recommended_qty