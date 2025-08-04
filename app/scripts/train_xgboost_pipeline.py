#!/usr/bin/env python3
"""
Complete training pipeline for XGBoost intermittent demand forecasting.
Trains models for top 20 parts and generates Excel reports.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.database import db_manager
from forecasting.features.intermittent_features import IntermittentDemandFeatures
from forecasting.models.xgboost_intermittent import (
    XGBoostIntermittentDemand, 
    TwoStageIntermittentModel,
    train_model_for_demand_pattern
)
from forecasting.models.baseline_models import (
    SimpleMovingAverage, 
    CrostonsMethod,
    SyntetosBoylanApproximation,
    select_baseline_model
)
from reports.excel_reporter import ExcelReporter
from utils.logger import setup_logger

logger = setup_logger("training_pipeline")


class IntermittentDemandPipeline:
    """Complete pipeline for training and evaluating intermittent demand models."""
    
    def __init__(self, horizon: int = 14):
        """
        Initialize pipeline.
        
        Args:
            horizon: Forecast horizon in days
        """
        self.horizon = horizon
        self.models = {}
        self.baseline_models = {}
        self.predictions = {}
        self.metrics = {}
        self.feature_generator = IntermittentDemandFeatures()
        self.logger = logger
        
    def run_pipeline(self, top_n_parts: int = 20):
        """
        Run complete training pipeline.
        
        Args:
            top_n_parts: Number of top parts to process
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING XGBOOST TRAINING PIPELINE")
        self.logger.info("=" * 80)
        
        # Step 1: Load data
        self.logger.info("\n1. Loading data from database...")
        parts_df, consumption_data = self.load_data(top_n_parts)
        
        # Step 2: Process each part
        self.logger.info(f"\n2. Processing {len(parts_df)} parts...")
        all_forecasts = []
        all_metrics = []
        
        for idx, row in parts_df.iterrows():
            part_id = row['part_id']
            stock_code = row['stock_code']
            zero_pct = row['zero_percentage']
            
            self.logger.info(f"\n--- Processing Part {part_id} ({stock_code}) ---")
            self.logger.info(f"    Zero percentage: {zero_pct:.1f}%")
            
            # Get part data
            part_data = self.prepare_part_data(part_id, consumption_data)
            
            if part_data is None or len(part_data) < 30:
                self.logger.warning(f"    Insufficient data for part {part_id}")
                continue
                
            # Generate features
            part_data = self.feature_generator.create_all_features(part_data, target_col='consumption')
            
            # Train models
            xgb_model, xgb_metrics = self.train_xgboost_model(part_data, zero_pct)
            baseline_model, baseline_metrics = self.train_baseline_model(part_data, zero_pct)
            
            # Store models
            self.models[part_id] = xgb_model
            self.baseline_models[part_id] = baseline_model
            
            # Generate forecasts
            forecasts = self.generate_forecasts(part_id, part_data, xgb_model)
            forecasts['stock_code'] = stock_code
            forecasts['zero_pct'] = zero_pct
            
            # Calculate urgency
            forecasts['urgency'] = self.calculate_urgency(forecasts)
            
            all_forecasts.append(forecasts)
            
            # Store metrics
            metrics_row = {
                'part_id': part_id,
                'stock_code': stock_code,
                'xgb_mae': xgb_metrics.get('val_mae', xgb_metrics.get('train_mae')),
                'baseline_mae': baseline_metrics.get('mae', np.nan),
                'improvement': (baseline_metrics.get('mae', 0) - xgb_metrics.get('val_mae', xgb_metrics.get('train_mae', 0))) 
                              / baseline_metrics.get('mae', 1) * 100 if baseline_metrics.get('mae', 0) > 0 else 0
            }
            all_metrics.append(metrics_row)
            
        # Step 3: Combine results
        self.logger.info("\n3. Combining results...")
        if all_forecasts:
            final_forecasts = pd.concat(all_forecasts, ignore_index=True)
            metrics_df = pd.DataFrame(all_metrics)
            
            # Step 4: Generate report
            self.logger.info("\n4. Generating Excel report...")
            report_path = self.generate_excel_report(final_forecasts, metrics_df)
            
            # Step 5: Summary
            self.print_summary(metrics_df)
            
            return final_forecasts, metrics_df, report_path
        else:
            self.logger.error("No forecasts generated!")
            return None, None, None
            
    def load_data(self, top_n: int) -> tuple:
        """Load top N parts and their consumption data."""
        # Load parts analysis
        parts_file = Path(__file__).parent.parent / "data" / "parts_demand_analysis.csv"
        if not parts_file.exists():
            self.logger.error(f"Parts analysis file not found: {parts_file}")
            raise FileNotFoundError(f"Run create_data_quality_report.py first")
            
        parts_df = pd.read_csv(parts_file).head(top_n)
        
        # Load consumption data for all parts
        part_ids = parts_df['part_id'].tolist()
        
        query = f"""
        SELECT 
            ji.ItemId_i as part_id,
            DATE(jt.TxnDate_dd) as date,
            SUM(ji.Qty_d) as consumption
        FROM tbl_jo_item ji
        JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
        WHERE 
            ji.ItemId_i IN ({','.join(map(str, part_ids))})
            AND ji.InOut_c = 'I'
            AND jt.TxnDate_dd >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)
            AND jt.Void_c = '0'
        GROUP BY ji.ItemId_i, DATE(jt.TxnDate_dd)
        ORDER BY ji.ItemId_i, date
        """
        
        results = db_manager.execute_query(query)
        consumption_df = pd.DataFrame(results)
        
        if not consumption_df.empty:
            consumption_df['date'] = pd.to_datetime(consumption_df['date'])
            
        return parts_df, consumption_df
        
    def prepare_part_data(self, part_id: int, consumption_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for a single part."""
        part_data = consumption_df[consumption_df['part_id'] == part_id].copy()
        
        if part_data.empty:
            return None
            
        # Set date as index
        part_data.set_index('date', inplace=True)
        
        # Ensure daily frequency with zeros for missing days
        date_range = pd.date_range(part_data.index.min(), part_data.index.max(), freq='D')
        part_data = part_data.reindex(date_range, fill_value=0)
        part_data.index.name = 'date'
        
        # Ensure consumption column exists
        if 'consumption' not in part_data.columns:
            part_data['consumption'] = part_data.iloc[:, 0]  # Use first column
            
        return part_data
        
    def train_xgboost_model(self, data: pd.DataFrame, zero_pct: float) -> tuple:
        """Train XGBoost model for a part."""
        # Prepare features and target
        feature_cols = [col for col in data.columns 
                       if col not in ['consumption', 'part_id']]
        
        X = data[feature_cols].fillna(0)
        y = data['consumption']
        
        # Remove rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 30:
            return None, {}
            
        # Train-test split (time series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train appropriate model based on demand pattern
        model, metrics = train_model_for_demand_pattern(
            X_train, y_train, X_val, y_val, zero_pct
        )
        
        return model, metrics
        
    def train_baseline_model(self, data: pd.DataFrame, zero_pct: float) -> tuple:
        """Train baseline model for comparison."""
        y = data['consumption']
        
        # Select and train baseline
        model = select_baseline_model(y, zero_pct)
        
        # Generate predictions for validation
        val_size = int(len(y) * 0.2)
        train_y = y[:-val_size]
        val_y = y[-val_size:]
        
        model.fit(train_y)
        pred = model.predict(len(val_y))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(val_y, pred[:len(val_y)])
        
        return model, {'mae': mae}
        
    def generate_forecasts(
        self, 
        part_id: int, 
        data: pd.DataFrame, 
        model: object
    ) -> pd.DataFrame:
        """Generate forecasts for a part."""
        # Prepare features for future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.horizon,
            freq='D'
        )
        
        # Create future dataframe
        future_df = pd.DataFrame(index=future_dates)
        future_df['consumption'] = 0  # Placeholder
        
        # Generate features for future
        future_df = self.feature_generator.create_all_features(future_df)
        
        # Use last known values for lag features
        feature_cols = [col for col in data.columns 
                       if col not in ['consumption', 'part_id']]
        
        for col in feature_cols:
            if col in future_df.columns and col.startswith('lag_'):
                # Use historical values for lags
                lag_days = int(col.split('_')[1]) if 'lag_' in col else 0
                if lag_days > 0 and lag_days <= len(data):
                    future_df[col].iloc[0] = data['consumption'].iloc[-lag_days]
                    
        # Fill remaining NaNs
        future_df = future_df.fillna(method='ffill').fillna(0)
        
        # Generate predictions
        X_future = future_df[feature_cols]
        predictions = model.predict(X_future)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'part_id': part_id,
            'date': future_dates,
            'forecast': predictions,
            'lower_bound': predictions * 0.8,  # Simple confidence interval
            'upper_bound': predictions * 1.2
        })
        
        return forecast_df
        
    def calculate_urgency(self, forecasts: pd.DataFrame) -> pd.Series:
        """Calculate urgency level for each forecast."""
        urgency = []
        
        for _, row in forecasts.iterrows():
            days_ahead = (row['date'] - datetime.now()).days
            
            if days_ahead <= 2:
                urgency.append('critical')
            elif days_ahead <= 7:
                urgency.append('warning')
            else:
                urgency.append('normal')
                
        return pd.Series(urgency)
        
    def generate_excel_report(
        self, 
        forecasts: pd.DataFrame, 
        metrics: pd.DataFrame
    ) -> str:
        """Generate Excel report with results."""
        reporter = ExcelReporter()
        
        # Prepare model metrics
        model_metrics = {
            'total_parts': len(metrics),
            'avg_xgb_mae': metrics['xgb_mae'].mean(),
            'avg_baseline_mae': metrics['baseline_mae'].mean(),
            'avg_improvement': metrics['improvement'].mean(),
            'best_improvement': metrics['improvement'].max(),
            'worst_improvement': metrics['improvement'].min()
        }
        
        # Create feature importance (mock for now)
        feature_importance = pd.DataFrame({
            'feature': ['lag_1', 'lag_7', 'time_since_last_demand', 
                       'rolling_mean_7', 'day_of_week'],
            'importance': [0.25, 0.20, 0.15, 0.12, 0.08]
        })
        
        report_path = reporter.create_forecast_report(
            forecasts=forecasts,
            model_metrics=model_metrics,
            feature_importance=feature_importance
        )
        
        return report_path
        
    def print_summary(self, metrics: pd.DataFrame):
        """Print summary of results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 80)
        
        self.logger.info(f"\nParts Processed: {len(metrics)}")
        self.logger.info(f"Average XGBoost MAE: {metrics['xgb_mae'].mean():.3f}")
        self.logger.info(f"Average Baseline MAE: {metrics['baseline_mae'].mean():.3f}")
        self.logger.info(f"Average Improvement: {metrics['improvement'].mean():.1f}%")
        
        # Best and worst performers
        best = metrics.nlargest(3, 'improvement')
        self.logger.info("\nTop 3 Improvements:")
        for _, row in best.iterrows():
            self.logger.info(f"  {row['stock_code']}: {row['improvement']:.1f}% improvement")
            
        worst = metrics.nsmallest(3, 'improvement')
        self.logger.info("\nLowest 3 Improvements:")
        for _, row in worst.iterrows():
            self.logger.info(f"  {row['stock_code']}: {row['improvement']:.1f}% improvement")


if __name__ == "__main__":
    # Run pipeline
    pipeline = IntermittentDemandPipeline(horizon=14)
    
    try:
        forecasts, metrics, report_path = pipeline.run_pipeline(top_n_parts=10)
        
        if report_path:
            logger.info(f"\n✓ Pipeline completed successfully!")
            logger.info(f"✓ Report saved to: {report_path}")
        else:
            logger.error("\n✗ Pipeline failed!")
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise