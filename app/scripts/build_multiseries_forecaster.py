#!/usr/bin/env python
"""
Build multi-series forecaster for efficient scaling to 6000+ items.
Uses skforecast's ForecasterAutoregMultiSeries for shared learning across items.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from lightgbm import LGBMRegressor

from utils.logger import setup_logger
from data.extractors.job_order_extractor import JobOrderExtractor

logger = setup_logger("multiseries_forecaster")


class MultiSeriesForecasting:
    """Multi-series forecasting for all items."""
    
    def __init__(self, forecast_horizon: int = 14):
        self.forecast_horizon = forecast_horizon
        self.forecaster = None
        self.item_categories = {}
        
    def prepare_multiseries_data(self, days_back: int = 365) -> pd.DataFrame:
        """
        Extract and prepare data for multi-series forecasting.
        
        Returns:
            DataFrame with DatetimeIndex and columns for each item
        """
        print("Extracting consumption data for all items...")
        
        # Use job order extractor to get all parts
        extractor = JobOrderExtractor()
        series_dict = extractor.extract_all_parts_series(days_back=days_back)
        
        if not series_dict:
            raise ValueError("No data extracted")
            
        print(f"Extracted data for {len(series_dict)} items")
        
        # Convert to DataFrame format required by skforecast
        # All series must have the same index
        all_dates = pd.date_range(
            end=pd.Timestamp.now().date(),
            periods=days_back,
            freq='D'
        )
        
        # Create DataFrame with all items
        data_dict = {}
        for item_id, series in series_dict.items():
            # Reindex to ensure all dates are present
            series_reindexed = series.reindex(all_dates, fill_value=0)
            data_dict[item_id] = series_reindexed
            
        df = pd.DataFrame(data_dict)
        
        # Calculate basic statistics for categorization
        self._categorize_items(df)
        
        return df
    
    def _categorize_items(self, df: pd.DataFrame):
        """Categorize items by demand pattern."""
        
        categories = {
            'high_volume': [],
            'regular': [],
            'intermittent': [],
            'very_intermittent': []
        }
        
        for col in df.columns:
            series = df[col]
            total_demand = series.sum()
            demand_days = (series > 0).sum()
            demand_frequency = demand_days / len(series)
            
            if total_demand > 1000:
                categories['high_volume'].append(col)
            elif demand_frequency > 0.3:
                categories['regular'].append(col)
            elif demand_frequency > 0.1:
                categories['intermittent'].append(col)
            else:
                categories['very_intermittent'].append(col)
                
        self.item_categories = categories
        
        print("\nItem categorization:")
        for cat, items in categories.items():
            print(f"  {cat}: {len(items)} items")
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar features for the time series."""
        
        exog_df = pd.DataFrame(index=df.index)
        
        # Basic calendar features
        exog_df['day_of_week'] = df.index.dayofweek
        exog_df['day_of_month'] = df.index.day
        exog_df['week_of_year'] = df.index.isocalendar().week
        exog_df['month'] = df.index.month
        exog_df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        exog_df['day_sin'] = np.sin(2 * np.pi * exog_df['day_of_week'] / 7)
        exog_df['day_cos'] = np.cos(2 * np.pi * exog_df['day_of_week'] / 7)
        exog_df['month_sin'] = np.sin(2 * np.pi * exog_df['month'] / 12)
        exog_df['month_cos'] = np.cos(2 * np.pi * exog_df['month'] / 12)
        
        # Working days indicator (assuming weekends are off)
        exog_df['is_weekend'] = (exog_df['day_of_week'] >= 5).astype(int)
        
        return exog_df
    
    def train_multiseries_model(self, df: pd.DataFrame, 
                               exog: Optional[pd.DataFrame] = None,
                               use_subset: bool = True) -> Dict:
        """
        Train multi-series forecasting model.
        
        Args:
            df: DataFrame with time series for each item
            exog: Optional exogenous features
            use_subset: Whether to use subset for initial testing
        """
        
        if use_subset:
            # Use high volume and regular items for initial model
            subset_items = (self.item_categories['high_volume'] + 
                          self.item_categories['regular'])[:100]
            
            if not subset_items:
                # If no high volume/regular items, use top 100 by total demand
                total_demand = df.sum().sort_values(ascending=False)
                subset_items = total_demand.head(100).index.tolist()
                
            print(f"\nTraining on subset of {len(subset_items)} items...")
            df_train = df[subset_items]
        else:
            df_train = df
            print(f"\nTraining on all {len(df.columns)} items...")
        
        # Split train/test
        train_size = len(df_train) - self.forecast_horizon
        train_df = df_train.iloc[:train_size]
        
        # Initialize forecaster with LightGBM (faster than XGBoost for many series)
        self.forecaster = ForecasterAutoregMultiSeries(
            regressor=LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            lags=30,  # Use 30 days of lags
            encoding='ordinal',  # Encode series IDs
            dropna_from_series=True
        )
        
        print("Training multi-series model...")
        if exog is not None:
            exog_train = exog.iloc[:train_size]
            self.forecaster.fit(
                series=train_df,
                exog=exog_train
            )
        else:
            self.forecaster.fit(series=train_df)
        
        # Make predictions
        print("Generating forecasts...")
        if exog is not None:
            exog_test = exog.iloc[train_size:]
            predictions = self.forecaster.predict(
                steps=self.forecast_horizon,
                exog=exog_test
            )
        else:
            predictions = self.forecaster.predict(steps=self.forecast_horizon)
        
        # Calculate metrics
        actual_test = df_train.iloc[train_size:]
        metrics = self._calculate_metrics(predictions, actual_test)
        
        return {
            'predictions': predictions,
            'actual': actual_test,
            'metrics': metrics,
            'train_size': train_size,
            'n_series': len(df_train.columns)
        }
    
    def _calculate_metrics(self, predictions: pd.DataFrame, 
                          actual: pd.DataFrame) -> Dict:
        """Calculate metrics for multi-series predictions."""
        
        metrics = {}
        
        # Overall metrics
        mae_overall = mean_absolute_error(
            actual.values.flatten(), 
            predictions.values.flatten()
        )
        
        # Per-series metrics
        series_metrics = {}
        for col in predictions.columns:
            if col in actual.columns:
                mae = mean_absolute_error(actual[col], predictions[col])
                mape = np.mean(np.abs((actual[col] - predictions[col]) / 
                              (actual[col] + 1e-8))) * 100
                series_metrics[col] = {'mae': mae, 'mape': mape}
        
        metrics['overall_mae'] = mae_overall
        metrics['series_metrics'] = series_metrics
        metrics['avg_series_mae'] = np.mean([m['mae'] for m in series_metrics.values()])
        
        return metrics
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """Analyze feature importance from the trained model."""
        
        if self.forecaster is None:
            raise ValueError("Model not trained yet")
            
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.forecaster.regressor.feature_name_,
            'importance': self.forecaster.regressor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def forecast_specific_items(self, item_ids: List[str], 
                               steps: int = 14,
                               exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts for specific items."""
        
        if self.forecaster is None:
            raise ValueError("Model not trained yet")
            
        # Filter to requested items that are in the model
        available_items = [item for item in item_ids 
                          if item in self.forecaster.series_col_names]
        
        if not available_items:
            raise ValueError("None of the requested items are in the trained model")
            
        predictions = self.forecaster.predict(
            steps=steps,
            levels=available_items,
            exog=exog
        )
        
        return predictions


def create_multiseries_visualizations(results: Dict, output_dir: Path):
    """Create visualizations for multi-series results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall performance summary
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance by series
    ax = axes[0, 0]
    series_maes = [m['mae'] for m in results['metrics']['series_metrics'].values()]
    ax.hist(series_maes, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results['metrics']['avg_series_mae'], color='red', 
               linestyle='--', label=f"Mean: {results['metrics']['avg_series_mae']:.2f}")
    ax.set_xlabel('MAE')
    ax.set_ylabel('Number of Series')
    ax.set_title('Distribution of Forecast Errors Across Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top 10 best/worst performers
    ax = axes[0, 1]
    sorted_series = sorted(results['metrics']['series_metrics'].items(), 
                          key=lambda x: x[1]['mae'])
    
    top5 = sorted_series[:5]
    bottom5 = sorted_series[-5:]
    
    items = [x[0] for x in top5 + bottom5]
    maes = [x[1]['mae'] for x in top5 + bottom5]
    colors = ['green']*5 + ['red']*5
    
    ax.barh(range(len(items)), maes, color=colors, alpha=0.7)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items)
    ax.set_xlabel('MAE')
    ax.set_title('Best and Worst Performing Series')
    ax.grid(True, alpha=0.3)
    
    # Sample forecasts for top items
    ax = axes[1, 0]
    sample_items = list(results['predictions'].columns)[:3]
    
    for item in sample_items:
        if item in results['actual'].columns:
            ax.plot(results['actual'].index, results['actual'][item], 
                   '-', label=f'{item} Actual', alpha=0.7)
            ax.plot(results['predictions'].index, results['predictions'][item], 
                   '--', label=f'{item} Forecast')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Consumption')
    ax.set_title('Sample Forecasts vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Multi-Series Forecasting Summary
    
    Total Series: {results['n_series']}
    Forecast Horizon: {results['predictions'].shape[0]} days
    
    Overall MAE: {results['metrics']['overall_mae']:.3f}
    Average Series MAE: {results['metrics']['avg_series_mae']:.3f}
    
    Best Performer: {sorted_series[0][0]}
    MAE: {sorted_series[0][1]['mae']:.3f}
    
    Worst Performer: {sorted_series[-1][0]}
    MAE: {sorted_series[-1][1]['mae']:.3f}
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multiseries_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_multiseries_results(results: Dict, forecaster: MultiSeriesForecasting, 
                           output_dir: Path):
    """Save multi-series results and model artifacts."""
    
    # Save predictions
    results['predictions'].to_csv(output_dir / 'multiseries_predictions.csv')
    
    # Save metrics
    metrics_df = pd.DataFrame([
        {'item_id': k, **v} 
        for k, v in results['metrics']['series_metrics'].items()
    ])
    metrics_df.to_csv(output_dir / 'multiseries_metrics.csv', index=False)
    
    # Save feature importance
    if forecaster.forecaster is not None:
        importance = forecaster.analyze_feature_importance()
        importance.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance - Multi-Series Model')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate report
    report_lines = []
    report_lines.append("# Multi-Series Forecasting Results")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n## Model Configuration")
    report_lines.append(f"- Algorithm: LightGBM")
    report_lines.append(f"- Lags: 30 days")
    report_lines.append(f"- Series encoding: Ordinal")
    report_lines.append(f"- Number of series: {results['n_series']}")
    
    report_lines.append(f"\n## Performance Summary")
    report_lines.append(f"- Overall MAE: {results['metrics']['overall_mae']:.3f}")
    report_lines.append(f"- Average MAE across series: {results['metrics']['avg_series_mae']:.3f}")
    
    # Category performance
    if forecaster.item_categories:
        report_lines.append(f"\n## Performance by Category")
        for cat, items in forecaster.item_categories.items():
            cat_items = [item for item in items if item in results['metrics']['series_metrics']]
            if cat_items:
                cat_maes = [results['metrics']['series_metrics'][item]['mae'] for item in cat_items]
                report_lines.append(f"- {cat}: {np.mean(cat_maes):.3f} average MAE ({len(cat_items)} items)")
    
    report_lines.append(f"\n## Next Steps")
    report_lines.append("1. Scale to all 6000+ items")
    report_lines.append("2. Add exogenous features (holidays, promotions)")
    report_lines.append("3. Implement separate models for different demand patterns")
    report_lines.append("4. Set up automated retraining pipeline")
    
    with open(output_dir / 'multiseries_report.md', 'w') as f:
        f.write('\n'.join(report_lines))


def main():
    """Main function to run multi-series forecasting."""
    
    output_dir = Path("/Users/carrickcheah/Project/prediction/outputs/multiseries_forecast")
    
    print("MULTI-SERIES FORECASTING SYSTEM")
    print("="*50)
    
    # Initialize forecaster
    forecaster = MultiSeriesForecasting(forecast_horizon=14)
    
    # Prepare data
    try:
        df = forecaster.prepare_multiseries_data(days_back=365)
        print(f"\nData shape: {df.shape}")
        
        # Create calendar features
        exog = forecaster.create_calendar_features(df)
        
        # Train model on subset first
        results = forecaster.train_multiseries_model(df, exog=exog, use_subset=True)
        
        print(f"\nModel trained successfully!")
        print(f"Overall MAE: {results['metrics']['overall_mae']:.3f}")
        print(f"Average series MAE: {results['metrics']['avg_series_mae']:.3f}")
        
        # Create visualizations
        create_multiseries_visualizations(results, output_dir)
        
        # Save results
        save_multiseries_results(results, forecaster, output_dir)
        
        print(f"\nResults saved to: {output_dir}")
        
        # Example: Forecast specific items
        print("\nExample forecast for top 5 items:")
        top_items = df.sum().sort_values(ascending=False).head(5).index.tolist()
        # Need to provide future exog features
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
        future_exog = pd.DataFrame(index=future_dates)
        future_exog['day_of_week'] = future_exog.index.dayofweek
        future_exog['day_of_month'] = future_exog.index.day
        future_exog['week_of_year'] = future_exog.index.isocalendar().week
        future_exog['month'] = future_exog.index.month
        future_exog['quarter'] = future_exog.index.quarter
        future_exog['day_sin'] = np.sin(2 * np.pi * future_exog['day_of_week'] / 7)
        future_exog['day_cos'] = np.cos(2 * np.pi * future_exog['day_of_week'] / 7)
        future_exog['month_sin'] = np.sin(2 * np.pi * future_exog['month'] / 12)
        future_exog['month_cos'] = np.cos(2 * np.pi * future_exog['month'] / 12)
        future_exog['is_weekend'] = (future_exog['day_of_week'] >= 5).astype(int)
        
        specific_forecast = forecaster.forecast_specific_items(top_items, steps=7, exog=future_exog)
        print(specific_forecast.round(2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()