#!/usr/bin/env python
"""
Build XGBoost forecasting models for top items using skforecast.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from utils.logger import setup_logger
from forecasting.features.calendar_features import create_calendar_features

logger = setup_logger("xgboost_models")


def create_features_for_item(item_data: pd.Series) -> pd.DataFrame:
    """Create features for a single item time series."""
    
    # Create DataFrame
    df = pd.DataFrame({'consumption': item_data})
    
    # Add date features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    
    # Add cyclical encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Add rolling statistics (will be handled by skforecast lags)
    # These are calculated on the target variable
    
    return df


def train_xgboost_forecaster(item_data: pd.Series, item_id: str, horizon: int = 14) -> Dict:
    """Train XGBoost forecaster for a single item."""
    
    print(f"\nTraining XGBoost for Item {item_id}...")
    
    # Ensure the series has a frequency
    item_data = item_data.asfreq('D', fill_value=0)
    
    # Create features
    df = create_features_for_item(item_data)
    
    # Separate target and features
    y = df['consumption']
    exog = df.drop('consumption', axis=1)
    
    # Create and train forecaster
    forecaster = ForecasterAutoreg(
        regressor=xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        lags=30  # Use last 30 days as features
    )
    
    # Split train/test
    train_size = len(y) - horizon
    
    # Perform backtesting
    print("  Performing backtesting...")
    metrics = backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        exog=exog,
        steps=horizon,
        metric='mean_absolute_error',
        initial_train_size=train_size - horizon * 2,  # Leave some data for validation
        refit=True,
        verbose=False
    )
    
    # Train final model on all available data except test set
    forecaster.fit(y=y[:train_size], exog=exog[:train_size])
    
    # Make predictions
    predictions_test = forecaster.predict(steps=horizon, exog=exog[train_size:])
    
    # Calculate metrics
    actual_test = y[train_size:]
    mae = mean_absolute_error(actual_test, predictions_test)
    rmse = np.sqrt(mean_squared_error(actual_test, predictions_test))
    
    # Get feature importance
    # First 30 features are lags, rest are exog features
    all_features = forecaster.regressor.feature_names_in_
    feature_importances = forecaster.regressor.feature_importances_
    
    # Separate lag and exog features
    lag_features = [f for f in all_features if f.startswith('lag_')]
    exog_features = [f for f in all_features if not f.startswith('lag_')]
    
    # Get lag importance
    lag_importance = pd.DataFrame({
        'lag': lag_features,
        'importance': [feature_importances[list(all_features).index(f)] for f in lag_features]
    }).sort_values('importance', ascending=False)
    
    # Get exog feature importance
    feature_importance = pd.DataFrame({
        'feature': exog_features,
        'importance': [feature_importances[list(all_features).index(f)] for f in exog_features]
    }).sort_values('importance', ascending=False)
    
    results = {
        'forecaster': forecaster,
        'predictions': predictions_test,
        'actual': actual_test,
        'mae': mae,
        'rmse': rmse,
        'backtesting_mae': metrics['mean_absolute_error'].mean(),
        'feature_importance': feature_importance,
        'lag_importance': lag_importance,
        'train_size': train_size
    }
    
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Backtesting MAE: {results['backtesting_mae']:.3f}")
    
    return results


def compare_with_baselines(xgb_results: Dict, baseline_results_path: Path) -> pd.DataFrame:
    """Compare XGBoost results with baseline models."""
    
    # Load baseline results
    baseline_df = pd.read_csv(baseline_results_path)
    
    # Create comparison DataFrame
    comparison_data = []
    
    for item_id, xgb_data in xgb_results.items():
        # Get baseline results for this item
        item_baselines = baseline_df[baseline_df['item_id'] == item_id]
        
        if not item_baselines.empty:
            # Get best baseline
            best_baseline = item_baselines.loc[item_baselines['mae'].idxmin()]
            
            # Calculate improvement
            improvement = (best_baseline['mae'] - xgb_data['mae']) / best_baseline['mae'] * 100
            
            comparison_data.append({
                'item_id': item_id,
                'xgboost_mae': xgb_data['mae'],
                'best_baseline': best_baseline['model'],
                'baseline_mae': best_baseline['mae'],
                'improvement_%': improvement,
                'xgboost_wins': xgb_data['mae'] < best_baseline['mae']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def visualize_xgboost_results(xgb_results: Dict, consumption_data: pd.DataFrame, output_dir: Path):
    """Create visualizations for XGBoost model results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for item_id, results in xgb_results.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        ax = axes[0, 0]
        actual = consumption_data[item_id]
        train_end = results['train_size']
        
        ax.plot(actual.index[:train_end], actual[:train_end], 'b-', label='Training Data', alpha=0.7)
        ax.plot(results['actual'].index, results['actual'], 'k-', label='Actual Test', linewidth=2)
        ax.plot(results['predictions'].index, results['predictions'], 'r--', 
                label=f'XGBoost Forecast (MAE={results["mae"]:.2f})', linewidth=2)
        
        ax.set_title(f'XGBoost Forecast for Item {item_id}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Feature Importance
        ax = axes[0, 1]
        top_features = results['feature_importance'].head(10)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # 3. Lag Importance
        ax = axes[1, 0]
        top_lags = results['lag_importance'].head(10)
        ax.barh(range(len(top_lags)), top_lags['importance'])
        ax.set_yticks(range(len(top_lags)))
        ax.set_yticklabels(top_lags['lag'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Lag Importance')
        ax.grid(True, alpha=0.3)
        
        # 4. Residual Plot
        ax = axes[1, 1]
        residuals = results['actual'] - results['predictions']
        ax.scatter(results['predictions'], residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'xgboost_analysis_item_{item_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create comparison plot with baselines
    if Path("/Users/carrickcheah/Project/prediction/outputs/baseline_models/baseline_results.csv").exists():
        baseline_df = pd.read_csv("/Users/carrickcheah/Project/prediction/outputs/baseline_models/baseline_results.csv")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get unique items
        items = list(xgb_results.keys())
        x = np.arange(len(items))
        width = 0.15
        
        # Get baseline MAEs for each model
        models = baseline_df['model'].unique()
        
        for i, model in enumerate(models[:5]):  # Top 5 baseline models
            mae_values = []
            for item in items:
                item_mae = baseline_df[(baseline_df['item_id'] == item) & 
                                     (baseline_df['model'] == model)]['mae'].values
                mae_values.append(item_mae[0] if len(item_mae) > 0 else np.nan)
            
            ax.bar(x + i * width, mae_values, width, label=model, alpha=0.7)
        
        # Add XGBoost results
        xgb_maes = [xgb_results[item]['mae'] for item in items]
        ax.bar(x + (len(models[:5])) * width, xgb_maes, width, 
               label='XGBoost', color='red', alpha=0.8)
        
        ax.set_xlabel('Item ID')
        ax.set_ylabel('MAE')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(items)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'xgboost_vs_baselines_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to train XGBoost models."""
    
    # Paths
    data_path = Path("/Users/carrickcheah/Project/prediction/data/raw/top_20_consumption_pivot_365days.csv")
    baseline_results_path = Path("/Users/carrickcheah/Project/prediction/outputs/baseline_models/baseline_results.csv")
    output_dir = Path("/Users/carrickcheah/Project/prediction/outputs/xgboost_models")
    
    print("XGBOOST MODEL TRAINING")
    print("="*50)
    
    # Load consumption data
    print("Loading consumption data...")
    consumption_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Train XGBoost models for top items
    xgb_results = {}
    
    # Start with same top 5 items as baselines for comparison
    for item_id in consumption_data.columns[:5]:
        item_data = consumption_data[item_id]
        
        # Skip if insufficient data
        if len(item_data.dropna()) < 60:
            print(f"\nSkipping Item {item_id} - insufficient data")
            continue
            
        results = train_xgboost_forecaster(item_data, item_id)
        xgb_results[item_id] = results
    
    # Compare with baselines
    if baseline_results_path.exists():
        print("\n" + "="*50)
        print("COMPARISON WITH BASELINES")
        print("="*50)
        
        comparison_df = compare_with_baselines(xgb_results, baseline_results_path)
        print("\n", comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv(output_dir / "xgboost_vs_baseline_comparison.csv", index=False)
        
        # Calculate summary statistics
        print(f"\nSummary:")
        print(f"  XGBoost wins: {comparison_df['xgboost_wins'].sum()} out of {len(comparison_df)} items")
        print(f"  Average improvement: {comparison_df['improvement_%'].mean():.1f}%")
        print(f"  Best improvement: {comparison_df['improvement_%'].max():.1f}% (Item {comparison_df.loc[comparison_df['improvement_%'].idxmax(), 'item_id']})")
    
    # Visualize results
    print("\nCreating visualizations...")
    visualize_xgboost_results(xgb_results, consumption_data, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    print("\nNext steps:")
    print("1. Scale to all top 20 items")
    print("2. Implement multi-series model for efficiency")
    print("3. Add more sophisticated features")
    print("4. Tune hyperparameters for better performance")


if __name__ == "__main__":
    main()