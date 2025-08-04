#!/usr/bin/env python
"""
Build baseline forecasting models for comparison.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

logger = setup_logger("baseline_models")


class BaselineForecasters:
    """Collection of baseline forecasting methods."""
    
    def __init__(self):
        self.models = {}
        
    def moving_average(self, data: pd.Series, window: int = 7) -> pd.Series:
        """Simple moving average forecast."""
        return data.rolling(window=window, min_periods=1).mean()
    
    def weighted_moving_average(self, data: pd.Series, weights: list = None) -> pd.Series:
        """Weighted moving average with more weight on recent values."""
        if weights is None:
            # Default weights for 7-day WMA: [1, 2, 3, 4, 5, 6, 7]
            weights = list(range(1, 8))
        
        weights = np.array(weights) / sum(weights)
        wma = data.rolling(window=len(weights)).apply(lambda x: np.sum(x * weights))
        return wma
    
    def naive_forecast(self, data: pd.Series) -> pd.Series:
        """Last value carry forward (naive forecast)."""
        return data.shift(1)
    
    def seasonal_naive(self, data: pd.Series, season_length: int = 7) -> pd.Series:
        """Same value from season_length periods ago (e.g., same day last week)."""
        return data.shift(season_length)
    
    def average_same_weekday(self, data: pd.Series, weeks: int = 4) -> pd.Series:
        """Average of same weekday from last n weeks."""
        # Create a DataFrame with date index
        df = pd.DataFrame({'value': data.values}, index=data.index)
        df['weekday'] = df.index.weekday
        
        # Calculate average for each weekday
        result = pd.Series(index=data.index, dtype=float)
        
        for date in df.index[weeks*7:]:  # Start after we have enough history
            weekday = date.weekday()
            # Get same weekday from last n weeks
            same_weekdays = []
            for w in range(1, weeks + 1):
                prev_date = date - timedelta(days=7*w)
                if prev_date in df.index:
                    same_weekdays.append(df.loc[prev_date, 'value'])
            
            if same_weekdays:
                result[date] = np.mean(same_weekdays)
                
        return result
    
    def exponential_smoothing(self, data: pd.Series, alpha: float = 0.3) -> pd.Series:
        """Simple exponential smoothing."""
        result = pd.Series(index=data.index, dtype=float)
        result.iloc[0] = data.iloc[0]
        
        for i in range(1, len(data)):
            result.iloc[i] = alpha * data.iloc[i-1] + (1 - alpha) * result.iloc[i-1]
            
        return result


def evaluate_baseline_models(data_path: Path, horizon: int = 14):
    """Evaluate all baseline models on the top 20 items."""
    
    # Load consumption data
    print("Loading consumption data...")
    consumption_pivot = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Initialize baseline models
    baselines = BaselineForecasters()
    
    # Store results
    results = {}
    
    # For each item in top 20
    for item_id in consumption_pivot.columns[:5]:  # Start with top 5 for testing
        print(f"\nEvaluating baselines for Item {item_id}...")
        
        # Get item data
        item_data = consumption_pivot[item_id]
        
        # Split train/test (last 'horizon' days for test)
        train_data = item_data[:-horizon]
        test_data = item_data[-horizon:]
        
        # Skip if not enough data
        if len(train_data) < 30:
            print(f"  Skipping - insufficient data")
            continue
            
        item_results = {}
        
        # 1. Moving Averages
        for window in [7, 14, 30]:
            ma_forecast = baselines.moving_average(train_data, window)
            # Use last value for future forecast
            forecast = pd.Series([ma_forecast.iloc[-1]] * horizon, index=test_data.index)
            mae = np.mean(np.abs(test_data - forecast))
            item_results[f'MA_{window}'] = {
                'forecast': forecast,
                'mae': mae,
                'rmse': np.sqrt(np.mean((test_data - forecast)**2))
            }
        
        # 2. Weighted Moving Average
        wma_forecast = baselines.weighted_moving_average(train_data)
        forecast = pd.Series([wma_forecast.iloc[-1]] * horizon, index=test_data.index)
        mae = np.mean(np.abs(test_data - forecast))
        item_results['WMA_7'] = {
            'forecast': forecast,
            'mae': mae,
            'rmse': np.sqrt(np.mean((test_data - forecast)**2))
        }
        
        # 3. Naive Forecast
        naive_forecast = baselines.naive_forecast(train_data)
        forecast = pd.Series([train_data.iloc[-1]] * horizon, index=test_data.index)
        mae = np.mean(np.abs(test_data - forecast))
        item_results['Naive'] = {
            'forecast': forecast,
            'mae': mae,
            'rmse': np.sqrt(np.mean((test_data - forecast)**2))
        }
        
        # 4. Seasonal Naive (same day last week)
        seasonal_forecast = baselines.seasonal_naive(train_data, 7)
        # For forecast, use pattern from last week
        forecast_values = []
        for i in range(horizon):
            idx = -7 + i if (-7 + i) < 0 else -1
            forecast_values.append(train_data.iloc[idx])
        forecast = pd.Series(forecast_values, index=test_data.index)
        mae = np.mean(np.abs(test_data - forecast))
        item_results['Seasonal_7'] = {
            'forecast': forecast,
            'mae': mae,
            'rmse': np.sqrt(np.mean((test_data - forecast)**2))
        }
        
        # 5. Exponential Smoothing
        es_forecast = baselines.exponential_smoothing(train_data, alpha=0.3)
        forecast = pd.Series([es_forecast.iloc[-1]] * horizon, index=test_data.index)
        mae = np.mean(np.abs(test_data - forecast))
        item_results['ExpSmooth'] = {
            'forecast': forecast,
            'mae': mae,
            'rmse': np.sqrt(np.mean((test_data - forecast)**2))
        }
        
        results[item_id] = item_results
        
        # Print summary for this item
        print(f"\n  Baseline Model Performance (MAE):")
        for model_name, model_results in sorted(item_results.items(), key=lambda x: x[1]['mae']):
            print(f"    {model_name:12s}: {model_results['mae']:.3f}")
    
    return results, consumption_pivot


def visualize_baseline_results(results: Dict, consumption_data: pd.DataFrame, output_dir: Path):
    """Create visualizations of baseline model performance."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For each item, create a comparison plot
    for item_id, item_results in results.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Get actual data
        actual = consumption_data[item_id]
        train_end = len(actual) - 14
        
        # Plot 1: Time series with forecasts
        ax1.plot(actual.index[:train_end], actual[:train_end], 'b-', label='Training Data', alpha=0.7)
        ax1.plot(actual.index[train_end:], actual[train_end:], 'k-', label='Actual', linewidth=2)
        
        # Plot each model's forecast
        colors = plt.cm.Set3(np.linspace(0, 1, len(item_results)))
        for (model_name, model_data), color in zip(item_results.items(), colors):
            ax1.plot(model_data['forecast'].index, model_data['forecast'], 
                    '--', color=color, label=f"{model_name} (MAE={model_data['mae']:.2f})")
        
        ax1.set_title(f'Baseline Forecasts for Item {item_id}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Consumption')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error comparison
        model_names = list(item_results.keys())
        mae_values = [item_results[m]['mae'] for m in model_names]
        rmse_values = [item_results[m]['rmse'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax2.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
        ax2.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Error')
        ax2.set_title('Model Error Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'baseline_comparison_item_{item_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create summary comparison across all items
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate average MAE across items for each model
    model_avg_mae = {}
    for item_id, item_results in results.items():
        for model_name, model_data in item_results.items():
            if model_name not in model_avg_mae:
                model_avg_mae[model_name] = []
            model_avg_mae[model_name].append(model_data['mae'])
    
    # Calculate means and plot
    models = list(model_avg_mae.keys())
    means = [np.mean(model_avg_mae[m]) for m in models]
    stds = [np.std(model_avg_mae[m]) for m in models]
    
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Average MAE')
    ax.set_title('Average Model Performance Across Items')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def save_baseline_forecasts(results: Dict, output_path: Path):
    """Save baseline forecasts to CSV for later comparison."""
    
    # Create a summary DataFrame
    summary_data = []
    
    for item_id, item_results in results.items():
        for model_name, model_data in item_results.items():
            summary_data.append({
                'item_id': item_id,
                'model': model_name,
                'mae': model_data['mae'],
                'rmse': model_data['rmse']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"\nBaseline results saved to {output_path}")
    
    # Print overall summary
    print("\nOVERALL BASELINE PERFORMANCE:")
    print("="*50)
    best_models = summary_df.loc[summary_df.groupby('item_id')['mae'].idxmin()]
    print("\nBest model per item:")
    for _, row in best_models.iterrows():
        print(f"  Item {row['item_id']}: {row['model']} (MAE={row['mae']:.3f})")
    
    print("\nAverage MAE by model:")
    avg_mae = summary_df.groupby('model')['mae'].mean().sort_values()
    for model, mae in avg_mae.items():
        print(f"  {model:12s}: {mae:.3f}")


def main():
    """Main function to run baseline model evaluation."""
    
    # Paths
    data_path = Path("/Users/carrickcheah/Project/prediction/data/raw/top_20_consumption_pivot_365days.csv")
    output_dir = Path("/Users/carrickcheah/Project/prediction/outputs/baseline_models")
    
    print("BASELINE MODEL EVALUATION")
    print("="*50)
    
    # Evaluate baseline models
    results, consumption_data = evaluate_baseline_models(data_path)
    
    # Visualize results
    visualize_baseline_results(results, consumption_data, output_dir)
    
    # Save results
    save_baseline_forecasts(results, output_dir / "baseline_results.csv")
    
    print("\nBaseline evaluation complete!")
    print("\nNext steps:")
    print("1. Build XGBoost models for the same items")
    print("2. Compare XGBoost performance against these baselines")
    print("3. Identify which items benefit most from ML approaches")


if __name__ == "__main__":
    main()