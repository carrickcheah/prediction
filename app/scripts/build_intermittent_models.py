#!/usr/bin/env python
"""
Build forecasting models specifically for intermittent demand patterns.
Implements Croston's method, SBA, and TSB methods.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

logger = setup_logger("intermittent_models")


class IntermittentDemandForecaster:
    """Forecasting methods for intermittent demand."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize forecaster.
        
        Args:
            alpha: Smoothing parameter (0 < alpha < 1)
        """
        self.alpha = alpha
        
    def croston(self, demand: pd.Series, periods: int = 14) -> np.ndarray:
        """
        Croston's method for intermittent demand forecasting.
        
        Separately forecasts demand size and inter-arrival time.
        """
        demand_array = demand.values
        n = len(demand_array)
        
        # Initialize
        level_est = []
        interval_est = []
        
        # Find non-zero demands
        demand_periods = np.where(demand_array > 0)[0]
        
        if len(demand_periods) == 0:
            return np.zeros(periods)
        
        # Initialize estimates
        if len(demand_periods) > 0:
            z_est = demand_array[demand_periods[0]]
            p_est = demand_periods[0] + 1 if demand_periods[0] > 0 else 1
        else:
            z_est = 0
            p_est = n
            
        level_est.append(z_est)
        interval_est.append(p_est)
        
        # Update estimates only when demand occurs
        for i in range(1, len(demand_periods)):
            # Demand size update
            z_est = self.alpha * demand_array[demand_periods[i]] + (1 - self.alpha) * z_est
            
            # Interval update
            interval = demand_periods[i] - demand_periods[i-1]
            p_est = self.alpha * interval + (1 - self.alpha) * p_est
            
            level_est.append(z_est)
            interval_est.append(p_est)
        
        # Forecast
        if len(level_est) > 0 and len(interval_est) > 0:
            forecast = level_est[-1] / interval_est[-1] if interval_est[-1] > 0 else 0
        else:
            forecast = 0
            
        return np.full(periods, forecast)
    
    def sba(self, demand: pd.Series, periods: int = 14) -> np.ndarray:
        """
        Syntetos-Boylan Approximation (SBA) - bias-corrected Croston's method.
        """
        # Get Croston's forecast
        croston_forecast = self.croston(demand, periods)
        
        # Apply bias correction factor
        # SBA = Croston * (1 - alpha/2)
        bias_factor = 1 - self.alpha / 2
        
        return croston_forecast * bias_factor
    
    def tsb(self, demand: pd.Series, periods: int = 14, 
            target_service_level: float = 0.95) -> np.ndarray:
        """
        Teunter-Syntetos-Babai (TSB) method.
        Updates probability of demand occurrence.
        """
        demand_array = demand.values
        n = len(demand_array)
        
        # Initialize
        prob_est = 0.1  # Initial probability estimate
        size_est = 0
        
        # Find first non-zero demand
        first_demand_idx = np.argmax(demand_array > 0)
        if demand_array[first_demand_idx] > 0:
            size_est = demand_array[first_demand_idx]
            prob_est = 1 / (first_demand_idx + 1)
        
        # Update estimates for each period
        for t in range(n):
            if demand_array[t] > 0:
                # Update size estimate
                size_est = self.alpha * demand_array[t] + (1 - self.alpha) * size_est
                # Update probability estimate
                prob_est = self.alpha * 1 + (1 - self.alpha) * prob_est
            else:
                # Only update probability (no demand occurred)
                prob_est = self.alpha * 0 + (1 - self.alpha) * prob_est
        
        # Base forecast
        base_forecast = prob_est * size_est
        
        # Add safety stock based on service level
        # Approximate using normal distribution
        from scipy import stats
        z_score = stats.norm.ppf(target_service_level)
        
        # Estimate standard deviation
        demand_variance = np.var(demand_array[demand_array > 0]) if np.sum(demand_array > 0) > 1 else 0
        std_dev = np.sqrt(demand_variance * prob_est + size_est**2 * prob_est * (1 - prob_est))
        
        # Forecast with safety stock
        forecast_with_safety = base_forecast + z_score * std_dev * np.sqrt(periods)
        
        return np.full(periods, max(base_forecast, 0))
    
    def moving_average_nonzero(self, demand: pd.Series, window: int = 3, periods: int = 14) -> np.ndarray:
        """
        Moving average of non-zero demands only.
        """
        non_zero_demands = demand[demand > 0]
        
        if len(non_zero_demands) == 0:
            return np.zeros(periods)
        
        if len(non_zero_demands) < window:
            avg = non_zero_demands.mean()
        else:
            avg = non_zero_demands.tail(window).mean()
            
        # Adjust by frequency of demand
        demand_frequency = len(non_zero_demands) / len(demand)
        adjusted_forecast = avg * demand_frequency
        
        return np.full(periods, adjusted_forecast)


def evaluate_intermittent_methods(item_data: pd.Series, item_id: str, 
                                test_periods: int = 14) -> Dict:
    """Evaluate multiple intermittent demand methods."""
    
    # Split train/test
    train_data = item_data[:-test_periods]
    test_data = item_data[-test_periods:]
    
    # Initialize forecaster
    forecaster = IntermittentDemandForecaster(alpha=0.1)
    
    # Generate forecasts
    methods = {
        'Croston': forecaster.croston(train_data, test_periods),
        'SBA': forecaster.sba(train_data, test_periods),
        'TSB': forecaster.tsb(train_data, test_periods),
        'MA_NonZero': forecaster.moving_average_nonzero(train_data, window=3, periods=test_periods),
        'Naive_Avg': np.full(test_periods, train_data.mean())
    }
    
    # Calculate metrics
    results = {}
    for method_name, forecast in methods.items():
        mae = np.mean(np.abs(test_data.values - forecast))
        rmse = np.sqrt(np.mean((test_data.values - forecast)**2))
        
        # Calculate metrics relevant for intermittent demand
        # Periods with correct zero forecast
        correct_zeros = np.sum((test_data.values == 0) & (forecast == 0))
        # Periods with demand correctly identified (direction)
        correct_direction = np.sum((test_data.values > 0) & (forecast > 0))
        
        results[method_name] = {
            'forecast': forecast,
            'mae': mae,
            'rmse': rmse,
            'correct_zeros': correct_zeros,
            'correct_direction': correct_direction,
            'total_forecast': np.sum(forecast),
            'total_actual': np.sum(test_data.values)
        }
    
    return results, train_data, test_data


def run_intermittent_analysis():
    """Run analysis on all intermittent demand items."""
    
    # Load data
    data_path = Path("/Users/carrickcheah/Project/prediction/data/raw/top_20_consumption_pivot_365days.csv")
    consumption_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Load intermittency analysis
    analysis_df = pd.read_csv("/Users/carrickcheah/Project/prediction/outputs/intermittent_analysis/intermittency_analysis.csv")
    
    # Filter intermittent and lumpy items
    intermittent_items = analysis_df[
        analysis_df['demand_pattern'].isin(['Intermittent', 'Lumpy'])
    ]['item_id'].astype(str).tolist()
    
    # Evaluate methods for each item
    all_results = {}
    
    for item_id in intermittent_items[:10]:  # Start with top 10
        if item_id in consumption_data.columns:
            print(f"\nEvaluating Item {item_id}...")
            item_data = consumption_data[item_id]
            
            results, train_data, test_data = evaluate_intermittent_methods(item_data, item_id)
            all_results[item_id] = {
                'methods': results,
                'train_data': train_data,
                'test_data': test_data
            }
            
            # Print summary
            print(f"  Actual total in test period: {results['Croston']['total_actual']:.0f}")
            for method, res in results.items():
                print(f"  {method:12s}: MAE={res['mae']:.2f}, "
                      f"Total forecast={res['total_forecast']:.0f}")
    
    # Create visualizations
    create_intermittent_visualizations(all_results, consumption_data)
    
    # Save results summary
    save_intermittent_results(all_results)
    
    return all_results


def create_intermittent_visualizations(results: Dict, consumption_data: pd.DataFrame):
    """Create visualizations for intermittent demand methods."""
    
    output_dir = Path("/Users/carrickcheah/Project/prediction/outputs/intermittent_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot top 4 items
    items_to_plot = list(results.keys())[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, item_id in enumerate(items_to_plot):
        ax = axes[idx]
        
        item_results = results[item_id]
        train_data = item_results['train_data']
        test_data = item_results['test_data']
        
        # Plot historical data
        full_data = consumption_data[item_id]
        ax.plot(full_data.index[:-14], train_data.values, 'b-', alpha=0.7, label='Training')
        ax.plot(test_data.index, test_data.values, 'k-', linewidth=2, label='Actual')
        
        # Plot forecasts
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for (method, res), color in zip(item_results['methods'].items(), colors):
            forecast_series = pd.Series(res['forecast'], index=test_data.index)
            ax.plot(forecast_series.index, forecast_series.values, '--', 
                   color=color, alpha=0.7, label=f"{method} (MAE={res['mae']:.2f})")
        
        ax.set_title(f'Item {item_id} - Intermittent Demand Forecasts')
        ax.set_xlabel('Date')
        ax.set_ylabel('Consumption')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intermittent_forecasts_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create method comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(next(iter(results.values()))['methods'].keys())
    avg_mae = {method: [] for method in methods}
    
    for item_results in results.values():
        for method, res in item_results['methods'].items():
            avg_mae[method].append(res['mae'])
    
    # Calculate average MAE for each method
    avg_mae_values = [np.mean(avg_mae[method]) for method in methods]
    
    ax.bar(methods, avg_mae_values, alpha=0.7)
    ax.set_xlabel('Method')
    ax.set_ylabel('Average MAE')
    ax.set_title('Average Performance of Intermittent Demand Methods')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intermittent_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_intermittent_results(results: Dict):
    """Save detailed results of intermittent demand analysis."""
    
    output_dir = Path("/Users/carrickcheah/Project/prediction/outputs/intermittent_models")
    
    # Create summary DataFrame
    summary_data = []
    
    for item_id, item_results in results.items():
        for method, res in item_results['methods'].items():
            summary_data.append({
                'item_id': item_id,
                'method': method,
                'mae': res['mae'],
                'rmse': res['rmse'],
                'total_forecast': res['total_forecast'],
                'total_actual': res['total_actual'],
                'forecast_error_%': abs(res['total_forecast'] - res['total_actual']) / 
                                   (res['total_actual'] + 1e-8) * 100
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'intermittent_methods_results.csv', index=False)
    
    # Find best method for each item
    best_methods = summary_df.loc[summary_df.groupby('item_id')['mae'].idxmin()]
    
    print("\n\nBest method by item:")
    for _, row in best_methods.iterrows():
        print(f"  Item {row['item_id']}: {row['method']} (MAE={row['mae']:.2f})")
    
    # Overall best method
    avg_by_method = summary_df.groupby('method')['mae'].mean().sort_values()
    print(f"\n\nOverall best method: {avg_by_method.index[0]} (Avg MAE={avg_by_method.iloc[0]:.2f})")


if __name__ == "__main__":
    results = run_intermittent_analysis()
    print("\n\nIntermittent demand analysis complete!")