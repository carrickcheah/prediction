#!/usr/bin/env python
"""
Build XGBoost forecasting models for all top 20 items.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from utils.logger import setup_logger

logger = setup_logger("xgboost_top20")


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
    # Calculate on original series before creating features
    df['rolling_mean_7'] = item_data.rolling(7, min_periods=1).mean()
    df['rolling_std_7'] = item_data.rolling(7, min_periods=1).std().fillna(0)
    df['rolling_mean_30'] = item_data.rolling(30, min_periods=1).mean()
    
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
    
    # Create and train forecaster with optimized parameters
    forecaster = ForecasterAutoreg(
        regressor=xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        lags=30  # Use last 30 days as features
    )
    
    # Split train/test
    train_size = len(y) - horizon
    
    # Train model
    print("  Training model...")
    forecaster.fit(y=y[:train_size], exog=exog[:train_size])
    
    # Make predictions
    predictions_test = forecaster.predict(steps=horizon, exog=exog[train_size:])
    
    # Calculate metrics
    actual_test = y[train_size:]
    mae = mean_absolute_error(actual_test, predictions_test)
    rmse = np.sqrt(mean_squared_error(actual_test, predictions_test))
    mape = np.mean(np.abs((actual_test - predictions_test) / (actual_test + 1e-8))) * 100
    
    # Get feature importance
    feature_importances = forecaster.regressor.feature_importances_
    feature_names = forecaster.regressor.get_booster().feature_names
    
    # Create importance dataframes
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Separate lag and exog features
    lag_importance = importance_df[importance_df['feature'].str.startswith('lag_')]
    feature_importance = importance_df[~importance_df['feature'].str.startswith('lag_')]
    
    results = {
        'forecaster': forecaster,
        'predictions': predictions_test,
        'actual': actual_test,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'feature_importance': feature_importance,
        'lag_importance': lag_importance,
        'train_size': train_size,
        'total_consumption': item_data.sum()
    }
    
    print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.1f}%")
    
    return results


def train_all_top20_models(data_path: Path, output_dir: Path):
    """Train XGBoost models for all top 20 items."""
    
    print("XGBOOST MODEL TRAINING - TOP 20 ITEMS")
    print("="*50)
    
    # Load consumption data
    print("Loading consumption data...")
    consumption_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Load top 20 items list
    top20_df = pd.read_csv("/Users/carrickcheah/Project/prediction/data/raw/top_20_items_365days.csv")
    top20_items = top20_df['item_id'].astype(str).tolist()
    
    # Train models for all top 20 items
    all_results = {}
    successful_items = []
    failed_items = []
    
    for idx, item_id in enumerate(top20_items, 1):
        print(f"\n[{idx}/20] Processing Item {item_id} (Annual consumption: {top20_df[top20_df['item_id']==int(item_id)]['consumption'].values[0]})")
        
        if item_id not in consumption_data.columns:
            print(f"  WARNING: Item {item_id} not found in consumption data")
            failed_items.append((item_id, "Not in data"))
            continue
            
        item_data = consumption_data[item_id]
        
        # Skip if insufficient data
        non_zero_days = (item_data > 0).sum()
        if non_zero_days < 30:
            print(f"  Skipping - insufficient data (only {non_zero_days} non-zero days)")
            failed_items.append((item_id, f"Only {non_zero_days} days"))
            continue
            
        try:
            results = train_xgboost_forecaster(item_data, item_id)
            all_results[item_id] = results
            successful_items.append(item_id)
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            failed_items.append((item_id, str(e)))
    
    # Save model performance summary
    save_model_summary(all_results, failed_items, output_dir)
    
    # Create visualizations for top performers
    create_top20_visualizations(all_results, consumption_data, output_dir)
    
    print(f"\n\nSUMMARY:")
    print(f"  Successfully trained: {len(successful_items)} models")
    print(f"  Failed: {len(failed_items)} items")
    
    if failed_items:
        print("\n  Failed items:")
        for item_id, reason in failed_items:
            print(f"    - {item_id}: {reason}")
    
    return all_results, successful_items, failed_items


def save_model_summary(results: Dict, failed_items: List, output_dir: Path):
    """Save comprehensive model performance summary."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create performance summary
    summary_data = []
    for item_id, result in results.items():
        summary_data.append({
            'item_id': item_id,
            'mae': result['mae'],
            'rmse': result['rmse'],
            'mape': result['mape'],
            'annual_consumption': result['total_consumption'],
            'avg_daily_consumption': result['total_consumption'] / 365
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('annual_consumption', ascending=False)
    
    # Add failed items
    if failed_items:
        failed_df = pd.DataFrame(failed_items, columns=['item_id', 'failure_reason'])
        failed_df['mae'] = np.nan
        failed_df['rmse'] = np.nan
        failed_df['mape'] = np.nan
    
    # Save to CSV
    summary_df.to_csv(output_dir / "xgboost_top20_performance.csv", index=False)
    
    # Create performance report
    report_lines = []
    report_lines.append("# XGBoost Top 20 Items Performance Report")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n## Summary Statistics")
    report_lines.append(f"- Models trained: {len(summary_df)}")
    report_lines.append(f"- Average MAE: {summary_df['mae'].mean():.3f}")
    report_lines.append(f"- Average MAPE: {summary_df['mape'].mean():.1f}%")
    report_lines.append(f"- Best MAE: {summary_df['mae'].min():.3f} (Item {summary_df.loc[summary_df['mae'].idxmin(), 'item_id']})")
    report_lines.append(f"- Worst MAE: {summary_df['mae'].max():.3f} (Item {summary_df.loc[summary_df['mae'].idxmax(), 'item_id']})")
    
    report_lines.append("\n## Performance by Item")
    report_lines.append("\n| Item ID | Annual Consumption | MAE | RMSE | MAPE |")
    report_lines.append("|---------|-------------------|-----|------|------|")
    
    for _, row in summary_df.iterrows():
        report_lines.append(f"| {row['item_id']} | {row['annual_consumption']:.0f} | "
                          f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['mape']:.1f}% |")
    
    if failed_items:
        report_lines.append("\n## Failed Items")
        for item_id, reason in failed_items:
            report_lines.append(f"- {item_id}: {reason}")
    
    with open(output_dir / "xgboost_top20_report.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nPerformance summary saved to {output_dir}")


def create_top20_visualizations(results: Dict, consumption_data: pd.DataFrame, output_dir: Path):
    """Create visualizations for top 20 model performance."""
    
    # Create performance overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. MAE vs Annual Consumption
    ax = axes[0, 0]
    mae_values = []
    consumption_values = []
    item_ids = []
    
    for item_id, result in results.items():
        mae_values.append(result['mae'])
        consumption_values.append(result['total_consumption'])
        item_ids.append(item_id)
    
    ax.scatter(consumption_values, mae_values, alpha=0.6)
    for i, item_id in enumerate(item_ids[:5]):  # Label top 5
        ax.annotate(item_id, (consumption_values[i], mae_values[i]), fontsize=8)
    
    ax.set_xlabel('Annual Consumption')
    ax.set_ylabel('MAE')
    ax.set_title('Forecast Error vs Item Volume')
    ax.grid(True, alpha=0.3)
    
    # 2. MAPE Distribution
    ax = axes[0, 1]
    mape_values = [result['mape'] for result in results.values()]
    ax.hist(mape_values, bins=15, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(mape_values), color='red', linestyle='--', 
               label=f'Mean: {np.mean(mape_values):.1f}%')
    ax.set_xlabel('MAPE (%)')
    ax.set_ylabel('Count')
    ax.set_title('MAPE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Top 10 Items by Volume - Performance
    ax = axes[1, 0]
    top10_by_volume = sorted(results.items(), 
                            key=lambda x: x[1]['total_consumption'], 
                            reverse=True)[:10]
    
    items = [x[0] for x in top10_by_volume]
    maes = [x[1]['mae'] for x in top10_by_volume]
    
    ax.bar(range(len(items)), maes, alpha=0.7)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(items, rotation=45)
    ax.set_xlabel('Item ID')
    ax.set_ylabel('MAE')
    ax.set_title('Forecast Error for Top 10 Items by Volume')
    ax.grid(True, alpha=0.3)
    
    # 4. Feature Importance Summary
    ax = axes[1, 1]
    
    # Aggregate feature importance across all models
    all_features = {}
    for result in results.values():
        for _, row in result['feature_importance'].iterrows():
            feature = row['feature']
            if feature not in all_features:
                all_features[feature] = []
            all_features[feature].append(row['importance'])
    
    # Calculate average importance
    avg_importance = {f: np.mean(v) for f, v in all_features.items()}
    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    features = [x[0] for x in top_features]
    importances = [x[1] for x in top_features]
    
    ax.barh(range(len(features)), importances, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Average Importance')
    ax.set_title('Top 10 Features Across All Models')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_top20_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for top 5 items
    top5_items = sorted(results.items(), 
                       key=lambda x: x[1]['total_consumption'], 
                       reverse=True)[:5]
    
    for item_id, result in top5_items:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Forecast plot
        actual = consumption_data[item_id]
        train_end = result['train_size']
        
        ax1.plot(actual.index[:train_end], actual[:train_end], 
                'b-', label='Training Data', alpha=0.7)
        ax1.plot(result['actual'].index, result['actual'], 
                'k-', label='Actual Test', linewidth=2)
        ax1.plot(result['predictions'].index, result['predictions'], 
                'r--', label=f'Forecast (MAE={result["mae"]:.2f})', linewidth=2)
        
        ax1.set_title(f'XGBoost Forecast for Item {item_id} (Annual: {result["total_consumption"]:.0f} units)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Consumption')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature importance
        top_features = result['feature_importance'].head(10)
        if not top_features.empty:
            ax2.barh(range(len(top_features)), top_features['importance'])
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features['feature'])
            ax2.set_xlabel('Importance')
            ax2.set_title('Top 10 Feature Importance')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'xgboost_item_{item_id}_detail.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to train XGBoost models for top 20 items."""
    
    # Paths
    data_path = Path("/Users/carrickcheah/Project/prediction/data/raw/top_20_consumption_pivot_365days.csv")
    output_dir = Path("/Users/carrickcheah/Project/prediction/outputs/xgboost_top20")
    
    # Train models
    results, successful, failed = train_all_top20_models(data_path, output_dir)
    
    print("\nTraining complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()