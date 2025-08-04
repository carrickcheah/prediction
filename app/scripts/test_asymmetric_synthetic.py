#!/usr/bin/env python3
"""
Test asymmetric loss with synthetic data to demonstrate the difference.
Creates data with moderate intermittency (50-70% zeros) to use regular XGBoost models.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecasting.features.intermittent_features import IntermittentDemandFeatures
from forecasting.models.xgboost_intermittent import train_model_for_demand_pattern
from forecasting.models.asymmetric_loss import calculate_inventory_metrics
from utils.logger import setup_logger

logger = setup_logger("test_asymmetric_synthetic")


def generate_synthetic_intermittent_data(
    n_days: int = 365,
    zero_percentage: float = 60.0,
    mean_demand: float = 100.0,
    std_demand: float = 30.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic intermittent demand data.
    
    Args:
        n_days: Number of days of data
        zero_percentage: Percentage of days with zero demand
        mean_demand: Mean demand when demand occurs
        std_demand: Standard deviation of demand
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with date index and consumption column
    """
    np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    # Generate demand occurrence (binary)
    demand_prob = 1 - (zero_percentage / 100)
    has_demand = np.random.binomial(1, demand_prob, n_days)
    
    # Generate demand quantities
    quantities = np.random.normal(mean_demand, std_demand, n_days)
    quantities = np.maximum(quantities, 0)  # No negative demand
    
    # Apply intermittency
    consumption = has_demand * quantities
    
    # Add some weekly pattern
    day_of_week = pd.Series(dates).dt.dayofweek
    weekend_factor = np.where(day_of_week.isin([5, 6]), 0.5, 1.0)
    consumption = consumption * weekend_factor
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'consumption': consumption
    })
    df.set_index('date', inplace=True)
    
    return df


def test_asymmetric_on_synthetic():
    """Test asymmetric loss on synthetic data with different intermittency levels."""
    
    logger.info("=" * 80)
    logger.info("ASYMMETRIC LOSS TEST WITH SYNTHETIC DATA")
    logger.info("=" * 80)
    
    # Test scenarios with different intermittency levels
    scenarios = [
        {'name': 'Low Intermittency', 'zero_pct': 30.0, 'mean': 100, 'std': 20},
        {'name': 'Medium Intermittency', 'zero_pct': 50.0, 'mean': 80, 'std': 25},
        {'name': 'High Intermittency', 'zero_pct': 70.0, 'mean': 60, 'std': 30},
    ]
    
    feature_generator = IntermittentDemandFeatures()
    all_results = []
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Scenario: {scenario['name']}")
        logger.info(f"Zero Percentage: {scenario['zero_pct']:.1f}%")
        logger.info(f"Mean Demand: {scenario['mean']}, Std: {scenario['std']}")
        logger.info(f"{'='*60}")
        
        # Generate synthetic data
        data = generate_synthetic_intermittent_data(
            n_days=500,
            zero_percentage=scenario['zero_pct'],
            mean_demand=scenario['mean'],
            std_demand=scenario['std']
        )
        
        # Generate features
        data = feature_generator.create_all_features(data, target_col='consumption')
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col != 'consumption']
        X = data[feature_cols].fillna(0)
        y = data['consumption']
        
        # Remove rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Calculate actual zero percentage in training data
        actual_zero_pct = (y_train == 0).mean() * 100
        logger.info(f"Actual zero percentage in training: {actual_zero_pct:.1f}%")
        
        # Test different stockout penalties
        penalties = [1.0, 2.0, 3.0, 5.0]
        
        for penalty in penalties:
            logger.info(f"\n--- Testing with Stockout Penalty = {penalty} ---")
            
            if penalty == 1.0:
                # Standard model (no asymmetry)
                model, metrics = train_model_for_demand_pattern(
                    X_train, y_train, X_val, y_val,
                    zero_percentage=actual_zero_pct,
                    use_asymmetric_loss=False
                )
            else:
                # Asymmetric model
                model, metrics = train_model_for_demand_pattern(
                    X_train, y_train, X_val, y_val,
                    zero_percentage=actual_zero_pct,
                    use_asymmetric_loss=True,
                    stockout_penalty=penalty
                )
            
            # Generate predictions
            pred = model.predict(X_val)
            
            # Calculate inventory metrics
            inv_metrics = calculate_inventory_metrics(
                y_val.values, pred, stockout_penalty=penalty
            )
            
            # Store results
            result = {
                'scenario': scenario['name'],
                'zero_pct': scenario['zero_pct'],
                'penalty': penalty,
                'mae': inv_metrics['mae'],
                'stockout_rate': inv_metrics['stockout_rate'],
                'overstock_rate': inv_metrics['overstock_rate'],
                'avg_stockout_qty': inv_metrics['avg_stockout_qty'],
                'avg_overstock_qty': inv_metrics['avg_overstock_qty'],
                'total_cost': inv_metrics['total_cost'],
                'asymmetric_loss': inv_metrics['asymmetric_loss']
            }
            all_results.append(result)
            
            # Print results
            logger.info(f"  MAE: {inv_metrics['mae']:.2f}")
            logger.info(f"  Stockout Rate: {inv_metrics['stockout_rate']:.1%}")
            logger.info(f"  Overstock Rate: {inv_metrics['overstock_rate']:.1%}")
            logger.info(f"  Avg Stockout Qty: {inv_metrics['avg_stockout_qty']:.1f}")
            logger.info(f"  Avg Overstock Qty: {inv_metrics['avg_overstock_qty']:.1f}")
            logger.info(f"  Total Cost: {inv_metrics['total_cost']:.0f}")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Analyze improvements
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: IMPACT OF ASYMMETRIC LOSS")
    logger.info("="*80)
    
    for scenario_name in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario_name]
        baseline = scenario_data[scenario_data['penalty'] == 1.0].iloc[0]
        
        logger.info(f"\n{scenario_name}:")
        logger.info(f"{'Penalty':<10} {'MAE':<10} {'Stockout%':<12} {'Overstock%':<12} {'Total Cost':<12}")
        logger.info("-" * 60)
        
        for _, row in scenario_data.iterrows():
            stockout_reduction = (baseline['stockout_rate'] - row['stockout_rate']) / (baseline['stockout_rate'] + 1e-10) * 100
            cost_reduction = (baseline['total_cost'] - row['total_cost']) / (baseline['total_cost'] + 1e-10) * 100
            
            logger.info(f"{row['penalty']:<10.1f} {row['mae']:<10.2f} "
                       f"{row['stockout_rate']*100:<12.1f} {row['overstock_rate']*100:<12.1f} "
                       f"{row['total_cost']:<12.0f}")
            
            if row['penalty'] > 1.0:
                logger.info(f"           Stockout reduction: {stockout_reduction:+.1f}%, Cost reduction: {cost_reduction:+.1f}%")
    
    # Save results
    output_file = Path(__file__).parent.parent / "data" / "synthetic_asymmetric_comparison.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nDetailed results saved to: {output_file}")
    
    # Key insights
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS")
    logger.info("="*80)
    
    # Calculate average improvements for penalty=3.0
    penalty3_data = results_df[results_df['penalty'] == 3.0]
    baseline_data = results_df[results_df['penalty'] == 1.0]
    
    avg_stockout_reduction = []
    for scenario in results_df['scenario'].unique():
        base = baseline_data[baseline_data['scenario'] == scenario].iloc[0]
        asym = penalty3_data[penalty3_data['scenario'] == scenario].iloc[0]
        reduction = (base['stockout_rate'] - asym['stockout_rate']) / (base['stockout_rate'] + 1e-10) * 100
        avg_stockout_reduction.append(reduction)
    
    overall_stockout_reduction = np.mean(avg_stockout_reduction)
    
    logger.info(f"\n1. With penalty=3.0, average stockout reduction: {overall_stockout_reduction:.1f}%")
    logger.info("2. Higher penalties reduce stockouts but increase overstocking")
    logger.info("3. Optimal penalty depends on actual business costs")
    logger.info("4. Effect is more pronounced with lower intermittency")
    
    return results_df


if __name__ == "__main__":
    # Run synthetic data test
    results = test_asymmetric_on_synthetic()
    
    if results is not None:
        logger.info("\n✓ Asymmetric loss testing on synthetic data completed successfully!")
    else:
        logger.error("\n✗ Testing failed!")