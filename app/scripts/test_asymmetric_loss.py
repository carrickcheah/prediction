#!/usr/bin/env python3
"""
Test and compare asymmetric loss vs standard loss for inventory forecasting.
Shows the impact of penalizing stockouts more heavily.
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.database import db_manager
from forecasting.features.intermittent_features import IntermittentDemandFeatures
from forecasting.models.xgboost_intermittent import train_model_for_demand_pattern
from forecasting.models.asymmetric_loss import calculate_inventory_metrics
from utils.logger import setup_logger

logger = setup_logger("test_asymmetric_loss")


def compare_loss_functions(top_n_parts: int = 5):
    """
    Compare standard vs asymmetric loss on top N parts.
    
    Args:
        top_n_parts: Number of parts to test
    """
    logger.info("=" * 80)
    logger.info("ASYMMETRIC LOSS COMPARISON TEST")
    logger.info("=" * 80)
    
    # Load top parts
    parts_file = Path(__file__).parent.parent / "data" / "parts_demand_analysis.csv"
    if not parts_file.exists():
        logger.error(f"Parts analysis file not found: {parts_file}")
        return
        
    parts_df = pd.read_csv(parts_file).head(top_n_parts)
    
    # Load consumption data
    logger.info(f"\nLoading consumption data for {top_n_parts} parts...")
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
        AND jt.TxnDate_dd >= '2024-01-01'
        AND jt.TxnDate_dd <= '2025-07-22'
        AND jt.Void_c = '0'
    GROUP BY ji.ItemId_i, DATE(jt.TxnDate_dd)
    ORDER BY ji.ItemId_i, date
    """
    
    results = db_manager.execute_query(query)
    consumption_df = pd.DataFrame(results)
    
    if consumption_df.empty:
        logger.error("No consumption data found")
        return
        
    consumption_df['date'] = pd.to_datetime(consumption_df['date'])
    
    # Feature generator
    feature_generator = IntermittentDemandFeatures()
    
    # Results storage
    comparison_results = []
    
    # Process each part
    for idx, row in parts_df.iterrows():
        part_id = row['part_id']
        stock_code = row['stock_code']
        zero_pct = row['zero_percentage']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Part {part_id} ({stock_code})")
        logger.info(f"Zero percentage: {zero_pct:.1f}%")
        logger.info(f"{'='*60}")
        
        # Prepare part data
        part_data = consumption_df[consumption_df['part_id'] == part_id].copy()
        
        if part_data.empty or len(part_data) < 10:
            logger.warning(f"Insufficient data for part {part_id} (has {len(part_data)} records)")
            continue
            
        # Set date as index and fill missing days
        part_data.set_index('date', inplace=True)
        date_range = pd.date_range(part_data.index.min(), part_data.index.max(), freq='D')
        part_data = part_data.reindex(date_range, fill_value=0)
        part_data.index.name = 'date'
        
        if 'consumption' not in part_data.columns:
            part_data['consumption'] = part_data.iloc[:, 0]
            
        # Generate features
        part_data = feature_generator.create_all_features(part_data, target_col='consumption')
        
        # Prepare features and target
        feature_cols = [col for col in part_data.columns 
                       if col not in ['consumption', 'part_id']]
        
        X = part_data[feature_cols].fillna(0)
        y = part_data['consumption']
        
        # Remove rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 20:
            logger.warning(f"Not enough samples after feature generation: {len(X)}")
            continue
            
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"\nTraining samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Train with STANDARD loss
        logger.info("\n--- Training with STANDARD Tweedie Loss ---")
        model_standard, metrics_standard = train_model_for_demand_pattern(
            X_train, y_train, X_val, y_val, 
            zero_percentage=zero_pct,
            use_asymmetric_loss=False
        )
        
        # Train with ASYMMETRIC loss
        logger.info("\n--- Training with ASYMMETRIC Loss (penalty=3.0) ---")
        model_asymmetric, metrics_asymmetric = train_model_for_demand_pattern(
            X_train, y_train, X_val, y_val,
            zero_percentage=zero_pct,
            use_asymmetric_loss=True,
            stockout_penalty=3.0
        )
        
        # Generate predictions on validation set
        pred_standard = model_standard.predict(X_val)
        pred_asymmetric = model_asymmetric.predict(X_val)
        
        # Calculate detailed inventory metrics
        metrics_std_inv = calculate_inventory_metrics(y_val.values, pred_standard, stockout_penalty=3.0)
        metrics_asym_inv = calculate_inventory_metrics(y_val.values, pred_asymmetric, stockout_penalty=3.0)
        
        # Compare results
        result = {
            'part_id': part_id,
            'stock_code': stock_code,
            'zero_pct': zero_pct,
            
            # Standard model metrics
            'std_mae': metrics_std_inv['mae'],
            'std_stockout_rate': metrics_std_inv['stockout_rate'],
            'std_overstock_rate': metrics_std_inv['overstock_rate'],
            'std_total_cost': metrics_std_inv['total_cost'],
            'std_stockout_cost': metrics_std_inv['stockout_cost'],
            'std_overstock_cost': metrics_std_inv['overstock_cost'],
            
            # Asymmetric model metrics
            'asym_mae': metrics_asym_inv['mae'],
            'asym_stockout_rate': metrics_asym_inv['stockout_rate'],
            'asym_overstock_rate': metrics_asym_inv['overstock_rate'],
            'asym_total_cost': metrics_asym_inv['total_cost'],
            'asym_stockout_cost': metrics_asym_inv['stockout_cost'],
            'asym_overstock_cost': metrics_asym_inv['overstock_cost'],
            
            # Improvements
            'stockout_reduction': (metrics_std_inv['stockout_rate'] - metrics_asym_inv['stockout_rate']) / (metrics_std_inv['stockout_rate'] + 1e-10) * 100,
            'cost_reduction': (metrics_std_inv['total_cost'] - metrics_asym_inv['total_cost']) / (metrics_std_inv['total_cost'] + 1e-10) * 100,
            'mae_change': (metrics_asym_inv['mae'] - metrics_std_inv['mae']) / (metrics_std_inv['mae'] + 1e-10) * 100
        }
        
        comparison_results.append(result)
        
        # Print comparison
        logger.info("\n" + "="*60)
        logger.info("COMPARISON RESULTS")
        logger.info("="*60)
        logger.info(f"\nStandard Model:")
        logger.info(f"  MAE: {metrics_std_inv['mae']:.3f}")
        logger.info(f"  Stockout Rate: {metrics_std_inv['stockout_rate']:.1%}")
        logger.info(f"  Overstock Rate: {metrics_std_inv['overstock_rate']:.1%}")
        logger.info(f"  Total Cost: {metrics_std_inv['total_cost']:.0f}")
        
        logger.info(f"\nAsymmetric Model (penalty=3.0):")
        logger.info(f"  MAE: {metrics_asym_inv['mae']:.3f}")
        logger.info(f"  Stockout Rate: {metrics_asym_inv['stockout_rate']:.1%}")
        logger.info(f"  Overstock Rate: {metrics_asym_inv['overstock_rate']:.1%}")
        logger.info(f"  Total Cost: {metrics_asym_inv['total_cost']:.0f}")
        
        logger.info(f"\nImprovements:")
        logger.info(f"  Stockout Reduction: {result['stockout_reduction']:.1f}%")
        logger.info(f"  Cost Reduction: {result['cost_reduction']:.1f}%")
        logger.info(f"  MAE Change: {result['mae_change']:+.1f}%")
    
    # Save results
    if comparison_results:
        results_df = pd.DataFrame(comparison_results)
        
        # Calculate summary statistics
        logger.info("\n" + "="*80)
        logger.info("OVERALL SUMMARY")
        logger.info("="*80)
        
        avg_stockout_reduction = results_df['stockout_reduction'].mean()
        avg_cost_reduction = results_df['cost_reduction'].mean()
        avg_mae_change = results_df['mae_change'].mean()
        
        logger.info(f"\nAverage Improvements Across {len(results_df)} Parts:")
        logger.info(f"  Stockout Reduction: {avg_stockout_reduction:.1f}%")
        logger.info(f"  Cost Reduction: {avg_cost_reduction:.1f}%")
        logger.info(f"  MAE Change: {avg_mae_change:+.1f}%")
        
        logger.info("\nKey Findings:")
        if avg_stockout_reduction > 20:
            logger.info("  ✓ Significant stockout reduction achieved (>20%)")
        if avg_cost_reduction > 10:
            logger.info("  ✓ Substantial cost savings achieved (>10%)")
        if avg_mae_change < 10:
            logger.info("  ✓ MAE increase is acceptable (<10%)")
            
        # Save to CSV
        output_file = Path(__file__).parent.parent / "data" / "asymmetric_loss_comparison.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to: {output_file}")
        
        return results_df
    else:
        logger.error("No results generated")
        return None


if __name__ == "__main__":
    # Run comparison test
    results = compare_loss_functions(top_n_parts=5)
    
    if results is not None:
        logger.info("\n✓ Asymmetric loss testing completed successfully!")
    else:
        logger.error("\n✗ Asymmetric loss testing failed!")