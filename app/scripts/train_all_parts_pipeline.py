#!/usr/bin/env python3
"""
Scalable training pipeline for all 6000+ parts.
Implements batch processing, parallel execution, and checkpointing.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from functools import partial
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.database import db_manager
from forecasting.features.intermittent_features import IntermittentDemandFeatures
from forecasting.features.lead_time_features import LeadTimeFeatures
from forecasting.models.xgboost_intermittent import train_model_for_demand_pattern
from reports.excel_reporter import ExcelReporter
from utils.logger import setup_logger

logger = setup_logger("train_all_parts")


class ScalableTrainingPipeline:
    """
    Scalable pipeline for training models on all parts.
    Features:
    - Batch processing to manage memory
    - Parallel training with multiprocessing
    - Checkpoint/resume capability
    - Progress tracking
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        n_workers: int = None,
        checkpoint_dir: str = None,
        use_asymmetric_loss: bool = True,
        stockout_penalty: float = 3.0
    ):
        """
        Initialize scalable pipeline.
        
        Args:
            batch_size: Number of parts per batch
            n_workers: Number of parallel workers (None = CPU count)
            checkpoint_dir: Directory for saving checkpoints
            use_asymmetric_loss: Whether to use asymmetric loss
            stockout_penalty: Penalty for stockouts
        """
        self.batch_size = batch_size
        self.n_workers = n_workers or mp.cpu_count()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).parent.parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.use_asymmetric_loss = use_asymmetric_loss
        self.stockout_penalty = stockout_penalty
        
        self.feature_generator = IntermittentDemandFeatures()
        self.lead_time_generator = LeadTimeFeatures()
        
        self.logger = logger
        
    def get_all_parts(self, min_data_points: int = 30) -> pd.DataFrame:
        """
        Get list of all parts with sufficient data.
        
        Args:
            min_data_points: Minimum number of data points required
            
        Returns:
            DataFrame with part information
        """
        self.logger.info("Getting list of all parts from database...")
        
        query = f"""
        SELECT 
            ji.ItemId_i as part_id,
            pc.StkCode_v as stock_code,
            COUNT(DISTINCT DATE(jt.TxnDate_dd)) as data_points,
            SUM(ji.Qty_d) as total_consumption,
            AVG(ji.Qty_d) as avg_consumption
        FROM tbl_jo_item ji
        JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
        LEFT JOIN tbl_product_code pc ON ji.ItemId_i = pc.ItemId_i
        WHERE 
            ji.InOut_c = 'I'
            AND jt.TxnDate_dd >= '2024-01-01'
            AND jt.TxnDate_dd <= '2025-07-22'
            AND jt.Void_c = '0'
        GROUP BY ji.ItemId_i, pc.StkCode_v
        HAVING COUNT(DISTINCT DATE(jt.TxnDate_dd)) >= {min_data_points}
        ORDER BY total_consumption DESC
        """
        
        results = db_manager.execute_query(query)
        parts_df = pd.DataFrame(results)
        
        self.logger.info(f"Found {len(parts_df)} parts with >= {min_data_points} data points")
        
        return parts_df
        
    def load_batch_data(self, part_ids: List[int]) -> pd.DataFrame:
        """
        Load consumption data for a batch of parts.
        
        Args:
            part_ids: List of part IDs to load
            
        Returns:
            DataFrame with consumption data
        """
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
        """
        
        results = db_manager.execute_query(query)
        consumption_df = pd.DataFrame(results)
        
        if not consumption_df.empty:
            consumption_df['date'] = pd.to_datetime(consumption_df['date'])
            
        return consumption_df
        
    def train_single_part(
        self,
        part_data: Tuple[int, str, pd.DataFrame],
        lead_time_stats: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Train model for a single part.
        
        Args:
            part_data: Tuple of (part_id, stock_code, consumption_df)
            lead_time_stats: Optional lead time statistics
            
        Returns:
            Dictionary with training results
        """
        part_id, stock_code, consumption_df = part_data
        
        try:
            # Prepare data
            part_df = consumption_df[consumption_df['part_id'] == part_id].copy()
            
            if part_df.empty or len(part_df) < 30:
                return {
                    'part_id': part_id,
                    'stock_code': stock_code,
                    'status': 'insufficient_data',
                    'error': f'Only {len(part_df)} data points'
                }
                
            # Set date index and fill missing days
            part_df.set_index('date', inplace=True)
            date_range = pd.date_range(part_df.index.min(), part_df.index.max(), freq='D')
            part_df = part_df.reindex(date_range, fill_value=0)
            part_df.index.name = 'date'
            
            # Ensure consumption column
            if 'consumption' not in part_df.columns:
                part_df['consumption'] = part_df.iloc[:, 0]
                
            # Calculate zero percentage
            zero_pct = (part_df['consumption'] == 0).mean() * 100
            
            # Generate features
            part_df = self.feature_generator.create_all_features(part_df, target_col='consumption')
            
            # Add lead time features if available
            if lead_time_stats is not None and not lead_time_stats.empty:
                part_df['item_id'] = part_id  # For merging
                part_df = self.lead_time_generator.create_lead_time_features(
                    part_df, lead_time_stats, item_col='item_id'
                )
                
            # Prepare for training
            feature_cols = [col for col in part_df.columns 
                          if col not in ['consumption', 'part_id', 'item_id']]
            
            X = part_df[feature_cols].fillna(0)
            y = part_df['consumption']
            
            # Remove NaN targets
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 30:
                return {
                    'part_id': part_id,
                    'stock_code': stock_code,
                    'status': 'insufficient_valid_data',
                    'error': f'Only {len(X)} valid samples'
                }
                
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            model, metrics = train_model_for_demand_pattern(
                X_train, y_train, X_val, y_val,
                zero_percentage=zero_pct,
                use_asymmetric_loss=self.use_asymmetric_loss,
                stockout_penalty=self.stockout_penalty
            )
            
            # Save model
            model_path = self.checkpoint_dir / f"model_part_{part_id}.pkl"
            model.save(str(model_path))
            
            return {
                'part_id': part_id,
                'stock_code': stock_code,
                'status': 'success',
                'zero_pct': zero_pct,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'mae': metrics.get('val_mae', metrics.get('train_mae')),
                'model_path': str(model_path)
            }
            
        except Exception as e:
            return {
                'part_id': part_id,
                'stock_code': stock_code,
                'status': 'error',
                'error': str(e)
            }
            
    def process_batch(
        self,
        batch_parts: pd.DataFrame,
        batch_num: int,
        total_batches: int,
        lead_time_stats: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Process a batch of parts.
        
        Args:
            batch_parts: DataFrame with part information
            batch_num: Current batch number
            total_batches: Total number of batches
            lead_time_stats: Optional lead time statistics
            
        Returns:
            List of training results
        """
        self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_parts)} parts)")
        
        # Load consumption data for batch
        part_ids = batch_parts['part_id'].tolist()
        consumption_df = self.load_batch_data(part_ids)
        
        if consumption_df.empty:
            self.logger.warning(f"No consumption data for batch {batch_num}")
            return []
            
        # Prepare data for parallel processing
        part_data_list = [
            (row['part_id'], row.get('stock_code', f'PART_{row["part_id"]}'), consumption_df)
            for _, row in batch_parts.iterrows()
        ]
        
        # Train models in parallel
        if self.n_workers > 1:
            with mp.Pool(self.n_workers) as pool:
                train_func = partial(self.train_single_part, lead_time_stats=lead_time_stats)
                results = pool.map(train_func, part_data_list)
        else:
            # Single threaded for debugging
            results = [
                self.train_single_part(part_data, lead_time_stats)
                for part_data in part_data_list
            ]
            
        # Save batch checkpoint
        checkpoint_file = self.checkpoint_dir / f"batch_{batch_num}_results.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(results, f)
            
        # Clear memory
        del consumption_df
        gc.collect()
        
        return results
        
    def run_training(
        self,
        max_parts: Optional[int] = None,
        resume_from_batch: int = 0
    ) -> pd.DataFrame:
        """
        Run the complete training pipeline.
        
        Args:
            max_parts: Maximum number of parts to train (None = all)
            resume_from_batch: Batch number to resume from
            
        Returns:
            DataFrame with training results
        """
        self.logger.info("=" * 80)
        self.logger.info("SCALABLE TRAINING PIPELINE FOR ALL PARTS")
        self.logger.info("=" * 80)
        
        # Get all parts
        all_parts = self.get_all_parts(min_data_points=10)
        
        if max_parts:
            all_parts = all_parts.head(max_parts)
            self.logger.info(f"Limited to {max_parts} parts for this run")
            
        total_parts = len(all_parts)
        
        # Load lead time statistics if available
        lead_time_file = Path(__file__).parent.parent / "data" / "lead_time_statistics.csv"
        lead_time_stats = None
        if lead_time_file.exists():
            lead_time_stats = pd.read_csv(lead_time_file)
            self.logger.info(f"Loaded lead time statistics for {lead_time_stats['item_id'].nunique()} items")
            
        # Create batches
        n_batches = (total_parts + self.batch_size - 1) // self.batch_size
        self.logger.info(f"Training {total_parts} parts in {n_batches} batches of {self.batch_size}")
        
        # Training results
        all_results = []
        
        # Process batches
        for batch_num in range(resume_from_batch, n_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_parts)
            
            batch_parts = all_parts.iloc[start_idx:end_idx]
            
            # Process batch
            batch_results = self.process_batch(
                batch_parts,
                batch_num + 1,
                n_batches,
                lead_time_stats
            )
            
            all_results.extend(batch_results)
            
            # Log progress
            completed_parts = end_idx
            success_count = sum(1 for r in all_results if r['status'] == 'success')
            error_count = sum(1 for r in all_results if r['status'] == 'error')
            
            self.logger.info(f"Progress: {completed_parts}/{total_parts} parts "
                           f"({completed_parts/total_parts*100:.1f}%)")
            self.logger.info(f"Success: {success_count}, Errors: {error_count}")
            
            # Save intermediate results
            if (batch_num + 1) % 10 == 0:
                self.save_results(all_results, f"intermediate_{batch_num+1}")
                
        # Final results
        results_df = pd.DataFrame(all_results)
        
        # Save final results
        self.save_results(all_results, "final")
        
        # Generate summary report
        self.generate_summary_report(results_df)
        
        return results_df
        
    def save_results(self, results: List[Dict], suffix: str):
        """Save training results to file."""
        results_file = self.checkpoint_dir / f"training_results_{suffix}.csv"
        pd.DataFrame(results).to_csv(results_file, index=False)
        self.logger.info(f"Saved results to {results_file}")
        
    def generate_summary_report(self, results_df: pd.DataFrame):
        """Generate summary report of training results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("=" * 80)
        
        # Overall statistics
        total_parts = len(results_df)
        successful = results_df[results_df['status'] == 'success']
        failed = results_df[results_df['status'] != 'success']
        
        self.logger.info(f"\nTotal Parts Processed: {total_parts}")
        self.logger.info(f"Successful: {len(successful)} ({len(successful)/total_parts*100:.1f}%)")
        self.logger.info(f"Failed: {len(failed)} ({len(failed)/total_parts*100:.1f}%)")
        
        if len(successful) > 0:
            # Performance metrics
            self.logger.info(f"\nPerformance Metrics (Successful Models):")
            self.logger.info(f"  Average MAE: {successful['mae'].mean():.3f}")
            self.logger.info(f"  Median MAE: {successful['mae'].median():.3f}")
            self.logger.info(f"  Best MAE: {successful['mae'].min():.3f}")
            self.logger.info(f"  Worst MAE: {successful['mae'].max():.3f}")
            
            # Zero percentage distribution
            self.logger.info(f"\nZero Percentage Distribution:")
            zero_bins = pd.cut(successful['zero_pct'], bins=[0, 30, 50, 70, 90, 100])
            for bin_name, count in zero_bins.value_counts().sort_index().items():
                self.logger.info(f"  {bin_name}: {count} parts")
                
        # Failure reasons
        if len(failed) > 0:
            self.logger.info(f"\nFailure Reasons:")
            for status, group in failed.groupby('status'):
                self.logger.info(f"  {status}: {len(group)} parts")
                
        # Save detailed report
        report_file = self.checkpoint_dir / "training_summary.txt"
        with open(report_file, 'w') as f:
            f.write(f"Training Summary - {datetime.now()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Parts: {total_parts}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n")
            if len(successful) > 0:
                f.write(f"Average MAE: {successful['mae'].mean():.3f}\n")
                
        self.logger.info(f"\nDetailed report saved to {report_file}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for all parts")
    parser.add_argument('--batch-size', type=int, default=100, help='Parts per batch')
    parser.add_argument('--max-parts', type=int, help='Maximum parts to train')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--resume-batch', type=int, default=0, help='Resume from batch number')
    parser.add_argument('--no-asymmetric', action='store_true', help='Disable asymmetric loss')
    parser.add_argument('--penalty', type=float, default=3.0, help='Stockout penalty')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ScalableTrainingPipeline(
        batch_size=args.batch_size,
        n_workers=args.workers,
        use_asymmetric_loss=not args.no_asymmetric,
        stockout_penalty=args.penalty
    )
    
    # Run training
    try:
        results = pipeline.run_training(
            max_parts=args.max_parts,
            resume_from_batch=args.resume_batch
        )
        
        logger.info(f"\n✓ Training pipeline completed successfully!")
        logger.info(f"✓ Trained models for {len(results[results['status'] == 'success'])} parts")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()