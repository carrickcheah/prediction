#!/usr/bin/env python3
"""
Extract and analyze lead time features from purchase order history.
Creates features for improving inventory forecasting accuracy.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.database import db_manager
from data.extractors.purchase_extractor import PurchaseExtractor
from forecasting.features.lead_time_features import LeadTimeFeatures
from utils.logger import setup_logger

logger = setup_logger("extract_lead_times")


def extract_purchase_history(lookback_days: int = 365) -> pd.DataFrame:
    """
    Extract purchase order history from database.
    
    Args:
        lookback_days: Number of days to look back
        
    Returns:
        DataFrame with purchase order data
    """
    logger.info(f"Extracting purchase orders for last {lookback_days} days")
    
    extractor = PurchaseExtractor()
    
    end_date = datetime(2025, 7, 22)  # Data available up to this date
    start_date = end_date - timedelta(days=lookback_days)
    
    purchase_df = extractor.extract(start_date, end_date)
    
    if purchase_df.empty:
        logger.warning("No purchase orders found")
        return pd.DataFrame()
        
    logger.info(f"Extracted {len(purchase_df)} purchase orders")
    
    # Add actual receive dates (simulated for now - in production would come from goods receipt)
    # For demonstration, simulate that 80% arrive on time, 15% late, 5% early
    np.random.seed(42)
    purchase_df['actual_receive_date'] = purchase_df['eta_date'].copy()
    
    # Simulate some delays and early arrivals
    n_orders = len(purchase_df)
    delay_mask = np.random.random(n_orders) < 0.15  # 15% late
    early_mask = np.random.random(n_orders) < 0.05  # 5% early
    
    # Add random delays (1-7 days late)
    purchase_df.loc[delay_mask, 'actual_receive_date'] = (
        pd.to_datetime(purchase_df.loc[delay_mask, 'eta_date']) + 
        pd.to_timedelta(np.random.randint(1, 8, delay_mask.sum()), unit='D')
    )
    
    # Add early arrivals (1-3 days early)
    purchase_df.loc[early_mask, 'actual_receive_date'] = (
        pd.to_datetime(purchase_df.loc[early_mask, 'eta_date']) - 
        pd.to_timedelta(np.random.randint(1, 4, early_mask.sum()), unit='D')
    )
    
    return purchase_df


def analyze_lead_times_for_top_parts(top_n: int = 20):
    """
    Analyze lead times for top N parts and create features.
    
    Args:
        top_n: Number of top parts to analyze
    """
    logger.info("=" * 80)
    logger.info("LEAD TIME FEATURE EXTRACTION")
    logger.info("=" * 80)
    
    # Load top parts
    parts_file = Path(__file__).parent.parent / "data" / "parts_demand_analysis.csv"
    if not parts_file.exists():
        logger.error(f"Parts analysis file not found: {parts_file}")
        return None
        
    parts_df = pd.read_csv(parts_file).head(top_n)
    part_ids = parts_df['part_id'].tolist()
    
    # Extract purchase history
    purchase_df = extract_purchase_history(lookback_days=365)
    
    if purchase_df.empty:
        logger.error("No purchase data available")
        return None
        
    # Filter for top parts
    purchase_df = purchase_df[purchase_df['item_id'].isin(part_ids)]
    
    if purchase_df.empty:
        logger.warning("No purchase orders found for top parts")
        # Try all parts instead
        purchase_df = extract_purchase_history(lookback_days=365)
        if len(purchase_df) > 0:
            # Take top ordered items
            top_ordered = purchase_df['item_id'].value_counts().head(20).index.tolist()
            purchase_df = purchase_df[purchase_df['item_id'].isin(top_ordered)]
            logger.info(f"Using top {len(top_ordered)} ordered items instead")
    
    if purchase_df.empty:
        logger.error("Still no purchase data after filtering")
        return None
        
    logger.info(f"Analyzing {len(purchase_df)} purchase orders for {purchase_df['item_id'].nunique()} parts")
    
    # Initialize lead time feature generator
    lt_features = LeadTimeFeatures(lookback_days=365)
    
    # Extract lead times
    lead_time_df = lt_features.extract_lead_times_from_purchases(purchase_df)
    
    # Calculate statistics
    logger.info("\n" + "="*60)
    logger.info("LEAD TIME STATISTICS BY ITEM-SUPPLIER")
    logger.info("="*60)
    
    lead_time_stats = lt_features.calculate_lead_time_stats(
        lead_time_df,
        group_by=['item_id', 'supplier_id']
    )
    
    if not lead_time_stats.empty:
        # Display top results
        logger.info("\nTop 10 Item-Supplier Combinations by Order Volume:")
        top_stats = lead_time_stats.nlargest(10, 'order_count')
        
        for _, row in top_stats.iterrows():
            logger.info(f"\nItem {row['item_id']} - Supplier {row['supplier_id']}:")
            logger.info(f"  Orders: {row['order_count']}")
            logger.info(f"  Lead Time: {row['lead_time_mean']:.1f} days (median: {row['lead_time_median']:.1f})")
            logger.info(f"  Std Dev: {row['lead_time_std']:.1f} days")
            logger.info(f"  Range: {row['lead_time_min']:.0f} - {row['lead_time_max']:.0f} days")
            logger.info(f"  95th Percentile: {row['lead_time_p95']:.0f} days")
            logger.info(f"  Reliability: {row['supplier_reliability']:.1%}")
        
        # Save statistics
        stats_file = Path(__file__).parent.parent / "data" / "lead_time_statistics.csv"
        lead_time_stats.to_csv(stats_file, index=False)
        logger.info(f"\nLead time statistics saved to: {stats_file}")
    
    # Calculate supplier performance
    logger.info("\n" + "="*60)
    logger.info("SUPPLIER PERFORMANCE METRICS")
    logger.info("="*60)
    
    supplier_metrics = lt_features.get_supplier_performance_metrics(lead_time_df)
    
    if not supplier_metrics.empty:
        # Display top suppliers
        logger.info("\nTop 5 Suppliers by Performance Score:")
        top_suppliers = supplier_metrics.nlargest(5, 'performance_score')
        
        for _, row in top_suppliers.iterrows():
            logger.info(f"\nSupplier {row['supplier_id']} (Rank #{row['supplier_rank']:.0f}):")
            logger.info(f"  Performance Score: {row['performance_score']:.1f}/100")
            logger.info(f"  Avg Lead Time: {row['avg_lead_time']:.1f} days")
            logger.info(f"  On-Time Rate: {row['on_time_rate']:.1%}")
            logger.info(f"  Avg Delay: {row['avg_delay']:.1f} days")
            logger.info(f"  Total Orders: {row['total_orders']:.0f}")
        
        # Save supplier metrics
        supplier_file = Path(__file__).parent.parent / "data" / "supplier_performance.csv"
        supplier_metrics.to_csv(supplier_file, index=False)
        logger.info(f"\nSupplier metrics saved to: {supplier_file}")
    
    # Calculate dynamic lead times
    logger.info("\n" + "="*60)
    logger.info("DYNAMIC LEAD TIME TRENDS")
    logger.info("="*60)
    
    reference_date = datetime(2025, 7, 22)
    dynamic_stats = lt_features.calculate_dynamic_lead_times(
        lead_time_df,
        reference_date,
        windows=[30, 60, 90, 180]
    )
    
    if not dynamic_stats.empty:
        # Find items with changing lead times
        dynamic_stats['trend_180d'] = dynamic_stats.get('lead_time_trend_180d', 0)
        trending_items = dynamic_stats[abs(dynamic_stats['trend_180d']) > 0.1].head(5)
        
        if len(trending_items) > 0:
            logger.info("\nItems with Changing Lead Times:")
            for _, row in trending_items.iterrows():
                trend_dir = "increasing" if row['trend_180d'] > 0 else "decreasing"
                logger.info(f"\nItem {row['item_id']}:")
                logger.info(f"  Lead time trend: {trend_dir} ({row['trend_180d']:.3f} days/day)")
                logger.info(f"  30-day avg: {row.get('lead_time_mean_30d', np.nan):.1f} days")
                logger.info(f"  180-day avg: {row.get('lead_time_mean_180d', np.nan):.1f} days")
        
        # Save dynamic stats
        dynamic_file = Path(__file__).parent.parent / "data" / "lead_time_dynamics.csv"
        dynamic_stats.to_csv(dynamic_file, index=False)
        logger.info(f"\nDynamic lead time features saved to: {dynamic_file}")
    
    # Create example consumption data with lead time features
    logger.info("\n" + "="*60)
    logger.info("CREATING ENHANCED FEATURES FOR FORECASTING")
    logger.info("="*60)
    
    # Load consumption data for one part as example
    if lead_time_stats['item_id'].nunique() > 0:
        example_item = lead_time_stats['item_id'].iloc[0]
        
        # Create mock consumption data
        dates = pd.date_range(start='2025-01-01', end='2025-07-22', freq='D')
        consumption_df = pd.DataFrame({
            'date': dates,
            'item_id': example_item,
            'consumption': np.random.poisson(5, len(dates))  # Mock consumption
        })
        
        # Add lead time features
        enhanced_df = lt_features.create_lead_time_features(
            consumption_df,
            lead_time_stats,
            item_col='item_id'
        )
        
        # Add urgency features
        enhanced_df = lt_features.create_procurement_urgency_features(enhanced_df)
        
        # Display feature summary
        lead_time_cols = [c for c in enhanced_df.columns if 'lead_time' in c or 'supplier' in c or 'order' in c]
        
        logger.info(f"\nAdded {len(lead_time_cols)} lead time features:")
        for col in lead_time_cols[:10]:  # Show first 10
            if not enhanced_df[col].isna().all():
                logger.info(f"  {col}: {enhanced_df[col].iloc[0]:.2f}" if enhanced_df[col].dtype in ['float64', 'int64'] else f"  {col}: {enhanced_df[col].iloc[0]}")
        
        # Save example
        example_file = Path(__file__).parent.parent / "data" / "example_with_lead_time_features.csv"
        enhanced_df.head(30).to_csv(example_file, index=False)
        logger.info(f"\nExample data with features saved to: {example_file}")
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\n✓ Processed {len(purchase_df)} purchase orders")
    logger.info(f"✓ Calculated lead times for {lead_time_stats['item_id'].nunique()} items")
    logger.info(f"✓ Evaluated {supplier_metrics['supplier_id'].nunique()} suppliers")
    logger.info(f"✓ Generated {len(lead_time_cols) if 'lead_time_cols' in locals() else 0} lead time features")
    
    return {
        'lead_time_stats': lead_time_stats,
        'supplier_metrics': supplier_metrics,
        'dynamic_stats': dynamic_stats
    }


if __name__ == "__main__":
    # Run lead time analysis
    results = analyze_lead_times_for_top_parts(top_n=20)
    
    if results:
        logger.info("\n✓ Lead time feature extraction completed successfully!")
    else:
        logger.error("\n✗ Lead time extraction failed!")