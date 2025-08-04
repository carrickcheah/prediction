#!/usr/bin/env python
"""
Extract full year (365 days) of historical data and save to CSV.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.extractors.sales_extractor import SalesExtractor
from data.extractors.purchase_extractor import PurchaseExtractor
from data.extractors.job_order_extractor import JobOrderExtractor
from utils.logger import setup_logger

logger = setup_logger("data_extraction")


def extract_full_year_data():
    """Extract 365 days of historical data from all sources."""
    # Set date range - 365 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    logger.info(f"Extracting data from {start_date} to {end_date}")
    print(f"\nExtracting 365 days of data: {start_date} to {end_date}")
    
    # Create output directory
    output_dir = Path("/Users/carrickcheah/Project/prediction/data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract Sales Data
    print("\n1. Extracting Sales Data...")
    try:
        sales_extractor = SalesExtractor()
        sales_data = sales_extractor.extract(start_date, end_date)
        
        if not sales_data.empty:
            # Save full dataset
            sales_path = output_dir / "sales_data_365days.csv"
            sales_data.to_csv(sales_path, index=False)
            print(f"   ✓ Saved {len(sales_data):,} sales records to {sales_path.name}")
            print(f"   - Unique items: {sales_data['item_id'].nunique()}")
            print(f"   - Date range: {sales_data['date'].min()} to {sales_data['date'].max()}")
            print(f"   - Total quantity: {sales_data['quantity'].sum():,.0f}")
        else:
            print("   ✗ No sales data found")
    except Exception as e:
        logger.error(f"Sales extraction failed: {str(e)}")
        print(f"   ✗ Sales extraction failed: {str(e)}")
    
    # Extract Purchase Data
    print("\n2. Extracting Purchase Data...")
    try:
        purchase_extractor = PurchaseExtractor()
        purchase_data = purchase_extractor.extract(start_date, end_date)
        
        if not purchase_data.empty:
            # Save full dataset
            purchase_path = output_dir / "purchase_data_365days.csv"
            purchase_data.to_csv(purchase_path, index=False)
            print(f"   ✓ Saved {len(purchase_data):,} purchase records to {purchase_path.name}")
            print(f"   - Unique items: {purchase_data['item_id'].nunique()}")
            print(f"   - Date range: {purchase_data['order_date'].min()} to {purchase_data['order_date'].max()}")
            print(f"   - Total quantity: {purchase_data['quantity'].sum():,.0f}")
            print(f"   - Average lead time: {purchase_data['lead_time_days'].mean():.1f} days")
        else:
            print("   ✗ No purchase data found")
    except Exception as e:
        logger.error(f"Purchase extraction failed: {str(e)}")
        print(f"   ✗ Purchase extraction failed: {str(e)}")
    
    # Extract Consumption Data (Job Orders)
    print("\n3. Extracting Consumption Data...")
    try:
        job_extractor = JobOrderExtractor()
        consumption_data = job_extractor.extract(start_date, end_date)
        
        if not consumption_data.empty:
            # Save full dataset
            consumption_path = output_dir / "consumption_data_365days.csv"
            consumption_data.to_csv(consumption_path, index=False)
            print(f"   ✓ Saved {len(consumption_data):,} consumption records to {consumption_path.name}")
            print(f"   - Unique items: {consumption_data['item_id'].nunique()}")
            print(f"   - Date range: {consumption_data['date'].min()} to {consumption_data['date'].max()}")
            print(f"   - Total consumption: {consumption_data['consumption'].sum():,.0f}")
            
            # Update top items based on full year
            top_items = consumption_data.groupby('item_id')['consumption'].sum().sort_values(ascending=False).head(20)
            top_items_path = output_dir / "top_20_items_365days.csv"
            top_items.to_csv(top_items_path)
            print(f"   ✓ Updated top 20 items saved to {top_items_path.name}")
            
            # Show new top 10
            print("\n   Top 10 items by annual consumption:")
            for i, (item_id, qty) in enumerate(top_items.head(10).items(), 1):
                print(f"      {i}. Item {item_id}: {qty:,.0f}")
        else:
            print("   ✗ No consumption data found")
    except Exception as e:
        logger.error(f"Consumption extraction failed: {str(e)}")
        print(f"   ✗ Consumption extraction failed: {str(e)}")
    
    # Create combined dataset for top 20 items
    print("\n4. Creating combined dataset for top 20 items...")
    try:
        if 'consumption_data' in locals() and not consumption_data.empty:
            # Get top 20 items
            top_20_items = top_items.head(20).index.tolist()
            
            # Filter consumption data for top 20
            top_consumption = consumption_data[consumption_data['item_id'].isin(top_20_items)].copy()
            
            # Create daily aggregated data
            daily_consumption = top_consumption.groupby(['date', 'item_id'])['consumption'].sum().reset_index()
            
            # Pivot to wide format (items as columns)
            consumption_pivot = daily_consumption.pivot(index='date', columns='item_id', values='consumption').fillna(0)
            
            # Save pivoted data
            pivot_path = output_dir / "top_20_consumption_pivot_365days.csv"
            consumption_pivot.to_csv(pivot_path)
            print(f"   ✓ Saved pivoted consumption data for top 20 items to {pivot_path.name}")
            print(f"   - Shape: {consumption_pivot.shape}")
            print(f"   - Date range: {consumption_pivot.index.min()} to {consumption_pivot.index.max()}")
            
    except Exception as e:
        logger.error(f"Combined dataset creation failed: {str(e)}")
        print(f"   ✗ Combined dataset creation failed: {str(e)}")
    
    print("\n" + "="*50)
    print("DATA EXTRACTION COMPLETE")
    print("="*50)
    print(f"\nAll files saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Run data quality analysis on full year data")
    print("2. Build baseline models using the extracted data")
    print("3. Implement forecasting models for top 20 items")


if __name__ == "__main__":
    extract_full_year_data()