#!/usr/bin/env python
"""
Test script to verify database connectivity and data extraction.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from config.database import db_manager
from data.extractors.sales_extractor import SalesExtractor
from data.extractors.purchase_extractor import PurchaseExtractor
from data.extractors.job_order_extractor import JobOrderExtractor


def test_database_connection():
    """Test basic database connectivity."""
    print("Testing database connection...")
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            print("✓ Database connection successful!")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {str(e)}")
        return False


def test_table_exists(table_name):
    """Check if a table exists in the database."""
    try:
        query = "SHOW TABLES LIKE %s"
        results = db_manager.execute_query(query, [table_name])
        exists = len(results) > 0
        print(f"  {'✓' if exists else '✗'} Table {table_name}: {'exists' if exists else 'not found'}")
        return exists
    except Exception as e:
        print(f"  ✗ Error checking table {table_name}: {str(e)}")
        return False


def test_extractors():
    """Test all data extractors."""
    print("\nTesting data extractors...")
    
    # Set date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)  # Last 90 days
    
    print(f"Date range: {start_date} to {end_date}")
    
    # Test Sales Extractor
    print("\n1. Testing Sales Extractor...")
    try:
        sales_extractor = SalesExtractor()
        sales_data = sales_extractor.extract(start_date, end_date)
        print(f"  ✓ Sales data extracted: {len(sales_data)} rows")
        if not sales_data.empty:
            print(f"  - Unique items: {sales_data['item_id'].nunique()}")
            print(f"  - Total quantity: {sales_data['quantity'].sum():,.0f}")
            print(f"  - Date range: {sales_data['date'].min()} to {sales_data['date'].max()}")
            # Save sample
            sales_data.head(100).to_csv('/Users/carrickcheah/Project/prediction/data/raw/sample_sales.csv', index=False)
            print("  - Sample saved to data/raw/sample_sales.csv")
    except Exception as e:
        print(f"  ✗ Sales extraction failed: {str(e)}")
    
    # Test Purchase Extractor
    print("\n2. Testing Purchase Extractor...")
    try:
        purchase_extractor = PurchaseExtractor()
        purchase_data = purchase_extractor.extract(start_date, end_date)
        print(f"  ✓ Purchase data extracted: {len(purchase_data)} rows")
        if not purchase_data.empty:
            print(f"  - Unique items: {purchase_data['item_id'].nunique()}")
            print(f"  - Total quantity: {purchase_data['quantity'].sum():,.0f}")
            print(f"  - Average lead time: {purchase_data['lead_time_days'].mean():.1f} days")
            # Save sample
            purchase_data.head(100).to_csv('/Users/carrickcheah/Project/prediction/data/raw/sample_purchases.csv', index=False)
            print("  - Sample saved to data/raw/sample_purchases.csv")
    except Exception as e:
        print(f"  ✗ Purchase extraction failed: {str(e)}")
    
    # Test Job Order Extractor
    print("\n3. Testing Job Order Extractor...")
    try:
        job_extractor = JobOrderExtractor()
        consumption_data = job_extractor.extract(start_date, end_date)
        print(f"  ✓ Consumption data extracted: {len(consumption_data)} rows")
        if not consumption_data.empty:
            print(f"  - Unique items: {consumption_data['item_id'].nunique()}")
            print(f"  - Total consumption: {consumption_data['consumption'].sum():,.0f}")
            print(f"  - Date range: {consumption_data['date'].min()} to {consumption_data['date'].max()}")
            
            # Find top consumed items
            top_items = consumption_data.groupby('item_id')['consumption'].sum().sort_values(ascending=False).head(20)
            print("\n  Top 10 consumed items:")
            for i, (item_id, qty) in enumerate(top_items.head(10).items(), 1):
                print(f"    {i}. Item {item_id}: {qty:,.0f}")
            
            # Save sample and top items
            consumption_data.head(100).to_csv('/Users/carrickcheah/Project/prediction/data/raw/sample_consumption.csv', index=False)
            top_items.to_csv('/Users/carrickcheah/Project/prediction/data/raw/top_20_items.csv')
            print("  - Sample saved to data/raw/sample_consumption.csv")
            print("  - Top 20 items saved to data/raw/top_20_items.csv")
    except Exception as e:
        print(f"  ✗ Job order extraction failed: {str(e)}")
    
    return consumption_data if 'consumption_data' in locals() else None


def analyze_data_quality(consumption_data):
    """Analyze data quality and patterns."""
    if consumption_data is None or consumption_data.empty:
        print("\nNo consumption data to analyze.")
        return
    
    print("\n" + "="*50)
    print("DATA QUALITY ANALYSIS")
    print("="*50)
    
    # Check for missing dates
    date_range = pd.date_range(consumption_data['date'].min(), consumption_data['date'].max(), freq='D')
    unique_dates = consumption_data['date'].unique()
    missing_dates = set(date_range) - set(unique_dates)
    
    print(f"\n1. Date Coverage:")
    print(f"  - Total days in range: {len(date_range)}")
    print(f"  - Days with data: {len(unique_dates)}")
    print(f"  - Missing days: {len(missing_dates)}")
    if missing_dates:
        print(f"  - Missing dates sample: {sorted(list(missing_dates))[:5]}")
    
    # Check for data anomalies
    print(f"\n2. Data Anomalies:")
    negative_qty = consumption_data[consumption_data['consumption'] < 0]
    print(f"  - Negative quantities: {len(negative_qty)} rows")
    
    future_dates = consumption_data[consumption_data['date'] > pd.Timestamp(datetime.now().date())]
    print(f"  - Future dates: {len(future_dates)} rows")
    
    # Basic statistics
    print(f"\n3. Consumption Statistics:")
    print(f"  - Mean daily consumption: {consumption_data['consumption'].mean():.2f}")
    print(f"  - Std deviation: {consumption_data['consumption'].std():.2f}")
    print(f"  - Min: {consumption_data['consumption'].min()}")
    print(f"  - Max: {consumption_data['consumption'].max()}")
    
    # Zero consumption analysis
    zero_consumption = consumption_data[consumption_data['consumption'] == 0]
    print(f"\n4. Zero Consumption:")
    print(f"  - Records with zero consumption: {len(zero_consumption)} ({len(zero_consumption)/len(consumption_data)*100:.1f}%)")
    
    # Items analysis
    print(f"\n5. Items Analysis:")
    items_per_day = consumption_data.groupby('date')['item_id'].nunique()
    print(f"  - Average items per day: {items_per_day.mean():.0f}")
    print(f"  - Max items in a day: {items_per_day.max()}")
    print(f"  - Min items in a day: {items_per_day.min()}")


def main():
    """Main test function."""
    print("INVENTORY FORECASTING - DATA EXTRACTION TEST")
    print("=" * 50)
    
    settings = get_settings()
    print(f"Database: {settings.MARIADB_HOST}:{settings.MARIADB_PORT}/{settings.MARIADB_DATABASE}")
    
    # Test database connection
    if not test_database_connection():
        print("\nCannot proceed without database connection.")
        return
    
    # Check required tables
    print("\nChecking required tables...")
    tables = [
        'tbl_sorder_item', 'tbl_sorder_txn',
        'tbl_porder_item', 'tbl_porder_txn',
        'tbl_jo_item', 'tbl_jo_txn',
        'tbl_product_code'
    ]
    
    for table in tables:
        test_table_exists(table)
    
    # Test data extraction
    consumption_data = test_extractors()
    
    # Analyze data quality
    analyze_data_quality(consumption_data)
    
    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()