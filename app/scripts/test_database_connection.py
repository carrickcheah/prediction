#!/usr/bin/env python3
"""
Test database connection and extract sample data for analysis.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.database import db_manager
from config.settings import get_settings
from utils.logger import setup_logger

logger = setup_logger("test_database")


def test_connection():
    """Test basic database connection."""
    logger.info("Testing database connection...")
    
    try:
        settings = get_settings()
        logger.info(f"Connecting to {settings.MARIADB_HOST}:{settings.MARIADB_PORT}/{settings.MARIADB_DATABASE}")
        
        # Test basic query using db_manager
        result = db_manager.execute_query("SELECT 1 as test")
        
        if result and result[0]['test'] == 1:
            logger.info("Database connection successful!")
            return True
        else:
            logger.error("Database connection test failed")
            return False
            
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False


def check_tables():
    """Check if required tables exist and have data."""
    logger.info("Checking required tables...")
    
    required_tables = [
        'tbl_jo_item',
        'tbl_jo_txn', 
        'tbl_sorder_item',
        'tbl_sorder_txn',
        'tbl_porder_item',
        'tbl_porder_txn',
        'tbl_product_code'
    ]
    
    try:
        for table in required_tables:
            result = db_manager.execute_query(f"SELECT COUNT(*) as count FROM {table}")
            count = result[0]['count'] if result else 0
            logger.info(f"  {table}: {count:,} records")
            
    except Exception as e:
        logger.error(f"Error checking tables: {e}")


def get_top_parts_by_consumption():
    """Get top 20 parts by consumption volume in last 90 days."""
    logger.info("Analyzing top parts by consumption...")
    
    query = """
    SELECT 
        ji.ItemId_i as part_id,
        pc.StkCode_v as stock_code,
        pc.ProdName_v as product_name,
        COUNT(DISTINCT DATE(jt.TxnDate_dd)) as active_days,
        SUM(ji.Qty_d) as total_consumption,
        AVG(ji.Qty_d) as avg_daily_consumption,
        MIN(ji.Qty_d) as min_consumption,
        MAX(ji.Qty_d) as max_consumption,
        STD(ji.Qty_d) as std_consumption
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    LEFT JOIN tbl_product_code pc ON ji.ItemId_i = pc.ItemId_i
    WHERE 
        ji.InOut_c = 'I'  -- Input/consumption only
        AND jt.TxnDate_dd >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
        AND jt.Void_c = '0'
        AND ji.Void_c = '0'
    GROUP BY ji.ItemId_i, pc.StkCode_v, pc.ProdName_v
    ORDER BY total_consumption DESC
    LIMIT 20
    """
    
    try:
        # Use db_manager to execute query and convert to DataFrame
        results = db_manager.execute_query(query)
        df = pd.DataFrame(results)
        
        if not df.empty:
            logger.info(f"\nTop 20 Parts by Consumption (Last 90 Days):")
            logger.info("-" * 80)
            
            for idx, row in df.iterrows():
                logger.info(
                    f"{idx+1:2}. Part {row['part_id']:5} | "
                    f"{row['stock_code'] if row['stock_code'] else 'N/A':15} | "
                    f"Total: {row['total_consumption']:8.0f} | "
                    f"Avg: {row['avg_daily_consumption']:6.1f} | "
                    f"Active Days: {row['active_days']:3}"
                )
                
            # Save to CSV for analysis
            output_file = Path(__file__).parent.parent / "data" / "top_20_parts.csv"
            output_file.parent.mkdir(exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"\nSaved top 20 parts to: {output_file}")
            
            return df
        else:
            logger.warning("No consumption data found in last 90 days")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error analyzing top parts: {e}")
        return pd.DataFrame()


def analyze_data_quality():
    """Analyze data quality for job orders."""
    logger.info("\nAnalyzing data quality...")
    
    query = """
    SELECT 
        DATE(jt.TxnDate_dd) as date,
        COUNT(DISTINCT ji.ItemId_i) as unique_parts,
        COUNT(*) as transaction_count,
        SUM(CASE WHEN ji.Qty_d < 0 THEN 1 ELSE 0 END) as negative_qty_count,
        SUM(CASE WHEN ji.Qty_d = 0 THEN 1 ELSE 0 END) as zero_qty_count,
        SUM(CASE WHEN jt.TxnDate_dd > CURDATE() THEN 1 ELSE 0 END) as future_date_count
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    WHERE 
        ji.InOut_c = 'I'
        AND jt.TxnDate_dd >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
        AND jt.Void_c = '0'
    GROUP BY DATE(jt.TxnDate_dd)
    ORDER BY date DESC
    """
    
    try:
        results = db_manager.execute_query(query)
        df = pd.DataFrame(results)
        
        if not df.empty:
            logger.info(f"Data Quality Summary (Last 90 Days):")
            logger.info(f"  Total Days with Data: {len(df)}")
            logger.info(f"  Avg Parts per Day: {df['unique_parts'].mean():.1f}")
            logger.info(f"  Avg Transactions per Day: {df['transaction_count'].mean():.1f}")
            
            # Check for issues
            total_negative = df['negative_qty_count'].sum()
            total_zero = df['zero_qty_count'].sum()
            total_future = df['future_date_count'].sum()
            
            if total_negative > 0:
                logger.warning(f"  Found {total_negative} transactions with negative quantities")
            if total_zero > 0:
                logger.warning(f"  Found {total_zero} transactions with zero quantities")
            if total_future > 0:
                logger.warning(f"  Found {total_future} transactions with future dates")
                
            # Check for missing dates (weekends/holidays)
            if not df['date'].empty:
                df['date'] = pd.to_datetime(df['date'])
                date_range = pd.date_range(df['date'].min(), df['date'].max())
                missing_dates = set(date_range.date) - set(df['date'].dt.date)
                
                if missing_dates:
                    logger.info(f"  Missing Dates: {len(missing_dates)} days (likely weekends/holidays)")
                
            return df
        else:
            logger.warning("No data found for quality analysis")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in data quality analysis: {e}")
        return pd.DataFrame()


def get_sample_consumption_data():
    """Get sample consumption data for one part."""
    logger.info("\nExtracting sample consumption data...")
    
    # First get the top consumed part
    query_top = """
    SELECT 
        ji.ItemId_i as part_id,
        SUM(ji.Qty_d) as total
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    WHERE 
        ji.InOut_c = 'I'
        AND jt.TxnDate_dd >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
        AND jt.Void_c = '0'
    GROUP BY ji.ItemId_i
    ORDER BY total DESC
    LIMIT 1
    """
    
    try:
        results = db_manager.execute_query(query_top)
        top_part = results[0] if results else None
        
        if top_part:
            part_id = top_part['part_id']
            logger.info(f"Getting consumption pattern for Part ID: {part_id}")
            
            # Get daily consumption for this part
            query_consumption = f"""
            SELECT 
                DATE(jt.TxnDate_dd) as date,
                SUM(ji.Qty_d) as consumption
            FROM tbl_jo_item ji
            JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
            WHERE 
                ji.ItemId_i = {part_id}
                AND ji.InOut_c = 'I'
                AND jt.TxnDate_dd >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
                AND jt.Void_c = '0'
            GROUP BY DATE(jt.TxnDate_dd)
            ORDER BY date
            """
            
            results = db_manager.execute_query(query_consumption)
            df = pd.DataFrame(results)
            
            if not df.empty:
                # Fill missing dates with 0
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.asfreq('D', fill_value=0)
                
                logger.info(f"  Data points: {len(df)}")
                logger.info(f"  Mean consumption: {df['consumption'].mean():.2f}")
                logger.info(f"  Std deviation: {df['consumption'].std():.2f}")
                logger.info(f"  Zero consumption days: {(df['consumption'] == 0).sum()}")
                
                # Save sample data
                output_file = Path(__file__).parent.parent / "data" / f"sample_consumption_part_{part_id}.csv"
                df.to_csv(output_file)
                logger.info(f"  Saved to: {output_file}")
                
                return df
            
    except Exception as e:
        logger.error(f"Error getting sample data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("DATABASE CONNECTION AND DATA QUALITY TEST")
    logger.info("=" * 80)
    
    # Test connection
    if test_connection():
        # Check tables
        check_tables()
        
        # Get top parts
        top_parts = get_top_parts_by_consumption()
        
        # Analyze data quality
        quality_df = analyze_data_quality()
        
        # Get sample data
        sample_data = get_sample_consumption_data()
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
    else:
        logger.error("Database connection failed. Please check your .env configuration.")