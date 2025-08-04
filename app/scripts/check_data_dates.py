#!/usr/bin/env python3
"""
Check the date ranges of data in the database tables.
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.database import db_manager
from utils.logger import setup_logger

logger = setup_logger("check_dates")


def check_date_ranges():
    """Check date ranges for all transaction tables."""
    
    tables_to_check = [
        ('tbl_jo_txn', 'TxnDate_dd', 'Job Orders'),
        ('tbl_sorder_txn', 'TxnDate_dd', 'Sales Orders'),
        ('tbl_porder_txn', 'TxnDate_dd', 'Purchase Orders'),
    ]
    
    logger.info("=" * 80)
    logger.info("CHECKING DATE RANGES IN DATABASE")
    logger.info("=" * 80)
    
    for table, date_field, description in tables_to_check:
        query = f"""
        SELECT 
            MIN({date_field}) as min_date,
            MAX({date_field}) as max_date,
            COUNT(*) as total_records,
            COUNT(DISTINCT DATE({date_field})) as unique_days
        FROM {table}
        WHERE {date_field} IS NOT NULL
        """
        
        try:
            result = db_manager.execute_query(query)
            if result and result[0]:
                row = result[0]
                logger.info(f"\n{description} ({table}):")
                logger.info(f"  Earliest Date: {row['min_date']}")
                logger.info(f"  Latest Date: {row['max_date']}")
                logger.info(f"  Total Records: {row['total_records']:,}")
                logger.info(f"  Unique Days: {row['unique_days']:,}")
                
                # Check recent data
                query_recent = f"""
                SELECT COUNT(*) as recent_count
                FROM {table}
                WHERE {date_field} >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
                """
                recent = db_manager.execute_query(query_recent)
                if recent:
                    logger.info(f"  Records in last 90 days: {recent[0]['recent_count']:,}")
                    
        except Exception as e:
            logger.error(f"Error checking {table}: {e}")


def check_job_order_consumption_dates():
    """Check specifically for job order consumption dates."""
    
    query = """
    SELECT 
        MIN(jt.TxnDate_dd) as min_date,
        MAX(jt.TxnDate_dd) as max_date,
        COUNT(*) as total_transactions,
        COUNT(DISTINCT ji.ItemId_i) as unique_parts,
        SUM(ji.Qty_d) as total_quantity
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    WHERE 
        ji.InOut_c = 'I'
        AND jt.Void_c = 'N'
        AND ji.Void_c = 'N'
    """
    
    try:
        result = db_manager.execute_query(query)
        if result and result[0]:
            row = result[0]
            logger.info("\n" + "=" * 80)
            logger.info("JOB ORDER CONSUMPTION (Parts Used in Manufacturing):")
            logger.info(f"  Earliest Date: {row['min_date']}")
            logger.info(f"  Latest Date: {row['max_date']}")
            logger.info(f"  Total Transactions: {row['total_transactions']:,}")
            logger.info(f"  Unique Parts: {row['unique_parts']:,}")
            logger.info(f"  Total Quantity Consumed: {row['total_quantity']:,.0f}")
            
            # Check monthly breakdown
            query_monthly = """
            SELECT 
                DATE_FORMAT(jt.TxnDate_dd, '%Y-%m') as month,
                COUNT(*) as transactions,
                COUNT(DISTINCT ji.ItemId_i) as unique_parts,
                SUM(ji.Qty_d) as total_consumed
            FROM tbl_jo_item ji
            JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
            WHERE 
                ji.InOut_c = 'I'
                AND jt.Void_c = 'N'
                AND ji.Void_c = 'N'
            GROUP BY DATE_FORMAT(jt.TxnDate_dd, '%Y-%m')
            ORDER BY month DESC
            LIMIT 12
            """
            
            monthly = db_manager.execute_query(query_monthly)
            if monthly:
                logger.info("\n  Monthly Breakdown (Last 12 months of data):")
                for m in monthly:
                    logger.info(f"    {m['month']}: {m['transactions']:5} transactions, "
                              f"{m['unique_parts']:4} parts, "
                              f"{m['total_consumed']:10.0f} units consumed")
                              
    except Exception as e:
        logger.error(f"Error checking job order consumption: {e}")


def get_data_recommendation():
    """Provide recommendation based on data availability."""
    
    # Get the latest date
    query = """
    SELECT MAX(TxnDate_dd) as latest_date
    FROM tbl_jo_txn
    """
    
    try:
        result = db_manager.execute_query(query)
        if result and result[0]:
            latest_date = pd.to_datetime(result[0]['latest_date'])
            current_date = pd.Timestamp.now()
            days_old = (current_date - latest_date).days
            
            logger.info("\n" + "=" * 80)
            logger.info("DATA ANALYSIS RECOMMENDATION:")
            logger.info(f"  Latest data is from: {latest_date.date()}")
            logger.info(f"  Data is {days_old} days old")
            
            if days_old > 90:
                logger.info("\n  RECOMMENDATION:")
                logger.info("  - Data is older than 90 days")
                logger.info("  - Adjust date ranges in queries to match available data")
                logger.info("  - Consider using historical data for model development")
                logger.info("  - Use the full available date range for analysis")
            else:
                logger.info("\n  Data is recent enough for real-time forecasting")
                
    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")


if __name__ == "__main__":
    check_date_ranges()
    check_job_order_consumption_dates()
    get_data_recommendation()