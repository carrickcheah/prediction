#!/usr/bin/env python3
"""
Check the actual values in tbl_jo_item to understand the data structure.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.database import db_manager
from utils.logger import setup_logger

logger = setup_logger("check_jo_item")


def check_inout_values():
    """Check what values exist in the InOut_c field."""
    
    query = """
    SELECT 
        InOut_c,
        COUNT(*) as count,
        SUM(Qty_d) as total_qty,
        MIN(Qty_d) as min_qty,
        MAX(Qty_d) as max_qty,
        AVG(Qty_d) as avg_qty
    FROM tbl_jo_item
    GROUP BY InOut_c
    """
    
    logger.info("Checking InOut_c values in tbl_jo_item:")
    logger.info("-" * 60)
    
    try:
        results = db_manager.execute_query(query)
        if results:
            for row in results:
                logger.info(f"InOut_c = '{row['InOut_c'] if row['InOut_c'] else 'NULL'}':")
                logger.info(f"  Count: {row['count']:,}")
                logger.info(f"  Total Qty: {row['total_qty']:,.2f}")
                logger.info(f"  Avg Qty: {row['avg_qty']:.2f}")
                logger.info(f"  Min Qty: {row['min_qty']:.2f}")
                logger.info(f"  Max Qty: {row['max_qty']:.2f}")
                logger.info("")
        else:
            logger.warning("No data found in tbl_jo_item")
            
    except Exception as e:
        logger.error(f"Error checking InOut_c values: {e}")


def check_sample_jo_items():
    """Get sample records from tbl_jo_item."""
    
    query = """
    SELECT 
        ji.*,
        jt.TxnDate_dd,
        jt.DocRef_v
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    ORDER BY jt.TxnDate_dd DESC
    LIMIT 10
    """
    
    logger.info("\nSample Job Order Items (Latest 10):")
    logger.info("-" * 60)
    
    try:
        results = db_manager.execute_query(query)
        if results:
            for i, row in enumerate(results, 1):
                logger.info(f"{i}. Date: {row['TxnDate_dd']}, "
                          f"ItemId: {row['ItemId_i']}, "
                          f"Qty: {row['Qty_d']:.2f}, "
                          f"InOut: '{row['InOut_c']}', "
                          f"DocRef: {row['DocRef_v']}")
        else:
            logger.warning("No job order items found")
            
    except Exception as e:
        logger.error(f"Error getting sample items: {e}")


def check_void_status():
    """Check Void_c values in job order items."""
    
    query = """
    SELECT 
        ji.Void_c as item_void,
        jt.Void_c as txn_void,
        COUNT(*) as count
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    GROUP BY ji.Void_c, jt.Void_c
    """
    
    logger.info("\nVoid Status Distribution:")
    logger.info("-" * 60)
    
    try:
        results = db_manager.execute_query(query)
        if results:
            for row in results:
                logger.info(f"Item Void: '{row['item_void']}', "
                          f"Txn Void: '{row['txn_void']}', "
                          f"Count: {row['count']:,}")
        else:
            logger.warning("No void status data found")
            
    except Exception as e:
        logger.error(f"Error checking void status: {e}")


def check_consumption_pattern():
    """Check consumption pattern without InOut_c filter."""
    
    query = """
    SELECT 
        DATE_FORMAT(jt.TxnDate_dd, '%Y-%m') as month,
        COUNT(DISTINCT ji.ItemId_i) as unique_parts,
        COUNT(*) as transactions,
        SUM(ji.Qty_d) as total_qty
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    WHERE jt.TxnDate_dd >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY DATE_FORMAT(jt.TxnDate_dd, '%Y-%m')
    ORDER BY month DESC
    """
    
    logger.info("\nMonthly Job Order Pattern (Last 6 Months):")
    logger.info("-" * 60)
    
    try:
        results = db_manager.execute_query(query)
        if results:
            for row in results:
                logger.info(f"{row['month']}: "
                          f"{row['unique_parts']:4} parts, "
                          f"{row['transactions']:5} transactions, "
                          f"Total Qty: {row['total_qty']:10.2f}")
        else:
            logger.warning("No monthly pattern data found")
            
    except Exception as e:
        logger.error(f"Error checking consumption pattern: {e}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CHECKING JOB ORDER ITEM DATA STRUCTURE")
    logger.info("=" * 60)
    
    check_inout_values()
    check_sample_jo_items()
    check_void_status()
    check_consumption_pattern()