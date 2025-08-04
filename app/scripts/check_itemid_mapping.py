#!/usr/bin/env python
"""
Check ItemId_i mapping between tables.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from config.database import DatabaseManager

def check_itemid_mapping():
    """Check how ItemId_i is used across tables."""
    db = DatabaseManager()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Check job order details - what column stores the item?
        print("Checking tbl_job_order_details structure...")
        cursor.execute("DESCRIBE tbl_job_order_details")
        print("\nRelevant columns in job order details:")
        for row in cursor.fetchall():
            col_name = row[0]
            if 'item' in col_name.lower() or 'stk' in col_name.lower() or 'prod' in col_name.lower():
                print(f"  {row[0]} - {row[1]}")
        
        # Check what item identifiers are in job orders
        print("\nSample job order details:")
        cursor.execute("""
            SELECT jo_item_code, COUNT(*) as count
            FROM tbl_job_order_details
            WHERE order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY jo_item_code
            ORDER BY count DESC
            LIMIT 10
        """)
        for row in cursor.fetchall():
            print(f"  jo_item_code: {row[0]}, count: {row[1]}")
        
        # Check if these codes exist in product table
        print("\nChecking if job order items exist in product table...")
        cursor.execute("""
            SELECT DISTINCT jod.jo_item_code, pc.ItemId_i, pc.ProdName_v
            FROM tbl_job_order_details jod
            LEFT JOIN tbl_product_code pc ON jod.jo_item_code = pc.StkCode_v
            WHERE jod.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            AND jod.jo_item_code IN ('1000045', '1000059', '1003053')
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        if results:
            print("\nMatches found:")
            for row in results:
                print(f"  jo_item_code: {row[0]}, ItemId_i: {row[1]}, ProdName: {row[2]}")
        else:
            print("\nNo matches found. Item codes might be stored differently.")
        
        # Check sales order details structure
        print("\n\nChecking sales order structure...")
        cursor.execute("DESCRIBE tbl_sales_order_details")
        print("\nRelevant columns in sales order details:")
        for row in cursor.fetchall():
            col_name = row[0]
            if 'item' in col_name.lower() or 'stk' in col_name.lower() or 'prod' in col_name.lower():
                print(f"  {row[0]} - {row[1]}")
        
        # Check purchase order details structure
        print("\n\nChecking purchase order structure...")
        cursor.execute("DESCRIBE tbl_purchase_order_details")
        print("\nRelevant columns in purchase order details:")
        for row in cursor.fetchall():
            col_name = row[0]
            if 'item' in col_name.lower() or 'stk' in col_name.lower() or 'prod' in col_name.lower():
                print(f"  {row[0]} - {row[1]}")
        
        cursor.close()

if __name__ == "__main__":
    check_itemid_mapping()