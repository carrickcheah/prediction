#!/usr/bin/env python
"""
Check how item IDs are currently being used.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from config.database import DatabaseManager

def check_item_usage():
    """Check how items are identified in the current data."""
    db = DatabaseManager()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Check tbl_jo_item structure
        print("Checking tbl_jo_item structure...")
        cursor.execute("DESCRIBE tbl_jo_item")
        print("\nColumns in tbl_jo_item:")
        for row in cursor.fetchall():
            print(f"  {row[0]} - {row[1]}")
        
        # Check what ItemId_i values exist in job orders
        print("\n\nTop consumed items by ItemId_i in last 30 days:")
        cursor.execute("""
            SELECT 
                ji.ItemId_i,
                SUM(ji.Qty_d) as total_consumption,
                COUNT(DISTINCT DATE(jt.TxnDate_dd)) as days_with_consumption
            FROM tbl_jo_item ji
            JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
            WHERE 
                ji.InOut_c = 'I'
                AND jt.TxnDate_dd >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                AND jt.Void_c = '0'
                AND ji.Void_c = '0'
            GROUP BY ji.ItemId_i
            ORDER BY total_consumption DESC
            LIMIT 10
        """)
        
        top_items = []
        for row in cursor.fetchall():
            top_items.append(row[0])
            print(f"  ItemId_i: {row[0]}, Total: {row[1]}, Days: {row[2]}")
        
        # Check if these ItemId_i values exist in product code table
        if top_items:
            print("\n\nChecking product info for top items:")
            placeholders = ','.join(['%s'] * len(top_items[:5]))
            cursor.execute(f"""
                SELECT ItemId_i, StkCode_v, ProdName_v
                FROM tbl_product_code
                WHERE ItemId_i IN ({placeholders})
                AND Deleted_c = 'N'
            """, top_items[:5])
            
            for row in cursor.fetchall():
                print(f"  ItemId_i: {row[0]}, StkCode_v: {row[1]}, Name: {row[2]}")
        
        # Check our current test items - are they ItemId_i or StkCode_v?
        print("\n\nChecking what our test items (1000045, etc) actually are...")
        test_items = ['1000045', '1000059', '1003053', '1003270', '1003271']
        
        # First check if they're ItemId_i values
        placeholders = ','.join(['%s'] * len(test_items))
        cursor.execute(f"""
            SELECT 
                ji.ItemId_i,
                SUM(ji.Qty_d) as consumption
            FROM tbl_jo_item ji
            JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
            WHERE 
                ji.ItemId_i IN ({placeholders})
                AND ji.InOut_c = 'I'
                AND jt.TxnDate_dd >= DATE_SUB(NOW(), INTERVAL 365 DAY)
                AND jt.Void_c = '0'
                AND ji.Void_c = '0'
            GROUP BY ji.ItemId_i
        """, [int(x) for x in test_items])
        
        results = cursor.fetchall()
        if results:
            print("\nThey appear to be ItemId_i values:")
            for row in results:
                print(f"  ItemId_i: {row[0]}, Consumption: {row[1]}")
        else:
            print("\nThey don't appear to be ItemId_i values in job orders.")
        
        cursor.close()

if __name__ == "__main__":
    check_item_usage()