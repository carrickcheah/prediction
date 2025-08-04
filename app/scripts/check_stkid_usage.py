#!/usr/bin/env python
"""
Check StkId_i usage and relationship with ItemId_i.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from config.database import DatabaseManager

def check_stkid_usage():
    """Check how StkId_i relates to ItemId_i."""
    db = DatabaseManager()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Check relationship between StkId_i and ItemId_i in job orders
        print("Checking StkId_i vs ItemId_i in job orders...")
        cursor.execute("""
            SELECT 
                ji.ItemId_i,
                ji.StkId_i,
                COUNT(*) as count,
                SUM(ji.Qty_d) as total_qty
            FROM tbl_jo_item ji
            JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
            WHERE 
                ji.InOut_c = 'I'
                AND jt.TxnDate_dd >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                AND jt.Void_c = '0'
                AND ji.Void_c = '0'
                AND ji.ItemId_i IN (1000045, 1000059, 1003053, 1003270, 1003271)
            GROUP BY ji.ItemId_i, ji.StkId_i
            ORDER BY ji.ItemId_i
        """)
        
        print("\nRelationship for our test items:")
        print("ItemId_i -> StkId_i (count, total_qty)")
        for row in cursor.fetchall():
            print(f"  {row[0]} -> {row[1]} ({row[2]} records, {row[3]} qty)")
        
        # Check if StkId_i matches ItemId_i in product code table
        print("\n\nChecking tbl_product_code StkId_i column...")
        cursor.execute("""
            SELECT StkId_i, ItemId_i, StkCode_v, ProdName_v
            FROM tbl_product_code
            WHERE StkId_i IN (1000045, 1000059, 1003053, 1003270, 1003271)
            OR ItemId_i IN (1000045, 1000059, 1003053, 1003270, 1003271)
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        if results:
            print("\nProduct code entries:")
            for row in results:
                print(f"  StkId_i: {row[0]}, ItemId_i: {row[1]}, StkCode_v: {row[2]}, Name: {row[3]}")
        
        # Check top consumed items by StkId_i
        print("\n\nTop consumed items by StkId_i in last 365 days:")
        cursor.execute("""
            SELECT 
                ji.StkId_i,
                SUM(ji.Qty_d) as total_consumption,
                COUNT(DISTINCT DATE(jt.TxnDate_dd)) as days_with_consumption,
                COUNT(DISTINCT ji.ItemId_i) as unique_itemids
            FROM tbl_jo_item ji
            JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
            WHERE 
                ji.InOut_c = 'I'
                AND jt.TxnDate_dd >= DATE_SUB(NOW(), INTERVAL 365 DAY)
                AND jt.Void_c = '0'
                AND ji.Void_c = '0'
            GROUP BY ji.StkId_i
            ORDER BY total_consumption DESC
            LIMIT 20
        """)
        
        for row in cursor.fetchall():
            print(f"  StkId_i: {row[0]}, Total: {row[1]}, Days: {row[2]}, Unique ItemIds: {row[3]}")
        
        cursor.close()

if __name__ == "__main__":
    check_stkid_usage()