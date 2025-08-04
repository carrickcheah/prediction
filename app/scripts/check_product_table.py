#!/usr/bin/env python
"""
Check tbl_product_code structure and ItemId_i column.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from config.database import DatabaseManager

def check_product_table():
    """Check product table structure."""
    db = DatabaseManager()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Check tbl_product_code structure
        cursor.execute("DESCRIBE tbl_product_code")
        print("tbl_product_code structure:")
        columns = cursor.fetchall()
        for row in columns:
            print(f"  {row[0]} - {row[1]}")
        
        # Check if ItemId_i exists and get sample data
        cursor.execute("SELECT ItemId_i, StkCode_v, ProdName_v FROM tbl_product_code WHERE Deleted_c = 'N' LIMIT 10")
        print("\nSample data:")
        for row in cursor.fetchall():
            print(f"  ItemId_i: {row[0]}, StkCode_v: {row[1]}, ProdName_v: {row[2]}")
        
        # Check how ItemId_i relates to our current item codes
        print("\nChecking relationship with current item codes...")
        cursor.execute("""
            SELECT DISTINCT p.ItemId_i, p.StkCode_v, p.ProdName_v
            FROM tbl_product_code p
            WHERE p.StkCode_v IN ('1000045', '1000059', '1003053', '1003270', '1003271')
            AND p.Deleted_c = 'N'
        """)
        
        print("\nMapping of our test items:")
        for row in cursor.fetchall():
            print(f"  StkCode_v: {row[1]} -> ItemId_i: {row[0]} ({row[2]})")
        
        cursor.close()

if __name__ == "__main__":
    check_product_table()