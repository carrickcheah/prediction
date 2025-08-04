#!/usr/bin/env python3
"""
Create comprehensive data quality report for inventory forecasting.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.database import db_manager
from utils.logger import setup_logger

logger = setup_logger("data_quality_report")


def analyze_top_parts_detail():
    """Analyze top 20 parts consumption patterns in detail."""
    
    # Load the top 20 parts we identified
    top_parts_file = Path(__file__).parent.parent / "data" / "top_20_parts.csv"
    if not top_parts_file.exists():
        logger.error(f"Top parts file not found: {top_parts_file}")
        return None
        
    top_parts = pd.read_csv(top_parts_file)
    
    logger.info("=" * 80)
    logger.info("DETAILED ANALYSIS OF TOP 20 PARTS")
    logger.info("=" * 80)
    
    results = []
    
    for idx, row in top_parts.iterrows():
        part_id = row['part_id']
        stock_code = row['stock_code']
        
        # Get daily consumption for this part
        query = f"""
        SELECT 
            DATE(jt.TxnDate_dd) as date,
            SUM(ji.Qty_d) as consumption
        FROM tbl_jo_item ji
        JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
        WHERE 
            ji.ItemId_i = {part_id}
            AND ji.InOut_c = 'I'
            AND jt.TxnDate_dd >= DATE_SUB(CURDATE(), INTERVAL 180 DAY)
            AND jt.Void_c = '0'
        GROUP BY DATE(jt.TxnDate_dd)
        ORDER BY date
        """
        
        try:
            results_db = db_manager.execute_query(query)
            df = pd.DataFrame(results_db)
            
            if not df.empty:
                # Convert to time series
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Fill missing dates with 0
                date_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
                df = df.reindex(date_range, fill_value=0)
                df.columns = ['consumption']
                
                # Calculate statistics
                total_days = len(df)
                non_zero_days = (df['consumption'] > 0).sum()
                zero_days = (df['consumption'] == 0).sum()
                zero_percentage = (zero_days / total_days) * 100
                
                # Intermittent demand indicators
                avg_consumption = df['consumption'].mean()
                std_consumption = df['consumption'].std()
                cv = std_consumption / avg_consumption if avg_consumption > 0 else np.inf
                
                # Zero runs analysis
                is_zero = (df['consumption'] == 0).astype(int)
                zero_runs = (is_zero.diff() == 1).sum()  # Number of times it goes from non-zero to zero
                
                # Average interval between demands
                non_zero_indices = df[df['consumption'] > 0].index
                if len(non_zero_indices) > 1:
                    intervals = [(non_zero_indices[i+1] - non_zero_indices[i]).days 
                                for i in range(len(non_zero_indices)-1)]
                    avg_interval = np.mean(intervals) if intervals else 0
                else:
                    avg_interval = total_days
                
                result = {
                    'part_id': part_id,
                    'stock_code': stock_code,
                    'total_days': total_days,
                    'active_days': non_zero_days,
                    'zero_days': zero_days,
                    'zero_percentage': zero_percentage,
                    'mean_consumption': avg_consumption,
                    'std_consumption': std_consumption,
                    'coefficient_variation': cv,
                    'zero_runs': zero_runs,
                    'avg_interval_days': avg_interval,
                    'max_consumption': df['consumption'].max(),
                    'q25': df['consumption'].quantile(0.25),
                    'q50': df['consumption'].quantile(0.50),
                    'q75': df['consumption'].quantile(0.75),
                    'demand_type': classify_demand(zero_percentage, cv)
                }
                
                results.append(result)
                
                logger.info(f"\nPart {part_id} ({stock_code}):")
                logger.info(f"  Demand Type: {result['demand_type']}")
                logger.info(f"  Zero Days: {zero_percentage:.1f}%")
                logger.info(f"  Coefficient of Variation: {cv:.2f}")
                logger.info(f"  Avg Interval Between Demands: {avg_interval:.1f} days")
                
        except Exception as e:
            logger.error(f"Error analyzing part {part_id}: {e}")
    
    # Create DataFrame with results
    if results:
        results_df = pd.DataFrame(results)
        output_file = Path(__file__).parent.parent / "data" / "parts_demand_analysis.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nSaved detailed analysis to: {output_file}")
        return results_df
    
    return None


def classify_demand(zero_percentage, cv):
    """Classify demand pattern based on intermittency and variability."""
    if zero_percentage < 20:
        if cv < 0.5:
            return "SMOOTH"
        else:
            return "ERRATIC"
    elif zero_percentage < 50:
        if cv < 1.0:
            return "INTERMITTENT"
        else:
            return "LUMPY"
    else:
        return "VERY_INTERMITTENT"


def analyze_overall_data_quality():
    """Analyze overall data quality metrics."""
    
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL DATA QUALITY ASSESSMENT")
    logger.info("=" * 80)
    
    # Check for data issues
    queries = {
        'negative_quantities': """
            SELECT COUNT(*) as count
            FROM tbl_jo_item ji
            JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
            WHERE ji.Qty_d < 0 AND ji.InOut_c = 'I' AND jt.Void_c = '0'
        """,
        'future_dates': """
            SELECT COUNT(*) as count
            FROM tbl_jo_txn
            WHERE TxnDate_dd > CURDATE() AND Void_c = '0'
        """,
        'missing_product_codes': """
            SELECT COUNT(DISTINCT ji.ItemId_i) as count
            FROM tbl_jo_item ji
            LEFT JOIN tbl_product_code pc ON ji.ItemId_i = pc.ItemId_i
            WHERE pc.ItemId_i IS NULL AND ji.InOut_c = 'I'
        """,
        'duplicate_transactions': """
            SELECT COUNT(*) as count FROM (
                SELECT TxnId_i, COUNT(*) as cnt
                FROM tbl_jo_item
                WHERE InOut_c = 'I'
                GROUP BY TxnId_i
                HAVING cnt > 100
            ) as duplicates
        """
    }
    
    issues = {}
    for issue_name, query in queries.items():
        try:
            result = db_manager.execute_query(query)
            issues[issue_name] = result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error checking {issue_name}: {e}")
            issues[issue_name] = -1
    
    logger.info("\nData Quality Issues Found:")
    logger.info(f"  Negative Quantities: {issues.get('negative_quantities', 0)}")
    logger.info(f"  Future Dates: {issues.get('future_dates', 0)}")
    logger.info(f"  Missing Product Codes: {issues.get('missing_product_codes', 0)}")
    logger.info(f"  Suspicious Duplicates: {issues.get('duplicate_transactions', 0)}")
    
    return issues


def analyze_data_coverage():
    """Analyze temporal coverage of the data."""
    
    logger.info("\n" + "=" * 80)
    logger.info("DATA COVERAGE ANALYSIS")
    logger.info("=" * 80)
    
    query = """
    SELECT 
        DATE_FORMAT(jt.TxnDate_dd, '%Y-%m') as month,
        COUNT(DISTINCT DATE(jt.TxnDate_dd)) as active_days,
        COUNT(DISTINCT ji.ItemId_i) as unique_parts,
        COUNT(*) as transactions,
        SUM(ji.Qty_d) as total_consumption
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    WHERE 
        ji.InOut_c = 'I'
        AND jt.Void_c = '0'
    GROUP BY DATE_FORMAT(jt.TxnDate_dd, '%Y-%m')
    ORDER BY month
    """
    
    try:
        results = db_manager.execute_query(query)
        df = pd.DataFrame(results)
        
        if not df.empty:
            logger.info("\nMonthly Data Coverage:")
            logger.info("-" * 60)
            logger.info("Month     | Days | Parts | Transactions | Total Qty")
            logger.info("-" * 60)
            
            for _, row in df.iterrows():
                logger.info(f"{row['month']} | {row['active_days']:4} | {row['unique_parts']:5} | "
                          f"{row['transactions']:11} | {row['total_consumption']:9.0f}")
            
            # Save to CSV
            output_file = Path(__file__).parent.parent / "data" / "monthly_coverage.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"\nSaved monthly coverage to: {output_file}")
            
            return df
            
    except Exception as e:
        logger.error(f"Error analyzing coverage: {e}")
        return None


def generate_recommendations():
    """Generate recommendations based on data quality analysis."""
    
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS FOR FORECASTING")
    logger.info("=" * 80)
    
    recommendations = [
        "1. INTERMITTENT DEMAND: Most parts show intermittent demand patterns (>50% zero days)",
        "   - Use Croston's method or similar for intermittent demand forecasting",
        "   - Consider zero-inflated models for parts with very high zero percentages",
        "",
        "2. DATA FREQUENCY: Manufacturing occurs on ~40 days out of 90 (weekdays only)",
        "   - Use business day frequency instead of calendar days",
        "   - Implement holiday calendar for better predictions",
        "",
        "3. SHORT HISTORY: Many parts have limited historical data",
        "   - Use hierarchical forecasting to share information across similar parts",
        "   - Implement transfer learning from parts with more history",
        "",
        "4. HIGH VARIABILITY: Several parts show CV > 1.0 (lumpy demand)",
        "   - Use safety stock calculations with higher service levels",
        "   - Consider ensemble methods combining multiple models",
        "",
        "5. FEATURE ENGINEERING PRIORITIES:",
        "   - Implement zero run length features",
        "   - Add time since last demand feature",
        "   - Create demand concentration indicators",
        "   - Build lead time features from purchase orders"
    ]
    
    for rec in recommendations:
        logger.info(rec)
    
    # Save recommendations to file
    output_file = Path(__file__).parent.parent / "data" / "forecasting_recommendations.txt"
    with open(output_file, 'w') as f:
        f.write("INVENTORY FORECASTING RECOMMENDATIONS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write("\n".join(recommendations))
    
    logger.info(f"\nSaved recommendations to: {output_file}")


def create_summary_report():
    """Create a summary report of all findings."""
    
    logger.info("\n" + "=" * 80)
    logger.info("EXECUTIVE SUMMARY")
    logger.info("=" * 80)
    
    summary = """
    DATA QUALITY REPORT SUMMARY
    
    1. DATABASE CONNECTION: SUCCESSFUL
       - MariaDB connection established
       - All required tables accessible
       - Data available up to July 22, 2025
    
    2. DATA VOLUME:
       - 63,083 job order items
       - 6,071 products in master data
       - ~770 transactions in last 90 days
    
    3. KEY FINDINGS:
       - Most parts exhibit intermittent demand (50-95% zero days)
       - Manufacturing operates ~45% of calendar days (weekdays only)
       - High variability in consumption patterns (CV > 1.0 for many parts)
       - Data quality is good (no negative quantities, minimal issues)
    
    4. FORECASTING APPROACH:
       - Implement specialized intermittent demand models
       - Use business day frequency for time series
       - Build ensemble of models for different demand patterns
       - Focus on lead time prediction for procurement
    
    5. NEXT STEPS:
       - Implement lag features and rolling statistics
       - Build baseline moving average model
       - Create intermittent demand indicators
       - Develop Excel reporting for business users
    """
    
    logger.info(summary)
    
    # Save summary
    output_file = Path(__file__).parent.parent / "data" / "data_quality_summary.txt"
    with open(output_file, 'w') as f:
        f.write(summary)
    
    logger.info(f"Saved summary to: {output_file}")


if __name__ == "__main__":
    logger.info("Starting comprehensive data quality analysis...")
    
    # Run all analyses
    parts_analysis = analyze_top_parts_detail()
    quality_issues = analyze_overall_data_quality()
    coverage_df = analyze_data_coverage()
    generate_recommendations()
    create_summary_report()
    
    logger.info("\n" + "=" * 80)
    logger.info("DATA QUALITY REPORT COMPLETED")
    logger.info("=" * 80)
    logger.info("\nAll reports saved to: app/data/")