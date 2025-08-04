#!/usr/bin/env python
"""
Create a comprehensive data quality report for inventory forecasting.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.extractors.sales_extractor import SalesExtractor
from data.extractors.purchase_extractor import PurchaseExtractor
from data.extractors.job_order_extractor import JobOrderExtractor


def create_data_quality_report():
    """Create a comprehensive data quality report."""
    # Set date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)
    
    # Extract data
    print("Extracting data...")
    sales_extractor = SalesExtractor()
    sales_data = sales_extractor.extract(start_date, end_date)
    
    purchase_extractor = PurchaseExtractor()
    purchase_data = purchase_extractor.extract(start_date, end_date)
    
    job_extractor = JobOrderExtractor()
    consumption_data = job_extractor.extract(start_date, end_date)
    
    # Create report
    report = []
    report.append("# INVENTORY FORECASTING - DATA QUALITY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Date Range: {start_date} to {end_date}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"- **Sales Orders**: {len(sales_data):,} records for {sales_data['item_id'].nunique()} unique items")
    report.append(f"- **Purchase Orders**: {len(purchase_data):,} records for {purchase_data['item_id'].nunique()} unique items")
    report.append(f"- **Manufacturing Consumption**: {len(consumption_data):,} records for {consumption_data['item_id'].nunique()} unique items")
    report.append("")
    
    # Data Coverage Analysis
    report.append("## Data Coverage Analysis")
    report.append("")
    
    # Date coverage for each dataset
    for name, data in [("Sales", sales_data), ("Purchases", purchase_data), ("Consumption", consumption_data)]:
        if not data.empty:
            # Handle different date column names
            date_col = 'date' if 'date' in data.columns else 'order_date'
            date_min = data[date_col].min()
            date_max = data[date_col].max()
            date_range = pd.date_range(date_min, date_max, freq='D')
            unique_dates = data[date_col].unique()
            coverage = len(unique_dates) / len(date_range) * 100
            
            report.append(f"### {name} Data")
            report.append(f"- Date range: {date_min} to {date_max}")
            report.append(f"- Days with data: {len(unique_dates)} out of {len(date_range)} ({coverage:.1f}% coverage)")
            report.append(f"- Average records per day: {len(data) / len(unique_dates):.1f}")
            report.append("")
    
    # Top Items Analysis
    report.append("## Top 20 Items by Consumption")
    report.append("")
    
    if not consumption_data.empty:
        top_items = consumption_data.groupby('item_id')['consumption'].sum().sort_values(ascending=False).head(20)
        report.append("| Rank | Item ID | Total Consumption | % of Total |")
        report.append("|------|---------|------------------|------------|")
        
        total_consumption = consumption_data['consumption'].sum()
        for i, (item_id, qty) in enumerate(top_items.items(), 1):
            pct = qty / total_consumption * 100
            report.append(f"| {i} | {item_id} | {qty:,.0f} | {pct:.1f}% |")
    
    report.append("")
    
    # Data Quality Issues
    report.append("## Data Quality Issues")
    report.append("")
    
    # Check for missing weekends
    if not consumption_data.empty:
        consumption_by_dow = consumption_data.groupby(consumption_data['date'].dt.dayofweek).size()
        weekend_data = consumption_by_dow.get(5, 0) + consumption_by_dow.get(6, 0)
        weekday_avg = consumption_by_dow[0:5].mean()
        
        report.append("### Weekend Data")
        report.append(f"- Weekend records: {weekend_data}")
        report.append(f"- Weekday average: {weekday_avg:.0f}")
        report.append(f"- Weekend coverage: {'Present' if weekend_data > 0 else 'Missing (factory closed on weekends)'}")
        report.append("")
    
    # Intermittent demand analysis
    report.append("### Intermittent Demand Analysis")
    if not consumption_data.empty:
        # Group by item and count days with demand
        item_demand_days = consumption_data.groupby('item_id')['date'].nunique()
        total_days = (consumption_data['date'].max() - consumption_data['date'].min()).days + 1
        
        intermittent_items = item_demand_days[item_demand_days < total_days * 0.3].count()
        report.append(f"- Items with demand on <30% of days: {intermittent_items} ({intermittent_items/len(item_demand_days)*100:.1f}%)")
        report.append(f"- This indicates many items have intermittent/sporadic demand patterns")
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("1. **Data Coverage**: Consider extracting more historical data (365 days) for better seasonality detection")
    report.append("2. **Weekend Handling**: Implement special handling for weekends when factory is closed")
    report.append("3. **Intermittent Demand**: Use specialized models for items with sporadic demand")
    report.append("4. **Lead Time Analysis**: Analyze purchase order lead times by supplier for better procurement planning")
    report.append("5. **Top Items Focus**: Start with top 20 items that represent significant consumption volume")
    
    # Save report
    report_path = Path("/Users/carrickcheah/Project/prediction/outputs/daily_reports/data_quality_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to: {report_path}")
    
    # Create visualizations
    create_visualizations(consumption_data, sales_data, purchase_data)


def create_visualizations(consumption_data, sales_data, purchase_data):
    """Create data visualization charts."""
    output_dir = Path("/Users/carrickcheah/Project/prediction/outputs/daily_reports")
    
    # Get date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Daily consumption trend
    if not consumption_data.empty:
        plt.figure(figsize=(12, 6))
        daily_consumption = consumption_data.groupby('date')['consumption'].sum()
        plt.plot(daily_consumption.index, daily_consumption.values, linewidth=2)
        plt.title('Daily Total Consumption Trend', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Total Consumption')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'daily_consumption_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Top 10 items consumption
    if not consumption_data.empty:
        plt.figure(figsize=(10, 6))
        top_items = consumption_data.groupby('item_id')['consumption'].sum().sort_values(ascending=False).head(10)
        plt.bar(range(len(top_items)), top_items.values)
        plt.xticks(range(len(top_items)), [f"Item\n{id}" for id in top_items.index], rotation=45)
        plt.title('Top 10 Items by Total Consumption', fontsize=14)
        plt.xlabel('Item ID')
        plt.ylabel('Total Consumption')
        plt.tight_layout()
        plt.savefig(output_dir / 'top_10_items.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Data coverage heatmap
    if not consumption_data.empty:
        plt.figure(figsize=(12, 4))
        
        # Create a matrix of dates x data source
        date_range = pd.date_range(start_date, end_date, freq='D')
        coverage_matrix = pd.DataFrame(index=date_range, columns=['Sales', 'Purchases', 'Consumption'])
        
        coverage_matrix['Sales'] = coverage_matrix.index.isin(sales_data['date'].unique()).astype(int)
        coverage_matrix['Purchases'] = coverage_matrix.index.isin(purchase_data['order_date'].unique()).astype(int)
        coverage_matrix['Consumption'] = coverage_matrix.index.isin(consumption_data['date'].unique()).astype(int)
        
        # Resample to weekly for better visualization
        weekly_coverage = coverage_matrix.resample('W').mean()
        
        sns.heatmap(weekly_coverage.T, cmap='RdYlGn', vmin=0, vmax=1, 
                    cbar_kws={'label': 'Data Coverage'}, 
                    xticklabels=[d.strftime('%m-%d') for d in weekly_coverage.index])
        plt.title('Weekly Data Coverage by Source', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'data_coverage_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizations saved to outputs/daily_reports/")


if __name__ == "__main__":
    create_data_quality_report()