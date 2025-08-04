#!/usr/bin/env python
"""
Analyze intermittent demand patterns and recommend appropriate forecasting methods.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

logger = setup_logger("intermittent_analysis")


def calculate_intermittency_metrics(series: pd.Series) -> dict:
    """Calculate metrics for intermittent demand characterization."""
    
    # Basic stats
    non_zero = series[series > 0]
    zero_periods = (series == 0).sum()
    total_periods = len(series)
    
    # ADI (Average Demand Interval) - average time between demands
    demand_indices = np.where(series > 0)[0]
    if len(demand_indices) > 1:
        intervals = np.diff(demand_indices)
        adi = np.mean(intervals)
    else:
        adi = total_periods
    
    # CV² (Coefficient of Variation squared) - variability of non-zero demand
    if len(non_zero) > 0:
        cv2 = (non_zero.std() / non_zero.mean()) ** 2 if non_zero.mean() > 0 else 0
    else:
        cv2 = 0
    
    return {
        'total_demand': series.sum(),
        'periods_with_demand': len(non_zero),
        'periods_without_demand': zero_periods,
        'demand_percentage': len(non_zero) / total_periods * 100,
        'adi': adi,
        'cv2': cv2,
        'mean_demand_when_positive': non_zero.mean() if len(non_zero) > 0 else 0,
        'max_demand': series.max(),
        'demand_pattern': classify_demand_pattern(adi, cv2)
    }


def classify_demand_pattern(adi: float, cv2: float) -> str:
    """Classify demand pattern based on Syntetos-Boylan categorization."""
    
    if adi < 1.32 and cv2 < 0.49:
        return "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        return "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        return "Erratic"
    else:
        return "Lumpy"


def analyze_all_items():
    """Analyze intermittency patterns for all top 20 items."""
    
    # Load data
    data_path = Path("/Users/carrickcheah/Project/prediction/data/raw/top_20_consumption_pivot_365days.csv")
    consumption_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Analyze each item
    results = []
    
    for item_id in consumption_data.columns:
        series = consumption_data[item_id]
        metrics = calculate_intermittency_metrics(series)
        metrics['item_id'] = item_id
        results.append(metrics)
    
    # Create DataFrame
    analysis_df = pd.DataFrame(results)
    analysis_df = analysis_df.sort_values('total_demand', ascending=False)
    
    # Save results
    output_dir = Path("/Users/carrickcheah/Project/prediction/outputs/intermittent_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_dir / "intermittency_analysis.csv", index=False)
    
    # Create visualizations
    create_intermittency_plots(analysis_df, output_dir)
    
    # Generate recommendations
    generate_recommendations(analysis_df, output_dir)
    
    return analysis_df


def create_intermittency_plots(df: pd.DataFrame, output_dir: Path):
    """Create visualizations for intermittency analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Demand Pattern Distribution
    ax = axes[0, 0]
    pattern_counts = df['demand_pattern'].value_counts()
    ax.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
    ax.set_title('Distribution of Demand Patterns')
    
    # 2. ADI vs CV² Scatter
    ax = axes[0, 1]
    for pattern in df['demand_pattern'].unique():
        mask = df['demand_pattern'] == pattern
        ax.scatter(df[mask]['adi'], df[mask]['cv2'], label=pattern, alpha=0.6, s=100)
    
    # Add classification boundaries
    ax.axvline(x=1.32, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0.49, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('ADI (Average Demand Interval)')
    ax.set_ylabel('CV² (Coefficient of Variation Squared)')
    ax.set_title('Syntetos-Boylan Classification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Demand Frequency Distribution
    ax = axes[1, 0]
    ax.hist(df['demand_percentage'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(df['demand_percentage'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["demand_percentage"].mean():.1f}%')
    ax.set_xlabel('Percentage of Days with Demand')
    ax.set_ylabel('Number of Items')
    ax.set_title('Demand Frequency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Top 10 Items Analysis
    ax = axes[1, 1]
    top10 = df.head(10)
    
    x = np.arange(len(top10))
    width = 0.35
    
    ax.bar(x - width/2, top10['periods_with_demand'], width, 
           label='Days with Demand', alpha=0.7)
    ax.bar(x + width/2, top10['periods_without_demand'], width, 
           label='Days without Demand', alpha=0.7)
    
    ax.set_xlabel('Item ID')
    ax.set_ylabel('Number of Days')
    ax.set_title('Demand Occurrence - Top 10 Items by Volume')
    ax.set_xticks(x)
    ax.set_xticklabels(top10['item_id'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intermittency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_recommendations(df: pd.DataFrame, output_dir: Path):
    """Generate recommendations based on intermittency analysis."""
    
    report_lines = []
    report_lines.append("# Intermittent Demand Analysis and Recommendations")
    report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary statistics
    report_lines.append("\n## Summary Statistics")
    report_lines.append(f"- Total items analyzed: {len(df)}")
    report_lines.append(f"- Average demand frequency: {df['demand_percentage'].mean():.1f}%")
    report_lines.append(f"- Items with <10% demand frequency: {(df['demand_percentage'] < 10).sum()}")
    
    # Demand pattern breakdown
    report_lines.append("\n## Demand Pattern Classification")
    for pattern, count in df['demand_pattern'].value_counts().items():
        pct = count / len(df) * 100
        report_lines.append(f"- {pattern}: {count} items ({pct:.1f}%)")
    
    # Recommended methods by pattern
    report_lines.append("\n## Recommended Forecasting Methods by Pattern")
    
    report_lines.append("\n### Lumpy Demand (High intermittency, High variability)")
    lumpy_items = df[df['demand_pattern'] == 'Lumpy']
    report_lines.append(f"Items: {', '.join(lumpy_items['item_id'].astype(str).tolist())}")
    report_lines.append("Recommended methods:")
    report_lines.append("- Croston's method or SBA (Syntetos-Boylan Approximation)")
    report_lines.append("- TSB (Teunter-Syntetos-Babai) method")
    report_lines.append("- Machine learning with zero-inflated models")
    
    report_lines.append("\n### Intermittent Demand (High intermittency, Low variability)")
    intermittent_items = df[df['demand_pattern'] == 'Intermittent']
    report_lines.append(f"Items: {', '.join(intermittent_items['item_id'].astype(str).tolist())}")
    report_lines.append("Recommended methods:")
    report_lines.append("- Croston's method")
    report_lines.append("- Simple exponential smoothing on non-zero demands")
    
    report_lines.append("\n### Erratic Demand (Low intermittency, High variability)")
    erratic_items = df[df['demand_pattern'] == 'Erratic']
    report_lines.append(f"Items: {', '.join(erratic_items['item_id'].astype(str).tolist())}")
    report_lines.append("Recommended methods:")
    report_lines.append("- Standard time series methods with robust estimators")
    report_lines.append("- XGBoost with appropriate features")
    
    report_lines.append("\n### Smooth Demand (Low intermittency, Low variability)")
    smooth_items = df[df['demand_pattern'] == 'Smooth']
    report_lines.append(f"Items: {', '.join(smooth_items['item_id'].astype(str).tolist())}")
    report_lines.append("Recommended methods:")
    report_lines.append("- Traditional time series (ARIMA, ETS)")
    report_lines.append("- XGBoost/Machine Learning")
    
    # Implementation recommendations
    report_lines.append("\n## Implementation Recommendations")
    report_lines.append("\n1. **For Lumpy/Intermittent items (majority of inventory):**")
    report_lines.append("   - Implement Croston's method or TSB as baseline")
    report_lines.append("   - Use separate forecasts for demand occurrence and demand size")
    report_lines.append("   - Consider lead time demand distribution instead of point forecasts")
    
    report_lines.append("\n2. **For XGBoost/ML approaches:**")
    report_lines.append("   - Use zero-inflated models (forecast P(demand>0) separately)")
    report_lines.append("   - Include features like 'days since last demand'")
    report_lines.append("   - Consider aggregating to weekly/monthly for more stable patterns")
    
    report_lines.append("\n3. **For all items:**")
    report_lines.append("   - Focus on service level rather than forecast accuracy")
    report_lines.append("   - Implement safety stock based on lead time demand variability")
    report_lines.append("   - Consider min/max inventory policies for very intermittent items")
    
    # Save report
    with open(output_dir / "intermittency_recommendations.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nRecommendations saved to {output_dir}")


if __name__ == "__main__":
    df = analyze_all_items()
    print("\nIntermittency Analysis Complete!")
    print(f"\nDemand Pattern Summary:")
    print(df['demand_pattern'].value_counts())