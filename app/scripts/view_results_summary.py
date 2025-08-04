#!/usr/bin/env python3
"""
Quick script to view all results in a formatted summary.
"""
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def view_results():
    """Display comprehensive results summary."""
    
    print("=" * 80)
    print("INVENTORY FORECASTING SYSTEM - RESULTS SUMMARY")
    print("=" * 80)
    
    # 1. Training Results
    training_file = Path(__file__).parent.parent / "checkpoints" / "training_results_final.csv"
    if training_file.exists():
        df = pd.read_csv(training_file)
        print("\n1. MODEL TRAINING RESULTS")
        print("-" * 40)
        print(f"Total Parts Trained: {len(df)}")
        print(f"Success Rate: {(df['status'] == 'success').mean():.1%}")
        print(f"Average MAE: {df['mae'].mean():.4f}")
        print(f"Best Model MAE: {df['mae'].min():.6f}")
        print(f"Average Zero %: {df['zero_pct'].mean():.1f}%")
        
        print("\nTop 5 Best Performing Models:")
        top5 = df.nsmallest(5, 'mae')[['stock_code', 'mae', 'zero_pct']]
        for _, row in top5.iterrows():
            print(f"  {row['stock_code'][:30]:30} MAE: {row['mae']:.6f} (Zero: {row['zero_pct']:.1f}%)")
    
    # 2. Supplier Performance
    supplier_file = Path(__file__).parent.parent / "data" / "supplier_performance.csv"
    if supplier_file.exists():
        df = pd.read_csv(supplier_file)
        print("\n2. SUPPLIER PERFORMANCE")
        print("-" * 40)
        print(f"Total Suppliers Evaluated: {len(df)}")
        print(f"Average Lead Time: {df['avg_lead_time'].mean():.1f} days")
        print(f"Average On-Time Rate: {df['on_time_rate'].mean():.1%}")
        
        print("\nTop Suppliers:")
        for _, row in df.head(3).iterrows():
            print(f"  Supplier {int(row['supplier_id'])} - Score: {row['performance_score']:.1f}/100, "
                  f"Lead: {row['avg_lead_time']:.0f} days")
    
    # 3. Lead Time Statistics
    lead_time_file = Path(__file__).parent.parent / "data" / "lead_time_statistics.csv"
    if lead_time_file.exists():
        df = pd.read_csv(lead_time_file)
        print("\n3. LEAD TIME ANALYSIS")
        print("-" * 40)
        print(f"Item-Supplier Combinations: {len(df)}")
        print(f"Average Lead Time: {df['lead_time_mean'].mean():.1f} days")
        print(f"95th Percentile Lead Time: {df['lead_time_p95'].mean():.1f} days")
        print(f"Average Reliability: {df['supplier_reliability'].mean():.1%}")
    
    # 4. Asymmetric Loss Testing
    asym_file = Path(__file__).parent.parent / "data" / "synthetic_asymmetric_comparison.csv"
    if asym_file.exists():
        df = pd.read_csv(asym_file)
        print("\n4. ASYMMETRIC LOSS IMPACT")
        print("-" * 40)
        baseline = df[df['penalty'] == 1.0]
        penalty3 = df[df['penalty'] == 3.0]
        
        if len(baseline) > 0 and len(penalty3) > 0:
            stockout_reduction = (baseline['stockout_rate'].mean() - penalty3['stockout_rate'].mean()) / baseline['stockout_rate'].mean() * 100
            print(f"Stockout Reduction with 3x Penalty: {stockout_reduction:.1f}%")
            print(f"Baseline Stockout Rate: {baseline['stockout_rate'].mean():.1%}")
            print(f"With Asymmetric Loss: {penalty3['stockout_rate'].mean():.1%}")
    
    # 5. Model Files
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    model_files = list(checkpoint_dir.glob("model_part_*.pkl"))
    print("\n5. SAVED MODELS")
    print("-" * 40)
    print(f"Total Models Saved: {len(model_files)}")
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
        print(f"Total Storage Used: {total_size:.1f} MB")
        print(f"Average Model Size: {total_size/len(model_files):.2f} MB")
    
    # 6. Data Coverage
    parts_file = Path(__file__).parent.parent / "data" / "parts_demand_analysis.csv"
    if parts_file.exists():
        df = pd.read_csv(parts_file)
        print("\n6. DATA COVERAGE")
        print("-" * 40)
        print(f"Total Parts Analyzed: {len(df)}")
        print(f"Parts with Very High Intermittency (>90% zeros): {(df['zero_percentage'] > 90).sum()}")
        print(f"Parts with High Intermittency (70-90% zeros): {((df['zero_percentage'] >= 70) & (df['zero_percentage'] <= 90)).sum()}")
        print(f"Parts with Moderate Intermittency (<70% zeros): {(df['zero_percentage'] < 70).sum()}")
    
    print("\n" + "=" * 80)
    print("OUTPUTS LOCATION:")
    print("-" * 40)
    print(f"Excel Reports:    {Path(__file__).parent.parent / 'outputs'}")
    print(f"Trained Models:   {Path(__file__).parent.parent / 'checkpoints'}")
    print(f"Data Analysis:    {Path(__file__).parent.parent / 'data'}")
    print(f"Logs:            {Path(__file__).parent.parent.parent / 'logs'}")
    print("=" * 80)

if __name__ == "__main__":
    view_results()