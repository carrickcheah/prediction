# Inventory Forecasting System - Data Analysis Report

**Generated**: 2025-08-04 14:42  
**Purpose**: Analysis of available parts data and training feasibility

---

## Executive Summary

The inventory forecasting system faces a significant data sparsity challenge. While the system contains **4,894 total parts** in the master data, only **14 parts** meet the minimum 30 data points requirement for reliable forecasting. With a 100 data points requirement, only **2 parts** qualify.

### Key Findings
- **97.2%** of parts have fewer than 30 consumption data points
- Only **0.1%** (2 parts) have 100+ data points
- Most parts exhibit extreme intermittency patterns
- Current training limited to 14 parts instead of expected 6000+

---

## Data Availability Analysis

### Parts Distribution by Data Points

| Data Points Range | Parts Count | Percentage | Cumulative % |
|-------------------|-------------|------------|--------------|
| **100+** | 2 | 0.1% | 0.1% |
| **50-99** | 7 | 0.3% | 0.4% |
| **30-49** | 62 | 2.4% | 2.8% |
| **10-29** | 260 | 10.3% | 13.1% |
| **5-9** | 557 | 22.0% | 35.1% |
| **1-4** | 1,648 | 65.0% | 100.0% |
| **Total with data** | **2,536** | **100%** | - |
| **No consumption** | 2,358 | - | - |
| **Grand Total** | **4,894** | - | - |

### Visual Representation

```
Data Points Distribution (2,536 parts with consumption):
100+  : ■ (0.1%)
50-99 : ■ (0.3%)
30-49 : ■■ (2.4%)
10-29 : ■■■■■■■■■■ (10.3%)
5-9   : ■■■■■■■■■■■■■■■■■■■■■■ (22.0%)
1-4   : ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ (65.0%)
```

---

## Parts Qualifying for Training

### With Different Minimum Requirements

| Min Data Points | Qualifying Parts | % of Total | Recommended Approach |
|-----------------|------------------|------------|----------------------|
| **100** | 2 | 0.04% | Full XGBoost with all features |
| **50** | 9 | 0.18% | XGBoost with reduced features |
| **30** | 71 | 1.45% | XGBoost or ARIMA |
| **20** | 86 | 1.76% | Simplified models |
| **10** | 275 | 5.62% | Moving averages |
| **5** | 832 | 17.00% | Basic forecasting |

### Currently Trained Parts (14 total)

These 14 parts have 30+ data points and were successfully trained:

| Part ID | Stock Code | Data Points | Zero % | MAE |
|---------|------------|-------------|--------|-----|
| 1003053 | VPSB-MTL | 126 | 76.6% | 0.119 |
| 1003646 | VPSB-SAMP-P01-01 | 126 | 76.6% | 0.119 |
| 1000045 | D0.4/0.5-SGCC-1220MM SGCC COIL | ~45 | 93.1% | 0.021 |
| 1000073 | D1.0-SGCC-Z27-COIL | ~40 | 93.1% | 0.000 |
| 1000087 | D1.2-SPCC-1219MM SPCC COIL | ~44 | 92.2% | 0.005 |
| (9 more parts with similar characteristics) | | | | |

---

## Data Quality Issues

### 1. Extreme Data Sparsity
- **65%** of parts have only 1-4 data points over 18+ months
- Insufficient history for traditional time series forecasting
- Requires alternative approaches for majority of inventory

### 2. High Intermittency
- **78.6%** of trainable parts have >90% zero demand days
- **21.4%** have 70-90% zero demand days
- No parts with regular, consistent demand patterns

### 3. Data Freshness Concerns
- **No consumption data in last 7 days** (as of 2025-08-04)
- Latest data point: 2025-07-22
- Potential data pipeline or business operation issues

### 4. Coverage Gap
- Expected: 6000+ parts
- Parts with data: 2,536 (42.3%)
- Trainable (30+ points): 14 (0.3%)
- **Gap: 99.7% of expected parts cannot be trained**

---

## Root Cause Analysis

### Why So Few Parts Have Sufficient Data?

1. **Business Model Characteristics**
   - Many parts may be finished goods (not consumed in manufacturing)
   - High product variety with low volume per SKU
   - Seasonal or project-based consumption

2. **Data Collection Issues**
   - Manual data entry errors (as noted in CLAUDE.md)
   - Incomplete job order recording
   - Parts consumed but not tracked in system

3. **Inventory Characteristics**
   - Large number of obsolete/inactive parts in master data
   - New parts without consumption history
   - Spare parts with rare usage

4. **Database Query Limitations**
   - Current queries focus only on job order consumption
   - May miss other consumption channels
   - Date range restrictions limiting historical data

---

## Recommendations

### Immediate Actions (Quick Wins)

1. **Adjust Training Requirements**
   ```python
   # Change from min_data_points=30 to min_data_points=10
   # This increases trainable parts from 14 to 275
   ```

2. **Extend Date Range**
   ```sql
   -- Current: AND jt.TxnDate_dd >= '2024-01-01'
   -- Recommended: AND jt.TxnDate_dd >= '2023-01-01'
   ```

3. **Implement Tiered Approach**
   - Tier 1 (30+ points): Full XGBoost - 71 parts
   - Tier 2 (10-29 points): Simplified models - 204 parts
   - Tier 3 (<10 points): Rule-based or category averages - 2,261 parts

### Medium-Term Solutions

1. **Alternative Data Sources**
   - Include sales order data for demand signals
   - Incorporate purchase order patterns
   - Use product category similarities

2. **Hierarchical Forecasting**
   - Group similar parts by category
   - Forecast at category level, disaggregate to parts
   - Use product attributes for clustering

3. **Data Quality Improvement**
   - Implement automated data validation
   - Flag and investigate data gaps
   - Regular data pipeline monitoring

### Long-Term Strategy

1. **Hybrid Approach**
   - ML models for high-volume parts
   - Statistical methods for medium-volume
   - Business rules for low-volume/new parts

2. **Continuous Learning**
   - Start with simple models, improve as data accumulates
   - Online learning to adapt to new patterns
   - Regular model retraining schedule

3. **Business Process Integration**
   - Align forecasting with business planning cycles
   - Incorporate external factors (promotions, projects)
   - Human-in-the-loop for critical decisions

---

## Impact Assessment

### Current State (14 parts trained)
- **Coverage**: 0.3% of inventory
- **Business Impact**: Minimal - critical parts may not be covered
- **Risk**: High stockout potential for unforecasted parts

### With Recommended Changes (275 parts)
- **Coverage**: 5.6% of inventory (20x improvement)
- **Business Impact**: Moderate - likely covers high-value parts
- **Risk**: Reduced for frequently used parts

### Ideal State (1000+ parts)
- **Coverage**: 20%+ of active inventory
- **Business Impact**: Significant - covers majority of consumption value
- **Risk**: Well-managed for business-critical parts

---

## Conclusion

The current data limitation to 14 trainable parts severely constrains the forecasting system's effectiveness. The root cause is extreme data sparsity, with 97.2% of parts having insufficient consumption history for traditional forecasting methods.

### Critical Next Steps
1. **Immediately reduce min_data_points to 10** to increase coverage to 275 parts
2. **Investigate data pipeline** for the 7-day gap in recent data
3. **Implement tiered forecasting** approach based on data availability
4. **Consider alternative data sources** beyond job order consumption

### Success Metrics
- Increase trainable parts from 14 to 275+ (short-term)
- Achieve 20% inventory coverage (medium-term)
- Reduce stockout incidents by 30% (long-term)

---

## Appendix: SQL Queries for Analysis

### Query to find parts with specific data point thresholds
```sql
SELECT 
    COUNT(CASE WHEN cnt >= 100 THEN 1 END) as parts_100plus,
    COUNT(CASE WHEN cnt >= 50 THEN 1 END) as parts_50plus,
    COUNT(CASE WHEN cnt >= 30 THEN 1 END) as parts_30plus,
    COUNT(CASE WHEN cnt >= 10 THEN 1 END) as parts_10plus
FROM (
    SELECT ItemId_i, COUNT(DISTINCT DATE(TxnDate_dd)) as cnt
    FROM tbl_jo_item ji
    JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
    WHERE ji.InOut_c = 'I' AND jt.Void_c = '0'
    GROUP BY ItemId_i
) counts
```

### Query to analyze data freshness
```sql
SELECT 
    MAX(DATE(TxnDate_dd)) as latest_date,
    DATEDIFF(CURDATE(), MAX(DATE(TxnDate_dd))) as days_since_last_data
FROM tbl_jo_txn
WHERE Void_c = '0'
```

---

*This report highlights critical data availability constraints that must be addressed for successful inventory forecasting implementation.*