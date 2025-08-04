# Next Steps Implementation Plan

## Current Status Summary

### ✅ Completed
1. **Data Pipeline**: Extracting 365 days of consumption data from MariaDB
2. **Baseline Models**: MA, Naive, Exponential Smoothing benchmarks
3. **XGBoost Models**: 47.9% improvement for regular demand items
4. **Intermittent Demand Analysis**: 90% of items have <10% demand frequency
5. **Specialized Methods**: Croston's, SBA, TSB for intermittent demand
6. **Multi-Series Forecasting**: LightGBM model handling 100+ items efficiently (0.207 MAE)

### Key Findings
- **2,016 unique items** in the system
- **99.85% have intermittent demand** (only 3 items have >10% demand frequency)
- **Multi-series approach works**: Can handle many items with single model
- **Simple methods often win**: MA of non-zero demands beats complex methods for intermittent items

## Recommended Next Steps (Priority Order)

### 1. Scale Multi-Series Model to All Items (1-2 weeks)
**Why**: Current model handles 100 items well, ready to scale
**How**:
```python
# Modify build_multiseries_forecaster.py
# Set use_subset=False to train on all 2016 items
# May need to adjust parameters for memory efficiency:
- Reduce lags from 30 to 14-21 days
- Use sampling for very low volume items
- Consider chunking if memory issues
```
**Expected Impact**: Single model serving all items, dramatic reduction in training time

### 2. Implement Hybrid Forecasting System (1 week)
**Why**: Different demand patterns need different methods
**Architecture**:
```
Item Classifier
    ├── High Volume (0 items) → XGBoost
    ├── Regular (3 items) → XGBoost  
    ├── Intermittent (200+ items) → Multi-Series LightGBM
    └── Very Intermittent (1800+ items) → MA Non-Zero + Safety Stock
```
**Implementation**: Create `hybrid_forecaster.py` that routes items to appropriate method

### 3. Add Purchase Order Integration (3-4 days)
**Why**: Lead time visibility crucial for inventory planning
**Features to add**:
- Average lead time by supplier/item
- Lead time variability
- Open PO quantities
- Supplier performance metrics

**Code location**: Update `purchase_extractor.py` and add features to models

### 4. Implement Asymmetric Loss Function (2-3 days)
**Why**: Stockouts cost 3x more than overstock
**Implementation**:
```python
def asymmetric_loss(y_true, y_pred, stockout_weight=3.0):
    errors = y_true - y_pred
    return np.where(errors > 0,  # Underforecast (stockout)
                   stockout_weight * errors**2,
                   errors**2)  # Overforecast
```
**Where**: Add to XGBoost and LightGBM models as custom objective

### 5. Create Production API (1 week)
**Why**: Need real-time forecast access
**Stack**:
- FastAPI for REST endpoints
- Redis for caching forecasts
- PostgreSQL for forecast history
- Docker for deployment

**Endpoints**:
```
POST /forecast/items        # Batch forecast
GET  /forecast/item/{id}    # Single item forecast  
GET  /health               # Model status
POST /retrain              # Trigger retraining
```

### 6. Automated Retraining Pipeline (3-4 days)
**Why**: Models degrade over time
**Components**:
- Daily data extraction
- Weekly model retraining
- A/B testing framework
- Performance monitoring
- Automated rollback on degradation

**Tools**: Airflow or Prefect for orchestration

## Quick Wins (Can implement immediately)

### 1. Safety Stock Calculator
```python
def calculate_safety_stock(item_id, service_level=0.95):
    # Based on lead time demand distribution
    # Account for intermittent demand
    # Include supplier reliability
```

### 2. Forecast Accuracy Dashboard
- Track MAE, MAPE by item category
- Alert on forecast degradation
- Show confidence intervals

### 3. Inventory Optimization Report
- Current stock vs forecast
- Suggested order quantities
- Expected stockout risks

## Architecture for Production

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   MariaDB       │────▶│  ETL Pipeline │────▶│ Feature     │
│   (Source)      │     │  (Airflow)    │     │ Store       │
└─────────────────┘     └──────────────┘     └─────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   REST API      │◀────│   Model      │◀────│  Training   │
│   (FastAPI)     │     │   Registry   │     │  Pipeline   │
└─────────────────┘     └──────────────┘     └─────────────┘
         │                                            │
         ▼                                            ▼
┌─────────────────┐                          ┌─────────────┐
│   ERP System    │                          │  Monitoring │
│   Integration   │                          │  (Grafana)  │
└─────────────────┘                          └─────────────┘
```

## Success Metrics

1. **Forecast Accuracy**
   - MAE < 2.0 for top 20% of items
   - MAPE < 30% for regular demand items

2. **Business Impact**
   - 20% reduction in safety stock
   - <2% stockout rate
   - 15% reduction in working capital

3. **Operational Efficiency**
   - <5 min to generate all forecasts
   - <1 hour for weekly retraining
   - 99.9% API uptime

## Risk Mitigation

1. **Data Quality**: Implement validation checks
2. **Model Drift**: Monitor performance, auto-rollback
3. **Scalability**: Use chunking, sampling for 6000+ items
4. **Integration**: Gradual rollout, maintain manual override

## Timeline Summary

- **Week 1-2**: Scale multi-series model, implement hybrid system
- **Week 3**: Add purchase orders, asymmetric loss
- **Week 4**: Build production API
- **Week 5**: Automated pipeline, monitoring
- **Week 6**: Integration testing, documentation

Total: 6 weeks to production-ready system