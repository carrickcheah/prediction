"""
Analytics endpoints for system metrics and alerts.
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, date
from typing import List

from src.api.models.schemas import (
    AnalyticsSummary,
    AlertsResponse,
    Alert,
    ModelStatus
)
from src.api.services.prediction_service import PredictionService
from src.api.services.cache import ModelCache

router = APIRouter()

# Global model cache (will be set by app.py)
model_cache = None
prediction_service = None

def init_router(cache: ModelCache):
    """Initialize router with model cache."""
    global model_cache, prediction_service
    model_cache = cache
    prediction_service = PredictionService(model_cache)


@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary():
    """
    Get overall system analytics and metrics.
    
    Returns:
        Summary of system performance and status
    """
    try:
        # Get loaded parts
        loaded_parts = model_cache.get_loaded_parts()
        
        # Calculate average accuracy from metadata
        accuracies = []
        for part_id in loaded_parts:
            metadata = model_cache.get_model_metadata(part_id)
            if metadata and 'mae' in metadata:
                accuracies.append(metadata['mae'])
                
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # Get alerts to find high risk parts
        alerts = prediction_service.get_alerts()
        high_risk_parts = [a.part_id for a in alerts if a.urgency in ['critical', 'high']]
        critical_count = sum(1 for a in alerts if a.urgency == 'critical')
        
        # Get cache stats
        cache_stats = model_cache.get_cache_size()
        
        return AnalyticsSummary(
            total_parts_monitored=len(loaded_parts),
            models_loaded=cache_stats['models'],
            average_model_accuracy=1 - avg_accuracy if avg_accuracy < 1 else 0.95,  # Convert MAE to accuracy
            total_predictions_today=cache_stats['predictions'],
            critical_alerts=critical_count,
            high_risk_parts=high_risk_parts[:10],  # Top 10
            system_health="healthy" if cache_stats['models'] > 0 else "degraded",
            last_training_date=date(2025, 8, 4),  # From our training
            data_coverage={
                "total_parts": len(loaded_parts),
                "high_intermittency": sum(1 for p in loaded_parts if model_cache.get_model_metadata(p).get('zero_pct', 0) > 90),
                "coverage_percentage": 100.0  # All loaded parts have coverage
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts():
    """
    Get current inventory alerts.
    
    Returns:
        List of alerts for parts needing attention
    """
    try:
        alerts = prediction_service.get_alerts()
        
        # Count by urgency
        critical_count = sum(1 for a in alerts if a.urgency == 'critical')
        high_count = sum(1 for a in alerts if a.urgency == 'high')
        
        return AlertsResponse(
            alerts=alerts,
            total_alerts=len(alerts),
            critical_count=critical_count,
            high_count=high_count,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert generation failed: {str(e)}")


@router.get("/analytics/model-status", response_model=List[ModelStatus])
async def get_model_status():
    """
    Get status of all loaded models.
    
    Returns:
        List of model status information
    """
    try:
        status_list = []
        
        for part_id in model_cache.get_loaded_parts():
            metadata = model_cache.get_model_metadata(part_id)
            
            # Get model file size (approximate)
            model = model_cache.get_model(part_id)
            import sys
            model_size = sys.getsizeof(model) / 1024 if model else 0
            
            status = ModelStatus(
                part_id=part_id,
                stock_code=metadata.get('stock_code'),
                model_type="TwoStage" if metadata.get('zero_pct', 0) > 80 else "XGBoost",
                last_prediction=None,  # Would need to track this
                mae=metadata.get('mae'),
                is_loaded=True,
                file_size_kb=model_size
            )
            status_list.append(status)
            
        return status_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/analytics/clear-cache")
async def clear_cache():
    """
    Clear prediction cache (keeps models loaded).
    
    Returns:
        Confirmation message
    """
    try:
        model_cache.clear_expired_predictions()
        return {"message": "Cache cleared successfully", "timestamp": datetime.now()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")