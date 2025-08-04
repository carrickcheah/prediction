"""
Prediction endpoints for the API.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List
from datetime import datetime

from src.api.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse
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


@router.post("/predict/{part_id}", response_model=PredictionResponse)
async def predict_single(
    part_id: int,
    horizon_days: int = Query(14, ge=1, le=90),
    include_confidence: bool = Query(False)
):
    """
    Generate prediction for a single part.
    
    Args:
        part_id: ID of the part to predict
        horizon_days: Number of days to forecast (1-90)
        include_confidence: Include confidence intervals
        
    Returns:
        Prediction response with forecasted values
    """
    try:
        response = prediction_service.predict_single(
            part_id=part_id,
            horizon_days=horizon_days,
            include_confidence=include_confidence
        )
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Generate predictions for multiple parts.
    
    Args:
        request: Batch prediction request with part IDs
        
    Returns:
        Batch response with all predictions
    """
    start_time = datetime.now()
    
    # Limit number of parts
    part_ids = request.part_ids[:request.max_parts]
    
    try:
        predictions = prediction_service.predict_batch(
            part_ids=part_ids,
            horizon_days=request.horizon_days
        )
        
        # Calculate statistics
        successful = sum(1 for p in predictions if p.predictions)
        failed = len(predictions) - successful
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_parts=len(part_ids),
            successful=successful,
            failed=failed,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/predict/available-parts", response_model=List[int])
async def get_available_parts():
    """
    Get list of parts with available models.
    
    Returns:
        List of part IDs that can be predicted
    """
    return model_cache.get_loaded_parts()