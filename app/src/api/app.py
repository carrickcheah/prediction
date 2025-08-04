"""
FastAPI application for inventory forecasting system.
Simple and direct API for predictions and analytics.
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.routes import predictions, analytics, reports
from src.api.services.cache import ModelCache
from src.utils.logger import setup_logger

logger = setup_logger("api")

# Global model cache
model_cache = ModelCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting FastAPI application...")
    logger.info("Loading models into cache...")
    model_cache.load_all_models()
    logger.info(f"Loaded {len(model_cache.models)} models")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    model_cache.clear()


# Create FastAPI app
app = FastAPI(
    title="Inventory Forecasting API",
    description="Simple API for inventory predictions and analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple API key authentication
async def verify_api_key(x_api_key: str = Header(None)):
    """Simple API key verification."""
    # For development, accept any key or no key
    # In production, check against environment variable
    if x_api_key and x_api_key != "development-key-123":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# Initialize routers with model cache
predictions.init_router(model_cache)
analytics.init_router(model_cache)
reports.init_router(model_cache)

# Include routers
app.include_router(
    predictions.router,
    prefix="/api",
    tags=["predictions"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    analytics.router,
    prefix="/api",
    tags=["analytics"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    reports.router,
    prefix="/api",
    tags=["reports"],
    dependencies=[Depends(verify_api_key)]
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Inventory Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": len(model_cache.models),
        "endpoints": {
            "predictions": "/api/predict/{part_id}",
            "batch": "/api/predict/batch",
            "analytics": "/api/analytics/summary",
            "alerts": "/api/alerts",
            "reports": "/api/reports/generate"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_cache.models),
        "cache_size": model_cache.get_cache_size()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)