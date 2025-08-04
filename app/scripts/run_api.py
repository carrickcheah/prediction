#!/usr/bin/env python3
"""
Script to run the FastAPI server.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if __name__ == "__main__":
    import uvicorn
    from src.api.app import app
    
    print("=" * 60)
    print("Starting Inventory Forecasting API")
    print("=" * 60)
    print("\nAPI will be available at:")
    print("  http://localhost:8000")
    print("\nAPI Documentation:")
    print("  http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )