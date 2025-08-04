"""
Report generation endpoints.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import uuid

from src.api.models.schemas import ReportRequest, ReportResponse
from src.api.services.prediction_service import PredictionService
from src.reports.excel_reporter import ExcelReporter
from src.api.services.cache import ModelCache

router = APIRouter()

# Global services (will be set by app.py)
model_cache = None
prediction_service = None
excel_reporter = ExcelReporter()

def init_router(cache: ModelCache):
    """Initialize router with model cache."""
    global model_cache, prediction_service
    model_cache = cache
    prediction_service = PredictionService(model_cache)

# Reports directory
REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "api_reports"
REPORTS_DIR.mkdir(exist_ok=True)


@router.post("/reports/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate a report (Excel or CSV).
    
    Args:
        request: Report generation request
        
    Returns:
        Report response with download information
    """
    try:
        report_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine which parts to include
        if request.include_parts:
            part_ids = request.include_parts
        else:
            part_ids = model_cache.get_loaded_parts()[:20]  # Default to first 20
            
        # Generate predictions for all parts
        all_predictions = []
        for part_id in part_ids:
            try:
                pred = prediction_service.predict_single(part_id, horizon_days=14)
                all_predictions.append({
                    'part_id': pred.part_id,
                    'stock_code': pred.stock_code,
                    'urgency': pred.urgency,
                    'zero_pct': pred.zero_percentage,
                    'mae': pred.model_mae,
                    'forecast_sum': sum(pred.predictions),
                    'forecast_mean': sum(pred.predictions) / len(pred.predictions) if pred.predictions else 0
                })
            except Exception as e:
                continue
                
        # Create DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        
        if request.report_type == "excel":
            # Generate Excel report
            file_name = f"forecast_report_{report_id}_{timestamp}.xlsx"
            file_path = REPORTS_DIR / file_name
            
            # Create multi-sheet Excel
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([{
                    'Report ID': report_id,
                    'Generated At': datetime.now(),
                    'Total Parts': len(predictions_df),
                    'Critical Alerts': (predictions_df['urgency'] == 'critical').sum() if not predictions_df.empty else 0,
                    'Average MAE': predictions_df['mae'].mean() if not predictions_df.empty else 0
                }])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Predictions sheet
                if not predictions_df.empty:
                    predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                    
        else:
            # Generate CSV report
            file_name = f"forecast_report_{report_id}_{timestamp}.csv"
            file_path = REPORTS_DIR / file_name
            predictions_df.to_csv(file_path, index=False)
            
        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        return ReportResponse(
            report_id=report_id,
            file_path=str(file_path),
            file_size_mb=file_size_mb,
            generated_at=datetime.now(),
            download_url=f"/api/reports/download/{file_name}",
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/reports/download/{file_name}")
async def download_report(file_name: str):
    """
    Download a generated report.
    
    Args:
        file_name: Name of the report file
        
    Returns:
        File download response
    """
    file_path = REPORTS_DIR / file_name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
        
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if file_name.endswith('.xlsx') else 'text/csv'
    )


@router.get("/reports/list")
async def list_reports():
    """
    List available reports.
    
    Returns:
        List of available report files
    """
    try:
        reports = []
        for file_path in REPORTS_DIR.glob("forecast_report_*"):
            reports.append({
                'file_name': file_path.name,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'created_at': datetime.fromtimestamp(file_path.stat().st_ctime),
                'download_url': f"/api/reports/download/{file_path.name}"
            })
            
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x['created_at'], reverse=True)
        
        return reports[:20]  # Return last 20 reports
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")