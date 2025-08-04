"""
Main entry point for the inventory forecasting system.
"""
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from data.extractors.sales_extractor import SalesExtractor
from data.extractors.purchase_extractor import PurchaseExtractor
from data.extractors.job_order_extractor import JobOrderExtractor
from data.processors.data_aggregator import DataAggregator
from forecasting.trainer import InventoryForecaster
from reports.report_generator import ReportGenerator
from reports.email_sender import EmailSender
from utils.logger import setup_logger

logger = setup_logger("main")


def run_daily_forecast():
    """Run the daily forecasting pipeline."""
    logger.info("Starting daily forecast run")
    settings = get_settings()
    
    try:
        # 1. Extract data
        logger.info("Extracting data from database")
        sales_extractor = SalesExtractor()
        purchase_extractor = PurchaseExtractor()
        job_order_extractor = JobOrderExtractor()
        
        # Get data for the past year
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        
        sales_data = sales_extractor.extract(start_date, end_date)
        purchase_data = purchase_extractor.extract(start_date, end_date)
        consumption_data = job_order_extractor.extract(start_date, end_date)
        
        # 2. Process and aggregate data
        logger.info("Processing and aggregating data")
        aggregator = DataAggregator()
        processed_data = aggregator.aggregate(
            sales_data=sales_data,
            purchase_data=purchase_data,
            consumption_data=consumption_data
        )
        
        # 3. Train and predict
        logger.info("Training model and generating predictions")
        forecaster = InventoryForecaster()
        predictions = forecaster.train_and_predict(
            processed_data,
            horizon=settings.FORECAST_HORIZON
        )
        
        # 4. Generate reports
        logger.info("Generating reports")
        report_generator = ReportGenerator()
        report = report_generator.create_procurement_report(predictions)
        
        # 5. Send alerts
        logger.info("Sending email alerts")
        email_sender = EmailSender()
        email_sender.send_critical_alerts(report.urgent_orders)
        
        logger.info("Daily forecast completed successfully")
        
    except Exception as e:
        logger.error(f"Error in daily forecast: {str(e)}", exc_info=True)
        raise


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inventory Forecasting System")
    parser.add_argument(
        "command",
        choices=["extract", "train", "predict", "report", "run"],
        help="Command to execute"
    )
    parser.add_argument("--days", type=int, default=90, help="Days of history to extract")
    parser.add_argument("--parts", default="all", help="Parts to process (all/top20/specific IDs)")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon in days")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_daily_forecast()
    else:
        logger.info(f"Command {args.command} not yet implemented")
        # TODO: Implement individual commands


if __name__ == "__main__":
    main()