# Inventory Forecasting System - Project Structure

## Recommended Directory Structure

```
prediction/
├── app/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── settings.py          # Environment variables, constants
│   │   │   └── database.py          # Database connection config
│   │   │
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── extractors/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_extractor.py
│   │   │   │   ├── sales_extractor.py      # tbl_sorder_item/txn
│   │   │   │   ├── purchase_extractor.py   # tbl_porder_item/txn
│   │   │   │   └── job_order_extractor.py  # tbl_jo_item/txn
│   │   │   │
│   │   │   ├── cleaners/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── data_cleaner.py    # Remove nulls, outliers
│   │   │   │   └── validator.py       # Data quality checks
│   │   │   │
│   │   │   └── database.py            # Database connection manager
│   │   │
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── time_features.py      # Day of week, month, holidays
│   │   │   ├── lag_features.py       # Previous consumption values
│   │   │   ├── rolling_features.py   # Moving averages, std dev
│   │   │   └── feature_pipeline.py   # Combine all features
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py         # Abstract base class
│   │   │   ├── xgboost_model.py      # XGBoost implementation
│   │   │   ├── baseline_model.py     # Simple moving average
│   │   │   ├── trainer.py            # Model training logic
│   │   │   └── predictor.py          # Prediction interface
│   │   │
│   │   ├── reports/
│   │   │   ├── __init__.py
│   │   │   ├── report_generator.py   # Main report creation
│   │   │   ├── excel_exporter.py     # Excel file generation
│   │   │   ├── email_sender.py       # Email alerts
│   │   │   └── templates/            # Report templates
│   │   │       ├── daily_report.html
│   │   │       └── alert_email.html
│   │   │
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── logger.py             # Logging configuration
│   │   │   ├── metrics.py            # RMSE, MAE calculations
│   │   │   └── date_utils.py        # Date handling helpers
│   │   │
│   │   └── main.py                   # Entry point
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_extractors/
│   │   ├── test_features/
│   │   ├── test_models/
│   │   └── test_integration.py
│   │
│   ├── notebooks/               # Jupyter notebooks for analysis
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_model_training.ipynb
│   │   └── 04_results_analysis.ipynb
│   │
│   ├── scripts/
│   │   ├── run_daily_forecast.py    # Daily cron job
│   │   ├── train_models.py          # Model training script
│   │   └── backtest.py              # Historical validation
│   │
│   └── web/                     # Dashboard (if needed)
│       ├── app.py               # Flask/FastAPI app
│       ├── static/
│       └── templates/
│
├── data/                        # Data storage
│   ├── raw/                     # Raw extracted data
│   ├── processed/               # Cleaned data
│   ├── features/                # Feature datasets
│   └── models/                  # Saved models
│
├── logs/                        # Application logs
│   ├── app.log
│   ├── errors.log
│   └── predictions.log
│
├── outputs/                     # Generated reports
│   ├── daily_reports/
│   ├── excel_exports/
│   └── alerts_sent/
│
├── docs/                        # Documentation
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── USER_GUIDE.md
│
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore
├── Dockerfile                  # Container setup
├── docker-compose.yml
└── README.md
```

## Key Design Principles

### 1. Separation of Concerns
Each module has a single responsibility:
- **Extractors**: Only extract data from database
- **Features**: Only create features from raw data
- **Models**: Only handle predictions
- **Reports**: Only generate outputs

### 2. Configuration Management
```python
# config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database
    DB_HOST: str
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str = "nex_valiant"
    
    # Model
    FORECAST_DAYS: int = 14
    SAFETY_FACTOR: float = 1.2
    
    # Alerts
    URGENT_THRESHOLD_DAYS: int = 2
    WARNING_THRESHOLD_DAYS: int = 7
    
    # Email
    SMTP_HOST: str
    SMTP_PORT: int = 587
    EMAIL_FROM: str
    EMAIL_TO: list[str]
    
    class Config:
        env_file = ".env"
```

### 3. Data Flow Pipeline
```python
# main.py
def run_daily_forecast():
    # 1. Extract
    sales_data = SalesExtractor().extract(days_back=90)
    purchase_data = PurchaseExtractor().extract()
    consumption_data = JobOrderExtractor().extract()
    
    # 2. Clean
    clean_data = DataCleaner().clean(consumption_data)
    
    # 3. Features
    features = FeaturePipeline().create_features(clean_data)
    
    # 4. Predict
    model = XGBoostModel.load("latest")
    predictions = model.predict(features)
    
    # 5. Report
    report = ReportGenerator().generate(predictions)
    EmailSender().send_alerts(report.urgent_items)
```

### 4. Model Management
```python
# models/xgboost_model.py
class XGBoostInventoryModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def train(self, X, y):
        # Train with cross-validation
        # Save model with timestamp
        
    def predict(self, X):
        # Make predictions
        # Add confidence intervals
        
    def save(self, path):
        # Save model, scaler, feature names
        
    def load(self, path):
        # Load saved model
```

### 5. Testing Strategy
```python
# tests/test_models/test_xgboost.py
def test_model_predictions():
    # Test with known data
    
def test_feature_importance():
    # Verify important features
    
def test_model_persistence():
    # Test save/load functionality
```

## Development Workflow

### 1. Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your database credentials

# Run tests
pytest

# Run locally
python app/src/main.py
```

### 2. Docker Development
```bash
# Build image
docker build -t inventory-forecast .

# Run with docker-compose
docker-compose up

# Run tests in container
docker-compose run app pytest
```

### 3. Production Deployment
```bash
# Schedule daily runs
crontab -e
# Add: 0 6 * * * /usr/bin/python /app/scripts/run_daily_forecast.py

# Or use Apache Airflow for more control
```

## Database Access Pattern
```python
# data/database.py
from contextlib import contextmanager
import mysql.connector
from mysql.connector import pooling

class DatabaseManager:
    def __init__(self, config):
        self.pool = pooling.MySQLConnectionPool(
            pool_name="inventory_pool",
            pool_size=5,
            **config
        )
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.get_connection()
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query, params=None):
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params)
            return cursor.fetchall()
```

## Logging Strategy
```python
# utils/logger.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    handler = RotatingFileHandler(
        f"logs/{name}.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
```

This structure is:
- **Modular**: Easy to test individual components
- **Scalable**: Can add new extractors/models easily
- **Maintainable**: Clear separation of concerns
- **Production-ready**: Includes logging, config, error handling