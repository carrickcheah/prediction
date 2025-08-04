# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an inventory forecasting system designed to predict manufacturing stock needs using machine learning. The system analyzes part consumption patterns, sales data, and purchase orders to prevent stockouts and reduce excess inventory.

## Key Business Context

- **Primary Goal**: Predict inventory needs for 6000+ parts to prevent stockouts and reduce excess inventory
- **Data Sources**: Sales orders, purchase orders, and job orders (manufacturing consumption)
- **Critical Constraint**: DO NOT use inventory data - it's unreliable due to manual update errors
- **Approach**: Focus on part consumption patterns rather than finished goods sales

## Technology Stack

- **ML Framework**: XGBoost with skforecast (ForecasterAutoregMultiSeries)
- **Package Manager**: UV (not pip)
- **Database**: MariaDB (database name: nex_valiant)
- **Core Libraries**: pandas, numpy, scikit-learn, feature-engine, mysql-connector-python

## Development Commands

### Environment Setup
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd app
uv init
uv sync

# Add new dependencies
uv add package-name
uv add package-name --dev  # for dev dependencies
```

### Running the Application
```bash
# Extract data for analysis
uv run python src/main.py extract --days 90

# Train models
uv run python src/main.py train --parts top20

# Generate predictions
uv run python src/main.py predict --horizon 14

# Run daily forecast (production)
uv run python scripts/run_daily_forecast.py
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src tests/

# Run specific test file
uv run pytest tests/test_features.py -v
```

### Linting and Type Checking
```bash
# Run linter (if configured)
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Type checking (if mypy is configured)
uv run mypy src/
```

## Architecture Overview

### Data Flow
1. **Extract**: Pull data from MariaDB (sales, purchases, job orders)
2. **Clean**: Validate and preprocess data
3. **Features**: Engineer time series features using skforecast
4. **Model**: Multi-series XGBoost model handling all 6000+ parts
5. **Predict**: Generate 14-day forecasts
6. **Report**: Email alerts and Excel reports

### Key Database Tables

#### Sales Data (Customer Demand)
- **tbl_sorder_item** - Sales order line items
  - Key fields: ItemId_i, Qty_d, QtyDone_d, EtaDate_dd, TxnId_i, Void_c
- **tbl_sorder_txn** - Sales order headers  
  - Key fields: TxnId_i, TxnDate_dd, CustId_i, DocRef_v, DocStatus_c, Void_c

#### Purchase Data (Incoming Supply)
- **tbl_porder_item** - Purchase order line items
  - Key fields: ItemId_i, Qty_d, QtyDone_d, EtaDate_dd, TxnId_i, Void_c
- **tbl_porder_txn** - Purchase order headers
  - Key fields: TxnId_i, TxnDate_dd, SuppId_i, DocRef_v, DocStatus_c, Void_c

#### Manufacturing Data (Part Consumption)
- **tbl_jo_item** - Job order items (parts consumed/produced)
  - Key fields: ItemId_i, Qty_d, InOut_c ('I'=input/consumption, 'O'=output), TxnId_i, ProcessId_i, Void_c
- **tbl_jo_txn** - Job order headers
  - Key fields: TxnId_i, TxnDate_dd, ItemId_i, JoQty_d, QtyDone_d, DocRef_v, DocStatus_c, Void_c

#### Master Data
- **tbl_product_code** - Product/part master data
  - Key fields: StkId_i, ItemId_i, StkCode_v, ProdName_v, ProdDescr_v, UomId_i

### Forecasting Approach
- **Primary Method**: ForecasterAutoregMultiSeries with XGBoost
- **Strategy**: Recursive for short horizons (1-7 days), Direct for long horizons (14+ days)
- **Loss Function**: Asymmetric (stockouts cost 3x more than overstock)
- **Validation**: Time series cross-validation with TimeSeriesFold

## Critical Implementation Notes

1. **Multi-Series Approach**: Use one global model for all parts rather than individual models

2. **Feature Engineering**: 
   - Lag features (1, 7, 14, 30 days)
   - Cyclical encoding for seasonality
   - Intermittent demand handling
   - Lead time learning from purchase history

3. **Data Constraints**:
   - Never use inventory data (unreliable)
   - Focus on transaction flow (sales, purchases, consumption)
   - Handle missing dates and zero-consumption periods
   
4. **Production Monitoring**:
   - Track model drift
   - Automated retraining triggers
   - Custom inventory metrics (stockout rate, holding cost)

## Project Structure

```
prediction/
├── app/
│   ├── src/
│   │   ├── config/          # Database and app configuration
│   │   ├── data/           # Data extraction and processing
│   │   ├── forecasting/    # ML models and predictions
│   │   ├── reports/        # Report generation
│   │   └── utils/          # Utilities and helpers
│   ├── tests/              # Unit and integration tests
│   ├── notebooks/          # Jupyter notebooks for analysis
│   └── scripts/            # Production scripts
├── data/                   # Data storage (raw, processed, models)
├── logs/                   # Application logs
└── outputs/               # Generated reports
```

## Environment Variables

Create `.env` file in the app directory:
```
MARIADB_HOST=localhost
MARIADB_PORT=3306
MARIADB_DATABASE=nex_valiant
MARIADB_PASSWORD=mypassword
FORECAST_HORIZON=14
SAFETY_FACTOR=1.2
```

## Common SQL Patterns

When querying job orders for consumption:
```sql
SELECT 
    ji.ItemId_i as part_id,
    DATE(jt.TxnDate_dd) as date,
    SUM(ji.Qty_d) as consumption
FROM tbl_jo_item ji
JOIN tbl_jo_txn jt ON ji.TxnId_i = jt.TxnId_i
WHERE 
    ji.InOut_c = 'I'
    AND jt.TxnDate_dd >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
GROUP BY ji.ItemId_i, DATE(jt.TxnDate_dd)
```

## Key Files Reference

- `TODO.md`: Detailed implementation plan with checkboxes
- `WORKFLOW.md`: Complete system workflow and architecture
- `PROJECT_STRUCTURE.md`: Recommended directory structure
- `ACTIVITY_LOG.md`: Project history and decisions

## Claude Coding Guidelines (compulsory)

- Apply Ultrathink methodology to every problem-solving scenario
- Emoji not allow in .md file
- Not allow generate this emoji

## Development Standards

### Dependency Management
- **ONLY use uv, NEVER pip** for Python package management with pyproject.toml configuration
- **Installation**: `uv add package`
- **Running tools**: `uv run tool` 
- **Upgrading**: `uv add --dev package --upgrade-package package`
- **FORBIDDEN**: `uv pip install`, `@latest` syntax
- Leverage uv's fast dependency resolution and virtual environment management
- Structure projects with proper pyproject.toml metadata including build system, dependencies, and development dependencies

### Configuration Standards
- Implement configuration management through **pydantic-settings** for type-safe environment variable handling
- Create BaseSettings classes with proper field validation, default values, and comprehensive documentation
- Leverage pydantic's automatic type conversion and validation features for robust application configuration

### Frontend Framework
- Build user interfaces using **React** with modern functional components and hooks
- Implement component architecture with reusable modules, custom hooks for state management, and TypeScript for type safety
- Focus on responsive design patterns, efficient rendering optimization, and seamless integration with backend APIs through proper data fetching strategies
- Use **FastAPI** for backend REST API endpoints to serve the frontend

### Python Environment Setup
- Use **Python 3.12+** for all projects (currently requires-python = ">=3.12")
- Initialize projects using `uv init` with Python 3.12+ specification
- Maintain consistent Python version across development, testing, and production environments
- Leverage Python's performance improvements and modern language features for optimal development experience
- **Virtual Environment Activation**: To activate the virtual environment, run:
  ```bash
  cd /Users/carrickcheah/Project/prediction/app && source .venv/bin/activate
  ```

### Code Organization Principles
- **Alphabetical sorting within directories** - Maintain consistency
- **Business intelligence first** - Organize by business capability
- **Single responsibility** - Each module has clear business purpose
- **Autonomous operation** - Minimize human intervention requirements
- **Organizational learning** - Every operation improves future performance

### Documentation Standards (MANDATORY)
- **NO EMOJIS in documentation** - When generating README.md or any .md files, use only standard ASCII characters
- **Professional formatting only** - Use markdown headers, bullets, code blocks, and tables
- **Clear section markers** - Use standard markdown formatting like `##`, `###`, `**bold**`, `*italic*`
- **No decorative characters** - Avoid Unicode symbols, emojis, or special characters for decoration
- **Examples of prohibited characters**: ❌ ✅ 🚀 🎯 💡 🔧 📝 🏗️ 📊 🧪 📈 ⚡ 🎉 etc.
- **Use descriptive text instead**: Replace emojis with clear words (e.g., "Success" instead of "✅", "Error" instead of "❌")

### Testing Standards
- **Auto-cleanup test files** - Automatically remove test files after completing testing
- **Test file naming**: Use descriptive names like `test_feature_name.py`
- **Cleanup implementation**: Use `rm` command or Python's `os.remove()` after test completion
- **No test artifacts**: Ensure no temporary test files remain in the repository
- **Example**:
  ```python
  # After testing
  if os.path.exists("test_qdrant_match.py"):
      os.remove("test_qdrant_match.py")
  ```

### Error Handling Standards (MANDATORY - NO FALLBACKS)
- **NO FALLBACK MECHANISMS**: When errors occur, raise meaningful error messages immediately
- **STRICTLY FORBIDDEN**: Implementing fallback logic, mock responses, or alternative paths when primary functionality fails
- **REQUIRED**: All errors must be propagated up with clear, descriptive messages