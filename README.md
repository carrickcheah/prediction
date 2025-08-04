# Inventory Forecasting System

An advanced machine learning system for predicting inventory needs using XGBoost and multi-series forecasting techniques. The system analyzes part consumption patterns, sales data, and purchase orders to prevent stockouts and reduce excess inventory.

## Overview

This system addresses critical inventory management challenges:
- **Prevent Stockouts**: Reduce stockouts by 50% within 3 months
- **Reduce Excess Inventory**: Decrease excess inventory by 30%
- **Automate Ordering**: Save 2-4 hours daily of manual checking
- **Scale to 6000+ Parts**: Handle entire inventory with single model

## Quick Start

### Prerequisites
- Python 3.12+
- MariaDB/MySQL database
- UV package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd prediction
```

2. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Set up the environment:
```bash
cd app
uv init
uv sync
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

5. Run the system:
```bash
uv run python src/main.py run
```

## Project Structure

```
prediction/
├── app/                    # Application code
│   ├── src/               # Source code
│   ├── tests/             # Unit tests
│   ├── notebooks/         # Jupyter notebooks
│   └── scripts/           # Production scripts
├── data/                  # Data storage
├── logs/                  # Application logs
├── outputs/               # Generated reports
└── docs/                  # Documentation
```

## Documentation

- [Claude Guidelines](CLAUDE.md) - Instructions for Claude AI assistants
- [Project Structure](PROJECT_STRUCTURE.md) - Detailed directory layout
- [Workflow](WORKFLOW.md) - System workflow and architecture
- [TODO](TODO.md) - Implementation roadmap
- [Activity Log](ACTIVITY_LOG.md) - Project history

## Development

### Running Tests
```bash
cd app
uv run pytest
```

### Running with Docker
```bash
docker-compose up
```

### Daily Forecast Script
```bash
uv run python scripts/run_daily_forecast.py
```

## License

Proprietary and confidential.