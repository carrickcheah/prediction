#!/usr/bin/env python
"""
Daily forecast script to be run via cron job.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import run_daily_forecast


if __name__ == "__main__":
    try:
        run_daily_forecast()
        sys.exit(0)
    except Exception as e:
        print(f"Daily forecast failed: {str(e)}")
        sys.exit(1)