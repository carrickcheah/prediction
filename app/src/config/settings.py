"""
Application settings using pydantic-settings for type-safe configuration.
"""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Database settings
    MARIADB_HOST: str = "localhost"
    MARIADB_PORT: int = 3306
    MARIADB_DATABASE: str = "nex_valiant"
    MARIADB_USER: str = "myuser"
    MARIADB_PASSWORD: str = "mypassword"
    
    # Model settings
    FORECAST_HORIZON: int = 14
    SAFETY_FACTOR: float = 1.2
    MODEL_LAGS: int = 30
    
    # Forecasting strategy
    SHORT_HORIZON_DAYS: int = 7
    LONG_HORIZON_DAYS: int = 14
    USE_ENSEMBLE: bool = True
    
    # Alert thresholds
    URGENT_THRESHOLD_DAYS: int = 2
    WARNING_THRESHOLD_DAYS: int = 7
    
    # Email settings
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USE_TLS: bool = True
    EMAIL_FROM: str = "inventory@company.com"
    EMAIL_TO: List[str] = ["procurement@company.com"]
    EMAIL_PASSWORD: str = ""
    
    # XGBoost parameters
    XGB_N_ESTIMATORS: int = 100
    XGB_MAX_DEPTH: int = 5
    XGB_LEARNING_RATE: float = 0.1
    XGB_RANDOM_STATE: int = 42
    
    # Performance settings
    N_JOBS: int = -1  # Use all CPU cores
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()