"""
Logging configuration with rotating file handlers.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import get_settings


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with rotating file handler.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(settings.LOG_LEVEL)
        
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger