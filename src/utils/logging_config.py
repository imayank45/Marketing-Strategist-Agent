"""Centralized logging configuration."""
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_file='logs/app.log', level=logging.INFO):
    """Configure logging with file and console output.
    
    Args:
        log_file (str): Path to log file.
        level (int): Logging level (e.g., logging.INFO).
        
    Returns:
        logging.Logger: Configured logger.
        
    Raises:
        IOError: If log file cannot be created.
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Avoid duplicate handlers
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.info(f"Logging initialized to {log_file}")
    return logger