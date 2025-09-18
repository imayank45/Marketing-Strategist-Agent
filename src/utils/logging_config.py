"""Centralized logging configuration."""
import logging

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s-%(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

# Create a function to get logger instance
def get_logger(name):
    """Get logger instance for a module.
    
    Args:
        name (str): Module name (e.g., __name__).
        
    Returns:
        logging.Logger: Configured logger.
    """
    return logging.getLogger(name)