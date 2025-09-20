"""Configuration loader for project params."""
import yaml
from src.utils.logging_config import setup_logging

logger = setup_logging()

def load_config(file_path='configs/params.yaml'):
    """Load YAML config file.
    
    Args:
        file_path (str): Path to YAML file.
        
    Returns:
        dict: Loaded config.
        
    Raises:
        FileNotFoundError: If file not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {file_path}")
        return config
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise