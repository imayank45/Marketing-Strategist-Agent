"""MLflow utilities for experiment tracking."""
import mlflow
from src.utils.config import load_config
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def setup_mlflow(experiment_name="StrategyAgent"):
    """Set up MLflow tracking and experiment.
    
    Args:
        experiment_name (str): Name of the experiment.
        
    Raises:
        ValueError: If config load fails and no fallback works.
    """
    try:
        config = load_config('configs/mlflow/backend_store_s3.yaml')
        mlflow.set_tracking_uri(config.get('tracking_uri', 'http://localhost:5000'))
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow set up with tracking URI: {config.get('tracking_uri')}, experiment: {experiment_name}")
    except FileNotFoundError as e:
        logger.warning(f"Config file not found: {e}. Using default local tracking URI.")
        mlflow.set_tracking_uri("file:///app/mlruns")  # Fallback to local file store
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow set up with fallback URI: file:///app/mlruns, experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        raise ValueError("MLflow configuration error")