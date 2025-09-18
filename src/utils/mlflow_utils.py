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
        ValueError: If config load fails.
    """
    try:
        config = load_config('configs/mlflow/backend_store_s3.yaml')
        mlflow.set_tracking_uri(config.get('tracking_uri', 'file:///mlruns'))
        mlflow.set_experiment(experiment_name)  # No artifact_uri here
        logger.info(f"MLflow set up for experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        raise ValueError("MLflow configuration error")