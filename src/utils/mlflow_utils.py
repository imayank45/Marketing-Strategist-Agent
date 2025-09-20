"""MLflow utilities for experiment tracking."""
import mlflow
from src.utils.config import load_config
from src.utils.logging_config import setup_logging
import logging

logger = setup_logging()

def setup_mlflow(experiment_name="StrategyAgent"):
    """Set up MLflow tracking and experiment.
    
    Args:
        experiment_name (str): Name of the experiment.
        
    Raises:
        ValueError: If config load or MLflow setup fails.
    """
    try:
        config = load_config('configs/mlflow/backend_store_s3.yaml')
        tracking_uri = config.get('tracking_uri', 'file:///mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow set up with tracking URI: {tracking_uri}, experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        raise ValueError("MLflow configuration error")