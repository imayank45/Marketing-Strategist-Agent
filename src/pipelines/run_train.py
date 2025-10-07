"""Run full training pipeline."""
from src.data.ingest_pipeline import ingest_pipeline
from src.models.train.train_forecaster import train_forecaster
from src.models.train.train_strategy_model import train_strategy_model
import logging
import mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    try:
        logger.info("Starting full training pipeline")
        
        # Ingest data
        df = ingest_pipeline('data/raw/bank.csv')
        
        # Train forecaster
        forecaster = train_forecaster()
        
        # Train strategy model
        strategy_model = train_strategy_model()
        
        logger.info("Full pipeline complete")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up MLflow run context
        mlflow.end_run()  # Ensure no active run lingers
        logger.debug("MLflow run context cleaned up")