"""Run full training pipeline."""
from src.data.ingest_pipeline import ingest_pipeline
from src.models.train.train_forecaster import train_forecaster
from src.models.train.train_strategy_model import train_strategy_model
from src.utils.logging_config import setup_logging
import mlflow

logger = setup_logging()

if __name__ == "__main__":
    try:
        logger.info("Starting full training pipeline")
        logger.debug("Running ingest pipeline")
        ingest_pipeline('data/raw/bank.csv')
        logger.debug("Running forecaster training")
        forecaster = train_forecaster()
        logger.debug("Running strategy model training")
        strategy_model = train_strategy_model()
        logger.info("Full pipeline complete")

        # Verify MLflow runs
        runs = mlflow.search_runs()
        if runs.empty:
            logger.warning("No MLflow runs found")
        else:
            logger.info(f"MLflow runs found: {len(runs)}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise