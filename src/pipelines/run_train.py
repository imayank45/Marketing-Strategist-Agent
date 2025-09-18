"""Run full training pipeline."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Path fix

from src.data.ingest_pipeline import ingest_pipeline
from src.pipelines.run_features import run_features  # Add this
from src.models.train.train_forecaster import train_forecaster
from src.models.train.train_strategy_model import train_strategy_model
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    try:
        ingest_pipeline('data/raw/bank.csv')
        run_features()  # Process to create processed_bank_features.csv
        train_forecaster()
        train_strategy_model()
        logger.info("Full pipeline complete")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise