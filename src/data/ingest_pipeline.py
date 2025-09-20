"""Data ingestion pipeline for Bank Marketing dataset."""
import pandas as pd
from src.utils.config import load_config
from src.utils.logging_config import setup_logging

logger = setup_logging()

def ingest_pipeline(raw_path='data/raw/bank.csv'):
    """Ingest and clean raw data.
    
    Args:
        raw_path (str): Path to raw CSV.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
        
    Raises:
        FileNotFoundError: If raw file missing.
        ValueError: If cleaning fails.
    """
    try:
        logger.info(f"Starting ingestion from {raw_path}")
        config = load_config()
        df = pd.read_csv(raw_path, sep=';')
        df.replace('unknown', pd.NA, inplace=True)
        df.fillna(df.mode().iloc[0], inplace=True)
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        output_path = 'data/interim/cleaned_bank.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Ingest complete: {df.shape} rows saved to {output_path}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Raw file not found: {raw_path}")
        raise
    except Exception as e:
        logger.error(f"Ingest pipeline error: {e}")
        raise ValueError("Data ingestion failed")