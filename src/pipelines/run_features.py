"""Feature engineering pipeline."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_features(interim_path='data/interim/cleaned_bank.csv'):
    """Process features from interim data.
    
    Args:
        interim_path (str): Path to interim CSV.
        
    Returns:
        pd.DataFrame: Processed features + target.
        
    Raises:
        FileNotFoundError: If interim file missing.
    """
    try:
        df = pd.read_csv(interim_path)
        # Encode ALL categoricals to numeric (one-hot)
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Derived ROI proxy (numeric)
        df['ROI'] = df['duration'] / (df['campaign'] + 1)  # Avoid division by zero
        
        # Select numeric columns for scaling (exclude target 'y')
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('y', errors='ignore')
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        output_path = 'data/processed/processed_bank_features.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Features processed: {df.shape} saved to {output_path}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Interim file not found: {interim_path}")
        raise
    except Exception as e:
        logger.error(f"Feature error: {e}")
        raise ValueError("Feature engineering failed")