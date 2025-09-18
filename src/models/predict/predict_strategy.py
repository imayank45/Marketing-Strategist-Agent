"""Predict strategy success probability."""
import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def predict_strategy(features_dict):
    """Predict high ROI probability from features.
    
    Args:
        features_dict (dict): Input features (e.g., {'age': 30, 'duration': 500}).
        
    Returns:
        float: Probability (0-1).
        
    Raises:
        FileNotFoundError: If model pickle missing.
        ValueError: If features mismatch.
    """
    try:
        model = joblib.load('models/rf_strategy_model.pkl')
        features = pd.DataFrame([features_dict])
        features = pd.get_dummies(features, drop_first=True)
        features = features.reindex(columns=model.feature_names_in_, fill_value=0)
        prob = model.predict_proba(features)[0][1]
        logger.info(f"Predicted success prob: {prob:.4f}")
        return prob
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise ValueError("Strategy prediction failed")