# src/models/predict/predict_forecast.py
"""Predict future trends using Prophet model."""
import pandas as pd
import mlflow
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def predict_forecast(model_uri, periods=30):
    """Generate forecast using loaded Prophet model.
    
    Args:
        model_uri (str): MLflow model URI (e.g., 'mlruns/0/artifacts/prophet_model').
        periods (int): Number of periods to forecast.
        
    Returns:
        pd.DataFrame: Forecast with 'ds' and 'yhat'.
        
    Raises:
        FileNotFoundError: If model not found.
        ValueError: If prediction fails.
    """
    try:
        model = mlflow.prophet.load_model(model_uri)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        logger.info(f"Forecast generated for {periods} periods")
        return forecast[['ds', 'yhat']]
    except FileNotFoundError as e:
        logger.error(f"Model not found at {model_uri}: {e}")
        raise
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise ValueError("Prediction failed")