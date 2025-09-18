"""Train Prophet forecaster for subscription trends."""
from prophet import Prophet
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error
from src.utils.mlflow_utils import setup_mlflow
from src.utils.config import load_config  # Add this import
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_forecaster(ts_path='data/time_series/bank_ts.csv'):
    """Train Prophet model on time-series data.
    
    Args:
        ts_path (str): Path to TS CSV.
        
    Returns:
        Prophet: Fitted model.
        
    Raises:
        FileNotFoundError: If TS file missing.
        ValueError: If fitting fails.
    """
    try:
        setup_mlflow("Forecasting")
        config = load_config()  # Load config here for artifact_uri
        ts = pd.read_csv(ts_path)
        ts['ds'] = pd.to_datetime(ts['ds'])
        split_date = ts['ds'].max() - pd.DateOffset(months=3)
        train = ts[ts['ds'] < split_date]
        test = ts[ts['ds'] >= split_date]
        
        m = Prophet(changepoint_prior_scale=0.05)
        m.fit(train)
        
        future = m.make_future_dataframe(periods=len(test))
        forecast = m.predict(future)
        y_pred = forecast['yhat'].tail(len(test)).values
        mae = mean_absolute_error(test['y'], y_pred)
        
        with mlflow.start_run():
            mlflow.log_param("model", "Prophet")
            mlflow.log_metric("mae", mae)
            mlflow.prophet.log_model(m, "prophet_model")
        
        logger.info(f"Forecaster trained: MAE={mae:.2f}")
        return m
    except FileNotFoundError as e:
        logger.error(f"TS file not found: {ts_path}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise ValueError("Prophet fitting failed")