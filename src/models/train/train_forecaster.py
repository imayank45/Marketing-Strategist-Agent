"""Train Prophet forecaster for subscription trends."""
from prophet import Prophet
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error
from src.utils.mlflow_utils import setup_mlflow
from src.utils.logging_config import setup_logging

logger = setup_logging()

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
        logger.info(f"Starting forecaster training from {ts_path}")
        setup_mlflow("Forecasting")
        logger.debug(f"Loading TS data from {ts_path}")
        ts = pd.read_csv(ts_path)
        ts['ds'] = pd.to_datetime(ts['ds'])
        split_date = ts['ds'].max() - pd.DateOffset(months=3)
        train = ts[ts['ds'] < split_date]
        test = ts[ts['ds'] >= split_date]
        
        logger.debug("Fitting Prophet model")
        m = Prophet(changepoint_prior_scale=0.05)
        m.fit(train)
        
        future = m.make_future_dataframe(periods=len(test))
        forecast = m.predict(future)
        y_pred = forecast['yhat'].tail(len(test)).values
        mae = mean_absolute_error(test['y'], y_pred)
        
        with mlflow.start_run():
            logger.debug("Logging to MLflow")
            mlflow.log_param("model", "Prophet")
            mlflow.log_metric("mae", mae)
            mlflow.prophet.log_model(m, "prophet_model")
        
        logger.info(f"Forecaster trained: MAE={mae:.2f}")
        return m
    except FileNotFoundError as e:
        logger.error(f"TS file not found: {ts_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise ValueError("Prophet fitting failed")