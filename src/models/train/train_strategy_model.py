"""Train Random Forest for strategy predictions."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import mlflow
from src.utils.mlflow_utils import setup_mlflow
import joblib
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_strategy_model(processed_path='data/processed/processed_bank_features.csv'):
    """Train RF Classifier on processed features.
    
    Args:
        processed_path (str): Path to processed CSV.
        
    Returns:
        RandomForestClassifier: Fitted model.
        
    Raises:
        FileNotFoundError: If processed file missing.
        ValueError: If training fails.
    """
    try:
        setup_mlflow("StrategyModel")
        df = pd.read_csv(processed_path)
        X = df.drop('y', axis=1)
        y = df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        with mlflow.start_run():
            mlflow.log_param("model", "RFClassifier")
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(rf, "rf_strategy_model")
        
        joblib.dump(rf, 'models/rf_strategy_model.pkl')
        logger.info(f"Strategy model trained: Accuracy={acc:.4f}")
        return rf
    except FileNotFoundError as e:
        logger.error(f"Processed file not found: {processed_path}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise ValueError("RF training failed")