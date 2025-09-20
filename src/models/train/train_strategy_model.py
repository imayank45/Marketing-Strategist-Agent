"""Train Random Forest for strategy predictions on Bank Marketing dataset."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pandas as pd
import mlflow
from src.utils.mlflow_utils import setup_mlflow
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_strategy_model(processed_path='data/processed/processed_bank_features.csv'):
    """Train RF Classifier and log classification metrics to MLflow.
    
    Args:
        processed_path (str): Path to processed CSV with 'y' as binary target.
        
    Returns:
        RandomForestClassifier: Fitted model.
        
    Raises:
        FileNotFoundError: If processed file missing.
        ValueError: If training or metrics computation fails.
    """
    try:
        setup_mlflow("StrategyModel")
        df = pd.read_csv(processed_path)
        X = df.drop('y', axis=1)  # Features (age, job, etc.)
        y = df['y']  # Binary target (subscription yes/no)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Classification Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion Matrix as Image
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - RF Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = 'confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("model", "RFClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", precision)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("baseline_rate", y.mean())  # 8.09% for comparison
            mlflow.log_artifact(cm_path)  # Confusion matrix image
            mlflow.log_dict(report, "classification_report.json")  # Full report as artifact
            mlflow.sklearn.log_model(rf, "rf_strategy_model")
        
        joblib.dump(rf, 'models/rf_strategy_model.pkl')
        logger.info(f"Strategy model trained: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        return rf
    except FileNotFoundError as e:
        logger.error(f"Processed file not found: {processed_path}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise ValueError("RF training failed")

if __name__ == "__main__":
    train_strategy_model()