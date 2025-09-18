"""Strategy Agent using CrewAI and models."""
from crewai import Agent, Task, Crew
from src.models.predict.predict_strategy import predict_strategy
from src.models.predict.predict_forecast import predict_forecast
import joblib
import mlflow
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class StrategyAgent:
    """Class for generating marketing strategies."""
    def __init__(self):
        """Initialize agent with models."""
        try:
            self.strategy_model = joblib.load('models/rf_strategy_model.pkl')
            config = load_config('configs/mlflow/backend_store_s3.yaml')
            model_uri = config.get('artifact_uri', 'mlruns/0/artifacts/prophet_model').replace('/', '\\')
            self.forecaster = mlflow.prophet.load_model(model_uri)
            logger.info("Strategy Agent initialized")
        except Exception as e:
            logger.error(f"Init error: {e}")
            raise ValueError("Agent initialization failed")

    def generate_strategy(self, campaign_type, audience, duration, channel, budget):
        """Generate strategy with predictions.
        
        Args:
            campaign_type (str): Type (e.g., 'Email').
            audience (str): Audience (e.g., 'Men 18-24').
            duration (int): Days.
            channel (str): Channel (e.g., 'Google Ads').
            budget (int): Budget.
        
        Returns:
            dict: {'success_prob': float, 'trend': float, 'strategy': str}.
        
        Raises:
            ValueError: If generation fails.
        """
        try:
            features = {'duration': duration, 'campaign': 1, 'age': 30}
            success_prob = predict_strategy(features)
            forecast = predict_forecast(model_uri, periods=duration)
            future_trend = forecast['yhat'].tail(4).mean()
            
            agent = Agent(role='Strategy', goal='Generate plan', llm='groq/llama3-70b-8192')
            task_desc = f"Campaign: {campaign_type}, Audience: {audience}, Duration: {duration}, Channel: {channel}, Budget: {budget}. Success prob: {success_prob:.2f}, Trend: {future_trend:.0f}."
            task = Task(description=task_desc, agent=agent)
            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()
            
            output = {'success_prob': success_prob, 'trend': future_trend, 'strategy': result}
            logger.info(f"Strategy generated: prob={success_prob:.2f}")
            return output
        except Exception as e:
            logger.error(f"Strategy generation error: {e}")
            raise ValueError("Strategy failed")