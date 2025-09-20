"""Strategy Agent using CrewAI and models."""
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI  # OpenAI LLM object for CrewAI
from src.models.predict.predict_strategy import predict_strategy
from src.models.predict.predict_forecast import predict_forecast
from src.utils.logging_config import setup_logging
from src.utils.mlflow_utils import setup_mlflow
import logging
import joblib
import mlflow
import os
from typing import Dict

logger = setup_logging()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class StrategyAgent:
    """Class for generating marketing strategies."""
    def __init__(self):
        """Initialize agent with models.
        
        Raises:
            ValueError: If model loading fails.
        """
        try:
            logger.info("Initializing Strategy Agent")
            setup_mlflow("Forecasting")  # Ensure MLflow context
            logger.debug("MLflow context set for Forecasting")

            # Load RF model
            self.strategy_model = joblib.load('models/rf_strategy_model.pkl')
            logger.debug("RF model loaded successfully")

            # Dynamically find Prophet model from latest run
            runs = mlflow.search_runs(filter_string="params.model = 'Prophet'")
            if runs.empty:
                raise FileNotFoundError("No Prophet model run found in MLflow")
            latest_run = runs.sort_values("start_time", ascending=False).iloc[0]
            self.artifact_path = f"runs:/{latest_run['run_id']}/prophet_model"
            self.forecaster = mlflow.prophet.load_model(self.artifact_path)
            logger.debug(f"Prophet model loaded from {self.artifact_path}")

            # Create OpenAI LLM object (required for CrewAI)
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Cost-effective; use "gpt-4o" for better quality
                api_key=openai_key
            )
            logger.debug("OpenAI LLM initialized")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise ValueError("Model initialization failed")
        except Exception as e:
            logger.error(f"Init error: {e}", exc_info=True)
            raise

    def generate_strategy(self, **kwargs) -> Dict:
        """Generate strategy with predictions, aligned with Bank Marketing dataset.
        
        Args:
            **kwargs: Dataset features (e.g., age=30, job='admin.', marital='single', duration=500, campaign=1, contact='cellular', month='may', budget=10000, and defaults like education='university.degree').
        
        Returns:
            dict: {'success_prob': float, 'trend': float, 'strategy': str, 'allocation': dict}.
        
        Raises:
            ValueError: If generation fails.
        """
        try:
            logger.info(f"Generating strategy with features: {kwargs}")
            
            # Use provided kwargs; fill defaults for missing Bank Marketing features
            features = {
                'age': kwargs.get('age', 30),
                'job': kwargs.get('job', 'admin.'),
                'marital': kwargs.get('marital', 'single'),
                'duration': kwargs.get('duration', 500),
                'campaign': kwargs.get('campaign', 1),
                'contact': kwargs.get('contact', 'cellular'),
                'month': kwargs.get('month', 'may'),
                'education': kwargs.get('education', 'university.degree'),
                'default': kwargs.get('default', 'no'),
                'housing': kwargs.get('housing', 'no'),
                'loan': kwargs.get('loan', 'no'),
                'pdays': kwargs.get('pdays', 999),
                'previous': kwargs.get('previous', 0),
                'poutcome': kwargs.get('poutcome', 'nonexistent'),
                'emp.var.rate': kwargs.get('emp.var.rate', 1.1),
                'cons.price.idx': kwargs.get('cons.price.idx', 93.8),
                'cons.conf.idx': kwargs.get('cons.conf.idx', -40),
                'euribor3m': kwargs.get('euribor3m', 4.857),
                'nr.employed': kwargs.get('nr.employed', 5191),
                'budget': kwargs.get('budget', 10000)  # For allocation, not RF
            }
            success_prob = predict_strategy(features)  # Pass full features dict to RF
            logger.debug(f"Success probability: {success_prob}")

            forecast = predict_forecast(self.artifact_path, periods=kwargs.get('duration', 30))
            future_trend = forecast['yhat'].tail(4).mean()
            logger.debug(f"Future trend: {future_trend}")

            # Agent with backstory and OpenAI LLM object
            agent = Agent(
                role='Strategy',
                goal='Generate plan based on dataset features',
                backstory='Expert in bank marketing campaigns, using age/job/marital/duration/campaign/contact/month to predict subscription success and allocate budgets.',
                llm=self.llm
            )
            task_desc = f"Age: {features['age']}, Job: {features['job']}, Marital: {features['marital']}, Duration: {features['duration']}s, Campaign: {features['campaign']}, Contact: {features['contact']}, Month: {features['month']}, Budget: {features['budget']}. Success prob: {success_prob:.2f}, Trend: {future_trend:.0f}."
            task = Task(
                description=task_desc,
                agent=agent,
                expected_output='Detailed strategy with budget split, considering dataset features like duration and job for high subscription prob.'
            )
            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()

            # Allocation based on prob
            allocation = {'primary': features['contact'], 'budget_split': {'primary': int(features['budget'] * 0.6), 'secondary': int(features['budget'] * 0.4)}} if success_prob > 0.5 else {'primary': 'telephone', 'budget_split': {'testing': int(features['budget'] * 0.7), 'low_risk': int(features['budget'] * 0.3)}}

            output = {'success_prob': success_prob, 'trend': future_trend, 'strategy': result, 'allocation': allocation}
            logger.info(f"Strategy generated with prob {success_prob:.2f}")
            return output
        except Exception as e:
            logger.error(f"Strategy generation error: {e}", exc_info=True)
            raise ValueError("Strategy generation failed")