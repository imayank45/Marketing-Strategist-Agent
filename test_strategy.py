"""Test script for Strategy Agent."""
from src.agents.strategy_agent import StrategyAgent
import logging
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    try:
        logger.info("Starting strategy agent test")
        agent = StrategyAgent()
        
        # Dataset-aligned test inputs (Bank Marketing features)
        result = agent.generate_strategy(
            age=30,           # From dataset (e.g., 18-80)
            job='admin.',     # Categorical from dataset
            marital='single', # Categorical
            duration=500,     # Seconds
            campaign=1,       # Contacts
            contact='cellular', # Type
            month='may',      # Month
            budget=10000      # For allocation
        )
        logger.info("Test completed successfully")
        
        # Pretty-print
        print("=== Strategy Agent Test Results ===")
        print(f"Success Probability: {result['success_prob']:.2%}")
        print(f"Future Trend: {result['trend']:.0f} subscriptions")
        print(f"Budget Allocation: {json.dumps(result['allocation'], indent=2)}")
        print(f"Generated Strategy: {result['strategy']}")
        
    except FileNotFoundError as e:
        logger.error(f"Model file error: {e}")
        print(f"Error: {e} - Check models/rf_strategy_model.pkl")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error: {e}")
        raise