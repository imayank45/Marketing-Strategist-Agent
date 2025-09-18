# test_strategy.py
import sys
import os
import logging
from src.agents.strategy_agent import StrategyAgent

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    # Initialize agent
    agent = StrategyAgent()
    logger.info("Strategy Agent initialized successfully")

    # Generate strategy with test inputs
    result = agent.generate_strategy('Email', 'Men 18-24', 30, 'Google Ads', 10000)
    logger.info("Strategy generated successfully")
    print("Strategy Result:", result)

except Exception as e:
    logger.error(f"Test failed: {e}")
    raise