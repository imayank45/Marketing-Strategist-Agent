"""FastAPI app for Strategy Agent with UI."""
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add CORS
from src.agents.strategy_agent import StrategyAgent
import mlflow
from src.utils.mlflow_utils import setup_mlflow
import logging
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Marketing Strategist Agent", version="1.0.0")
agent = StrategyAgent()
templates = Jinja2Templates(directory="templates")
setup_mlflow("StrategyAPI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing; restrict to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, etc.
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint for health check."""
    return {"message": "Marketing Strategist Agent API is running! Visit /strategy for UI."}

@app.get("/health")
def health():
    """Health check for ELB/Kubernetes."""
    return {"status": "healthy"}

@app.get("/strategy")
def strategy_form(request: Request):
    """Serve HTML form for strategy inputs."""
    return templates.TemplateResponse("strategy_form.html", {"request": request})

@app.post("/strategy")
async def get_strategy(request: Request):
    """Generate strategy from JSON POST."""
    try:
        data = await request.json()
        # Validate and convert
        age = int(data['age'])
        duration = int(data['duration'])
        campaign = int(data['campaign'])
        budget = int(data['budget'])
        
        # Validation (dataset ranges)
        if not (18 <= age <= 100):
            return JSONResponse(status_code=422, content={"error": "Age must be 18-100"})
        if not (1 <= duration <= 3600):
            return JSONResponse(status_code=422, content={"error": "Duration must be 1-3600 seconds"})
        if not (1 <= campaign <= 63):
            return JSONResponse(status_code=422, content={"error": "Campaign must be 1-63 contacts"})
        if budget < 1000:
            return JSONResponse(status_code=422, content={"error": "Budget must be at least $1000"})
        
        # Map to model features
        input_data = {
            "age": age,
            "job": data['job'],
            "marital": data['marital'],
            "duration": duration,
            "campaign": campaign,
            "contact": data['contact'],
            "month": data['month'],
            "budget": budget,
            "education": "university.degree",  # Default
            "default": "no",
            "housing": "no",
            "loan": "no",
            "pdays": 999,  # Default
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.1,
            "cons.price.idx": 93.8,
            "cons.conf.idx": -40,
            "euribor3m": 4.857,
            "nr.employed": 5191
        }
        result = agent.generate_strategy(**input_data)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_metric("success_prob", result['success_prob'])
            mlflow.log_param("age", age)
            mlflow.log_param("budget", budget)
        
        logger.info("Strategy API called via JSON")
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Strategy generation error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)