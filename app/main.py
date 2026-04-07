"""
PredictAPI — ML Inference Service
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional
import asyncio, time, logging, joblib
from pathlib import Path

app = FastAPI(title="PredictAPI", description="ML Model Inference Service", version="2.0.0")
logger = logging.getLogger(__name__)
MODEL_REGISTRY: dict = {}

def load_models():
    model_dir = Path("models")
    if model_dir.exists():
        for path in model_dir.glob("*.pkl"):
            name = path.stem
            MODEL_REGISTRY[name] = joblib.load(path)
            logger.info(f"Loaded model: {name}")

@app.on_event("startup")
async def startup():
    load_models()
    logger.info(f"PredictAPI started — {len(MODEL_REGISTRY)} models loaded")

class PredictRequest(BaseModel):
    model_name: str
    features: list
    return_proba: bool = False

class PredictResponse(BaseModel):
    model_name: str
    predictions: list
    probabilities: Optional[list] = None
    latency_ms: float
    model_version: str = "1.0"

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if req.model_name not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model not found. Available: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[req.model_name]
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    import numpy as np
    X = np.array(req.features)
    preds = await loop.run_in_executor(None, model.predict, X)
    probas = None
    if req.return_proba and hasattr(model, 'predict_proba'):
        probas = (await loop.run_in_executor(None, model.predict_proba, X)).tolist()
    latency = (time.perf_counter() - start) * 1000
    return PredictResponse(
        model_name=req.model_name,
        predictions=preds.tolist(),
        probabilities=probas,
        latency_ms=round(latency, 2)
    )

@app.get("/models")
async def list_models():
    return {"models": list(MODEL_REGISTRY.keys()), "count": len(MODEL_REGISTRY)}

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(MODEL_REGISTRY)}
