"""
PredictAPI — API Routes
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse, ModelInfo
from app.registry import ModelRegistry
import time, asyncio

router = APIRouter()
registry = ModelRegistry()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model = registry.get(request.model_id)
    if not model:
        raise HTTPException(404, f"Model '{request.model_id}' not found")

    start = time.perf_counter()
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, model.predict, request.features
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    latency = round((time.perf_counter() - start) * 1000, 2)

    return PredictResponse(
        prediction=result['prediction'],
        confidence=result.get('confidence'),
        probabilities=result.get('probabilities') if request.return_proba else None,
        model_id=request.model_id,
        latency_ms=latency
    )

@router.post("/predict/batch", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    model = registry.get(request.model_id)
    if not model:
        raise HTTPException(404, f"Model '{request.model_id}' not found")

    start = time.perf_counter()
    predictions = []
    for instance in request.instances:
        result = await asyncio.get_event_loop().run_in_executor(
            None, model.predict, instance
        )
        predictions.append(result['prediction'])

    latency = round((time.perf_counter() - start) * 1000, 2)
    return BatchPredictResponse(
        predictions=predictions,
        model_id=request.model_id,
        count=len(predictions),
        latency_ms=latency
    )

@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    return registry.list_all()

@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    info = registry.get_info(model_id)
    if not info:
        raise HTTPException(404, "Model not found")
    return info

@router.get("/health")
async def health():
    return {"status": "ok", "models_loaded": registry.count()}
