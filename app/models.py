"""
PredictAPI — Pydantic Models & Schemas
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class ModelType(str, Enum):
    classification = "classification"
    regression = "regression"
    timeseries = "timeseries"

class PredictRequest(BaseModel):
    model_id: str = Field(..., description="Registered model identifier")
    features: Dict[str, Any] = Field(..., description="Input features as key-value pairs")
    return_proba: bool = Field(False, description="Return class probabilities")

class PredictResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    model_id: str
    latency_ms: float

class BatchPredictRequest(BaseModel):
    model_id: str
    instances: List[Dict[str, Any]]
    return_proba: bool = False

class BatchPredictResponse(BaseModel):
    predictions: List[Any]
    model_id: str
    count: int
    latency_ms: float

class ModelInfo(BaseModel):
    model_id: str
    model_type: ModelType
    version: str
    accuracy: float
    features: List[str]
    created_at: str
    description: str
