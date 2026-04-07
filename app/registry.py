"""
PredictAPI — Model Registry
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
import joblib, os
from pathlib import Path
from typing import Optional

class ModelWrapper:
    def __init__(self, model, info: dict):
        self.model = model
        self.info = info

    def predict(self, features: dict) -> dict:
        import pandas as pd, numpy as np
        df = pd.DataFrame([features])
        pred = self.model.predict(df)[0]
        result = {'prediction': pred}
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(df)[0]
            classes = self.model.classes_
            result['confidence'] = float(max(proba))
            result['probabilities'] = {str(c): float(p) for c, p in zip(classes, proba)}
        return result

class ModelRegistry:
    def __init__(self):
        self._models: dict = {}
        self._load_from_disk()

    def _load_from_disk(self):
        models_dir = Path(os.getenv("MODELS_DIR", "./saved_models"))
        if not models_dir.exists():
            return
        for pkl in models_dir.glob("*.pkl"):
            meta_file = pkl.with_suffix('.json')
            if meta_file.exists():
                import json
                with open(meta_file) as f:
                    info = json.load(f)
                model = joblib.load(pkl)
                self._models[info['model_id']] = ModelWrapper(model, info)
                print(f"Loaded model: {info['model_id']}")

    def get(self, model_id: str) -> Optional[ModelWrapper]:
        return self._models.get(model_id)

    def get_info(self, model_id: str) -> Optional[dict]:
        m = self._models.get(model_id)
        return m.info if m else None

    def list_all(self) -> list:
        return [m.info for m in self._models.values()]

    def count(self) -> int:
        return len(self._models)
