from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config as cfg

app = FastAPI(title="Employee Attrition API", version="1.0.0",
              description="Serves the MLflow-registered attrition model (models:/employee-attrition@champion).")

Dept = Literal["sales", "accounting", "hr", "technical", "support", "management",
               "IT", "product_mng", "marketing", "RandD"]


class Employee(BaseModel):
    satisfaction_level: float = Field(..., ge=0, le=1, examples=[0.38])
    last_evaluation: float = Field(..., ge=0, le=1, examples=[0.53])
    number_project: int = Field(..., ge=0, le=20, examples=[2])
    average_montly_hours: float = Field(..., ge=0, le=400, examples=[157])
    time_spend_company: int = Field(..., ge=0, le=50, examples=[3])
    Work_accident: int = Field(..., ge=0, le=1, examples=[0])
    promotion_last_5years: int = Field(..., ge=0, le=1, examples=[0])
    dept: Dept = Field(..., examples=["sales"])
    salary: Literal["low", "medium", "high"] = Field(..., examples=["low"])


class Batch(BaseModel):
    employees: list[Employee] = Field(..., min_length=1, max_length=500)


@lru_cache(maxsize=1)
def load_model():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mv = MlflowClient().get_model_version_by_alias(cfg.MODEL_NAME, cfg.MODEL_ALIAS)
    run = MlflowClient().get_run(mv.run_id)
    model = mlflow.sklearn.load_model(f"models:/{cfg.MODEL_NAME}@{cfg.MODEL_ALIAS}")
    info = {"name": cfg.MODEL_NAME, "alias": cfg.MODEL_ALIAS, "version": mv.version, "run_id": mv.run_id,
            "params": {k: run.data.params[k] for k in ("model", "C", "n_train") if k in run.data.params},
            "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()}}
    return model, info


def _model():
    try:
        return load_model()
    except Exception as e:  # registry not ready / no model yet
        raise HTTPException(503, f"model not available: {e}")


@app.get("/health")
def health():
    _, info = _model()
    return {"status": "ok", "model_version": info["version"]}


@app.get("/model")
def model_info():
    return _model()[1]


@app.post("/predict")
def predict(batch: Batch):
    model, info = _model()
    df = pd.DataFrame([e.model_dump() for e in batch.employees])[cfg.FEATURES]
    for c in cfg.NUMERICAL:
        df[c] = df[c].astype(float)
    proba = model.predict_proba(df)[:, 1]
    return {
        "model_version": info["version"],
        "predictions": [{"attrition_probability": round(float(p), 4), "will_leave": bool(p >= 0.5),
                         "risk": "high" if p >= 0.5 else "medium" if p >= 0.25 else "low"} for p in proba],
    }
