import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from attrition import api, config as cfg
from attrition.pipeline import build_pipeline, prepare

EMP = {"satisfaction_level": 0.1, "last_evaluation": 0.9, "number_project": 7, "average_montly_hours": 290,
       "time_spend_company": 4, "Work_accident": 0, "promotion_last_5years": 0, "dept": "sales", "salary": "low"}


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "satisfaction_level": rng.uniform(0, 1, n), "last_evaluation": rng.uniform(0.3, 1, n),
        "number_project": rng.integers(2, 8, n), "average_montly_hours": rng.integers(90, 310, n),
        "time_spend_company": rng.integers(2, 10, n), "Work_accident": rng.integers(0, 2, n),
        "promotion_last_5years": rng.integers(0, 2, n),
        "sales": rng.choice(["sales", "technical", "support", "IT"], n), "salary": rng.choice(["low", "medium", "high"], n),
    })
    df["left"] = (df.satisfaction_level < 0.3).astype(int)
    return prepare(df)


def test_prepare_renames_columns():
    df = synthetic(10)
    assert {"attrition", "dept"} <= set(df.columns) and "left" not in df.columns


def test_pipeline_learns_signal():
    df = synthetic()
    pipe = build_pipeline(C=1.0).fit(df[cfg.FEATURES], df[cfg.TARGET])
    p = pipe.predict_proba(df[cfg.FEATURES])[:, 1]
    assert p[df.attrition == 1].mean() > p[df.attrition == 0].mean() + 0.3


def test_api_predict(monkeypatch):
    df = synthetic()
    pipe = build_pipeline(C=1.0).fit(df[cfg.FEATURES], df[cfg.TARGET])
    monkeypatch.setattr(api, "load_model", lambda: (pipe, {"version": "1", "metrics": {}}))
    c = TestClient(api.app)
    r = c.post("/predict", json={"employees": [EMP, {**EMP, "satisfaction_level": 0.95}]})
    assert r.status_code == 200
    a, b = r.json()["predictions"]
    assert a["attrition_probability"] > b["attrition_probability"]
    assert c.post("/predict", json={"employees": [{**EMP, "salary": "huge"}]}).status_code == 422


def test_api_503_without_model(monkeypatch):
    def boom():
        raise RuntimeError("no registry")
    monkeypatch.setattr(api, "load_model", boom)
    assert TestClient(api.app).get("/health").status_code == 503
