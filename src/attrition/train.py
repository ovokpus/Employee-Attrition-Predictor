"""Train the attrition pipeline, log the run to MLflow, register the model as @champion."""
import argparse
import time

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import config as cfg
from .pipeline import build_pipeline, prepare


def wait_for_server(uri: str, timeout: int = 120) -> None:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{uri}/health", timeout=3)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"MLflow server not reachable at {uri}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, nargs="+", default=[0.0005, 0.01, 1.0],
                    help="Regularization values to try; the best ROC AUC becomes @champion.")
    args = ap.parse_args()

    wait_for_server(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENT)

    df = prepare(pd.read_csv(cfg.DATA_URL))
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[cfg.FEATURES], df[cfg.TARGET], test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE, stratify=df[cfg.TARGET])

    best = None
    for C in args.C:
        with mlflow.start_run(run_name=f"logit-C{C}") as run:
            pipe = build_pipeline(C).fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = {
                "roc_auc": roc_auc_score(y_te, proba), "accuracy": accuracy_score(y_te, pred),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
            }
            mlflow.log_params({"model": "LogisticRegression", "C": C, "test_size": cfg.TEST_SIZE,
                               "n_train": len(X_tr), "features": ",".join(cfg.FEATURES)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, "model", signature=infer_signature(X_tr, proba),
                                     input_example=X_tr.head(3), registered_model_name=cfg.MODEL_NAME)
            print(f"C={C}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
            if best is None or metrics["roc_auc"] > best[0]:
                best = (metrics["roc_auc"], run.info.run_id)

    client = MlflowClient()
    version = next(v for v in client.search_model_versions(f"name='{cfg.MODEL_NAME}'") if v.run_id == best[1])
    client.set_registered_model_alias(cfg.MODEL_NAME, cfg.MODEL_ALIAS, version.version)
    print(f"Registered {cfg.MODEL_NAME} v{version.version} as @{cfg.MODEL_ALIAS} (roc_auc={best[0]:.3f})")


if __name__ == "__main__":
    main()
