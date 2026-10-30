"""Same pipeline as legacy/model-package/classifier/pipeline.py and the notebook."""
import numpy as np
import pandas as pd
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CATEGORICAL, NUMERICAL, RANDOM_STATE, TARGET


def build_pipeline(C: float = 0.0005) -> Pipeline:
    return Pipeline([
        ("categorical_imputation", CategoricalImputer(imputation_method="missing", variables=CATEGORICAL)),
        ("missing_indicator", AddMissingIndicator(variables=NUMERICAL)),
        ("median_imputation", MeanMedianImputer(imputation_method="median", variables=NUMERICAL)),
        ("rare_label_encoder", RareLabelEncoder(tol=0.05, n_categories=1, variables=CATEGORICAL)),
        ("categorical_encoder", OneHotEncoder(drop_last=True, variables=CATEGORICAL)),
        ("scaler", StandardScaler()),
        ("Logit", LogisticRegression(C=C, random_state=RANDOM_STATE)),
    ])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Map the public Kaggle HR dataset columns onto the names used in this project."""
    df = df.rename(columns={"left": TARGET, "sales": "dept", "Department": "dept"}).replace("?", np.nan)
    for c in NUMERICAL:
        df[c] = df[c].astype(float)
    return df
