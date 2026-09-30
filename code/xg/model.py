"""Load the new XGBoost xG model and score shots."""

from __future__ import annotations

import warnings
from pathlib import Path

import xgboost as xgb

from .features import (
    FEATURE_COLUMNS,
    ShotInput,
    encode_categorical_features,
    prepare_features_for_model,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "xgboost_xg_model2.joblib"

_MODEL: xgb.XGBClassifier | None = None


def load_xg_model(path: Path | None = None) -> xgb.XGBClassifier:
    global _MODEL
    model_path = Path(path) if path is not None else MODEL_PATH
    if _MODEL is not None and path is None:
        return _MODEL
    model = xgb.XGBClassifier()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Unknown file format.*")
        model.load_model(str(model_path))
    if path is None:
        _MODEL = model
    return model


def predict_xg(shot: ShotInput, model: xgb.XGBClassifier | None = None) -> tuple[float, dict]:
    clf = model if model is not None else load_xg_model()
    features = prepare_features_for_model(shot)
    X = encode_categorical_features(features)
    xg = float(clf.predict_proba(X)[0, 1])
    return xg, features


def predict_xg_batch(shots: list[ShotInput], model: xgb.XGBClassifier | None = None) -> list[float]:
    clf = model if model is not None else load_xg_model()
    out = []
    for shot in shots:
        xg, _ = predict_xg(shot, model=clf)
        out.append(xg)
    return out


__all__ = ["FEATURE_COLUMNS", "MODEL_PATH", "load_xg_model", "predict_xg", "predict_xg_batch"]
