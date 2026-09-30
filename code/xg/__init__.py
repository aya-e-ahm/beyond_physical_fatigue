"""Slim expected-goals extract: new XGBoost model only (philiprj/football_analytics)."""

from .model import FEATURE_COLUMNS, load_xg_model, predict_xg, predict_xg_batch
from .pff import pff_event_to_shot, score_pff_shots

__all__ = [
    "FEATURE_COLUMNS",
    "load_xg_model",
    "predict_xg",
    "predict_xg_batch",
    "pff_event_to_shot",
    "score_pff_shots",
]
