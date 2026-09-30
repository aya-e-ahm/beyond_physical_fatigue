"""Feature engineering for the new XGBoost xG model (model2).

Coordinates are StatsBomb: pitch 120 x 80, attacking the goal at (120, 40).
Dummy column names and order come from models/feature_columns2.joblib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent / "models"
FEATURE_COLUMNS = joblib.load(MODELS_DIR / "feature_columns2.joblib")

GOAL_X, GOAL_Y = 120.0, 40.0

# Values the model actually has columns for (after prefix).
BODY_PARTS = ("Left Foot", "Other", "Right Foot")
SHOT_TYPES = ("Free Kick", "Open Play", "Penalty")
TECHNIQUES = (
    "Diving Header",
    "Half Volley",
    "Lob",
    "Normal",
    "Overhead Kick",
    "Volley",
)
GAME_PERIODS = (
    "First Half",
    "First Half Ending",
    "Opening",
    "Second Half",
    "Second Half Beginning",
)

# Incoming API / StatsBomb-style labels that are not model columns.
BODY_PART_ALIASES = {"Head": "Other"}
SHOT_TYPE_ALIASES = {"Corner": "Free Kick", "Kick Off": "Open Play"}
TECHNIQUE_ALIASES = {"Backheel": "Normal"}
GAME_PERIOD_ALIASES = {
    "Extra Time 1": "Second Half Beginning",
    "Extra Time First Half": "Second Half Beginning",
    "Extra Time 2": "Second Half",
    "Extra Time Second Half": "Second Half",
}


@dataclass
class PlayerPosition:
    x: float
    y: float
    teammate: bool
    position: str = ""


@dataclass
class ShotInput:
    location: list[float]
    shot_body_part: str = "Right Foot"
    shot_type: str = "Open Play"
    shot_technique: str = "Normal"
    shot_one_on_one: int = 0
    under_pressure: int = 0
    shot_first_time: int = 0
    game_period: str = "First Half"
    minute: int = 46
    players: list[PlayerPosition] = field(
        default_factory=lambda: [PlayerPosition(x=95, y=42, teammate=False, position="Goalkeeper")]
    )


def _alias(value: str, aliases: dict[str, str], allowed: tuple[str, ...], default: str) -> str:
    v = aliases.get(value, value)
    return v if v in allowed else default


def calculate_distance_angle(x: float, y: float) -> tuple[float, float]:
    distance = float(np.sqrt((x - GOAL_X) ** 2 + (y - GOAL_Y) ** 2))
    angle = float(np.abs(np.arctan2(y - GOAL_Y, GOAL_X - x)))
    return distance, angle


def calculate_soft_blocking(shot_location, defenders) -> tuple[float, float]:
    x, y = shot_location
    soft_blocking = 0.0
    if defenders:
        for defender in defenders:
            defender_x, defender_y = defender["x"], defender["y"]
            if defender_x > x:
                shot_to_goal_angle = np.arctan2(GOAL_Y - y, GOAL_X - x)
                shot_to_defender_angle = np.arctan2(defender_y - y, defender_x - x)
                if abs(shot_to_goal_angle - shot_to_defender_angle) < 0.2:
                    distance_factor = 1 / (
                        1 + np.sqrt((defender_x - x) ** 2 + (defender_y - y) ** 2)
                    )
                    soft_blocking += distance_factor
        soft_blocking = min(float(soft_blocking), 1.0)
    return soft_blocking, soft_blocking * 0.8


def calculate_defender_features(shot_location, players) -> tuple[int, float, float, float]:
    x, y = shot_location
    defenders = [
        {"x": player.x, "y": player.y}
        for player in players
        if (not player.teammate) and player.position != "Goalkeeper"
    ]
    defenders_in_path = 0
    min_defender_distance = 100.0
    for defender in defenders:
        player_x, player_y = defender["x"], defender["y"]
        if player_x > x:
            shot_to_goal_angle = np.arctan2(GOAL_Y - y, GOAL_X - x)
            shot_to_defender_angle = np.arctan2(player_y - y, player_x - x)
            if abs(shot_to_goal_angle - shot_to_defender_angle) < 0.3:
                defenders_in_path += 1
        defender_distance = float(np.sqrt((player_x - x) ** 2 + (player_y - y) ** 2))
        min_defender_distance = min(min_defender_distance, defender_distance)
    if min_defender_distance == 100.0:
        min_defender_distance = 20.0
    soft_blocking, goal_blocking_percentage = calculate_soft_blocking(shot_location, defenders)
    return defenders_in_path, min_defender_distance, soft_blocking, goal_blocking_percentage


def calculate_gk_features(shot_location, players) -> tuple[float, float, float]:
    x, y = shot_location
    goalkeeper = next(
        (player for player in players if (not player.teammate) and player.position == "Goalkeeper"),
        None,
    )
    gk_distance = 15.0
    gk_angle = 0.0
    gk_goal_line_distance = 5.0
    if goalkeeper:
        gk_x, gk_y = goalkeeper.x, goalkeeper.y
        gk_distance = float(np.sqrt((gk_x - x) ** 2 + (gk_y - y) ** 2))
        gk_goal_line_distance = float(GOAL_X - gk_x)
        shot_to_goal_angle = np.arctan2(GOAL_Y - y, GOAL_X - x)
        shot_to_gk_angle = np.arctan2(gk_y - y, gk_x - x)
        gk_angle = float(abs(shot_to_goal_angle - shot_to_gk_angle))
    return gk_distance, gk_angle, gk_goal_line_distance


def prepare_features_for_model(shot: ShotInput) -> dict:
    x, y = float(shot.location[0]), float(shot.location[1])
    distance, angle = calculate_distance_angle(x, y)
    defenders_in_path, min_defender_distance, soft_blocking, goal_blocking_percentage = (
        calculate_defender_features(shot.location, shot.players)
    )
    gk_distance, gk_angle, gk_goal_line_distance = calculate_gk_features(shot.location, shot.players)
    return {
        "x": x,
        "y": y,
        "distance": distance,
        "angle": angle,
        "shot_body_part": _alias(shot.shot_body_part, BODY_PART_ALIASES, BODY_PARTS, "Right Foot"),
        "shot_type": _alias(shot.shot_type, SHOT_TYPE_ALIASES, SHOT_TYPES, "Open Play"),
        "shot_technique": _alias(shot.shot_technique, TECHNIQUE_ALIASES, TECHNIQUES, "Normal"),
        "shot_one_on_one": 1 if shot.shot_one_on_one else 0,
        "under_pressure": 1 if shot.under_pressure else 0,
        "first_time": 1 if shot.shot_first_time else 0,
        "game_period": _alias(shot.game_period, GAME_PERIOD_ALIASES, GAME_PERIODS, "First Half"),
        "minute": int(shot.minute),
        "defenders_in_path": defenders_in_path,
        "min_defender_distance": min_defender_distance,
        "soft_blocking": soft_blocking,
        "goal_blocking_percentage": goal_blocking_percentage,
        "gk_distance": gk_distance,
        "gk_angle": gk_angle,
        "gk_goal_line_distance": gk_goal_line_distance,
    }


def encode_categorical_features(features_dict: dict) -> pd.DataFrame:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    for key in (
        "x",
        "y",
        "distance",
        "angle",
        "shot_one_on_one",
        "under_pressure",
        "first_time",
        "minute",
        "defenders_in_path",
        "min_defender_distance",
        "soft_blocking",
        "goal_blocking_percentage",
        "gk_distance",
        "gk_angle",
        "gk_goal_line_distance",
    ):
        if key in row:
            row[key] = features_dict[key]
    mapping = {
        f"body_part_{features_dict['shot_body_part']}": 1.0,
        f"shot_type_{features_dict['shot_type']}": 1.0,
        f"technique_{features_dict['shot_technique']}": 1.0,
        f"game_period_{features_dict['game_period']}": 1.0,
    }
    for col, val in mapping.items():
        if col in row:
            row[col] = val
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
