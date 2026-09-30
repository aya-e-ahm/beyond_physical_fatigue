"""PFF event JSON -> StatsBomb-style ShotInput for the new xG model."""

from __future__ import annotations

from .features import PlayerPosition, ShotInput
from .model import predict_xg

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
SB_LENGTH = 120.0
SB_WIDTH = 80.0

# PFF codes -> model labels. Flagged in the notebook as things to double-check.
BODY_TYPE = {"R": "Right Foot", "L": "Left Foot", "HE": "Head", "O": "Other"}
SETPIECE_TYPE = {
    "O": "Open Play",
    "F": "Free Kick",
    "P": "Penalty",
    "C": "Corner",
    "K": "Open Play",
    "G": "Open Play",
    "T": "Open Play",
}
SHOT_TYPE_TO_TECHNIQUE = {
    "S": "Normal",
    "V": "Volley",
    "O": "Overhead Kick",
    "D": "Diving Header",
    "F": "Normal",
    "I": "Normal",
    "H": "Normal",
}
PERIOD_TO_GAME = {
    1: "First Half",
    2: "Second Half",
    3: "Extra Time 1",
    4: "Extra Time 2",
}

SHOT_OUTCOME = {
    "G": "Goal",
    "S": "Save on target",
    "B": "Block on target",
    "C": "Block off target",
    "O": "Off target",
    "F": "Save off target",
    "L": "Goalline clearance",
}


def pff_to_statsbomb(x: float, y: float, attacking_direction: str) -> tuple[float, float]:
    """Centre-origin PFF metres -> StatsBomb yards, always attacking x=120."""
    if str(attacking_direction).upper().startswith("L"):
        xm = PITCH_LENGTH / 2.0 - float(x)
        ym = PITCH_WIDTH / 2.0 - float(y)
    else:
        xm = float(x) + PITCH_LENGTH / 2.0
        ym = float(y) + PITCH_WIDTH / 2.0
    return xm * SB_LENGTH / PITCH_LENGTH, ym * SB_WIDTH / PITCH_WIDTH


def _xy(obj) -> tuple[float, float] | None:
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    x, y = obj.get("x"), obj.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def _technique(pe: dict) -> str:
    shot_type = pe.get("shotType")
    body = pe.get("bodyType")
    if shot_type == "D" and body != "HE":
        return "Normal"
    return SHOT_TYPE_TO_TECHNIQUE.get(shot_type, "Normal")


def pff_event_to_shot(event: dict) -> tuple[ShotInput, dict]:
    pe = event.get("possessionEvents") or {}
    ge = event.get("gameEvents") or {}
    sm = event.get("stadiumMetadata") or {}
    attacking = sm.get("teamAttackingDirection") or "R"

    ball_xy = _xy(event.get("ball"))
    shooter_id = pe.get("shooterPlayerId")
    shooter_team_id = ge.get("teamId")
    home_team = bool(ge.get("homeTeam"))

    shooter_xy = None
    players: list[PlayerPosition] = []
    for side, raw in (("home", event.get("homePlayers") or []), ("away", event.get("awayPlayers") or [])):
        is_home_side = side == "home"
        teammate = is_home_side == home_team
        for p in raw:
            xy = _xy(p)
            if xy is None:
                continue
            sb = pff_to_statsbomb(xy[0], xy[1], attacking)
            pos = "Goalkeeper" if str(p.get("positionGroupType", "")).upper() == "GK" else (
                str(p.get("positionGroupType") or "")
            )
            players.append(PlayerPosition(x=sb[0], y=sb[1], teammate=teammate, position=pos))
            if shooter_id is not None and str(p.get("playerId")) == str(shooter_id):
                shooter_xy = xy

    loc_pff = ball_xy or shooter_xy
    if loc_pff is None:
        raise ValueError("Shot event has no ball or shooter coordinates")
    loc_sb = pff_to_statsbomb(loc_pff[0], loc_pff[1], attacking)

    body = BODY_TYPE.get(pe.get("bodyType"), "Right Foot")
    shot_type = SETPIECE_TYPE.get(ge.get("setpieceType"), "Open Play")
    technique = _technique(pe)
    if body == "Head" and technique == "Diving Header":
        pass
    elif body != "Head" and technique == "Diving Header":
        technique = "Normal"

    clock_s = ge.get("startGameClock")
    if clock_s is None:
        clock_s = pe.get("gameClock")
    minute = int(round(float(clock_s) / 60.0)) if clock_s is not None else 45
    period = int(ge.get("period") or 1)
    game_period = PERIOD_TO_GAME.get(period, "First Half")

    shot = ShotInput(
        location=[loc_sb[0], loc_sb[1]],
        shot_body_part=body,
        shot_type=shot_type,
        shot_technique=technique,
        shot_one_on_one=0,
        under_pressure=1 if pe.get("pressureType") == "P" else 0,
        shot_first_time=1 if pe.get("shotType") in {"V", "F", "D"} else 0,
        game_period=game_period,
        minute=minute,
        players=players
        or [PlayerPosition(x=95, y=42, teammate=False, position="Goalkeeper")],
    )
    meta = {
        "shooter": pe.get("shooterPlayerName"),
        "shooter_id": shooter_id,
        "team": ge.get("teamName"),
        "team_id": shooter_team_id,
        "clock": ge.get("startFormattedGameClock") or pe.get("formattedGameClock"),
        "period": period,
        "outcome": pe.get("shotOutcomeType"),
        "outcome_label": SHOT_OUTCOME.get(pe.get("shotOutcomeType"), pe.get("shotOutcomeType")),
        "pff_shot_type": pe.get("shotType"),
        "pff_body": pe.get("bodyType"),
        "pff_setpiece": ge.get("setpieceType"),
        "pff_pressure": pe.get("pressureType"),
        "attacking_direction": attacking,
        "pff_x": loc_pff[0],
        "pff_y": loc_pff[1],
        "sb_x": loc_sb[0],
        "sb_y": loc_sb[1],
        "game_clock_s": float(clock_s) if clock_s is not None else None,
        "eventTime": event.get("eventTime"),
    }
    return shot, meta


def score_pff_shots(events: list[dict], shooter_name: str | None = None) -> list[dict]:
    rows = []
    for event in events:
        pe = event.get("possessionEvents") or {}
        if pe.get("possessionEventType") != "SH":
            continue
        name = pe.get("shooterPlayerName") or ""
        if shooter_name and shooter_name.lower() not in name.lower():
            continue
        shot, meta = pff_event_to_shot(event)
        xg, features = predict_xg(shot)
        rows.append({**meta, "xG": xg, "features": features})
    return rows
