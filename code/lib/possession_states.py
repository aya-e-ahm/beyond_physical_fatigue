"""
Possession-state segmentation for PFF events + tracking.

Method A — event two-layer baseline (player OTB + continuous team possession).
  Aligns OTB to tracking via start_frame/end_frame when available (Tracking Spec);
  fills same-team gaps when consistent with PFF sequence; uses tracking-native
  home_ball only as validation / fallback (not in events.json).
Method B — tracking possession zone (Vidal-Codina-style) with PFF-calibrated Rpz.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

PLAYER_FOCAL = "FOCAL"
PLAYER_TEAMMATE = "TEAMMATE"
PLAYER_OPPONENT = "OPPONENT"
PLAYER_NONE = "NONE"
PLAYER_DUEL = "DUEL"

TEAM_FOCAL = "FOCAL"
TEAM_OPP = "OPP"
TEAM_DEAD = "DEAD"
TEAM_UNRESOLVED = "UNRESOLVED"

CTX_PLAYER_ON_BALL = "PLAYER_ON_BALL"
CTX_TEAM_OFF_BALL = "TEAM_POSSESSION_OFF_BALL"
CTX_OPPOSITION = "OPPOSITION_POSSESSION"
CTX_DEAD = "BALL_OUT_OR_DEAD"
CTX_UNRESOLVED = "UNRESOLVED"

CONTROL = "CONTROL"
DUEL = "DUEL"
FREE = "FREE"
DEAD = "DEAD"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and "homeTeam" in data[0]:
        return data[0]
    return data


def load_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.json_normalize(raw)
    for col in ("eventTime", "startTime", "endTime", "duration", "sequence"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "gameEvents.period" in df.columns:
        df["gameEvents.period"] = pd.to_numeric(df["gameEvents.period"], errors="coerce")
    if "gameEvents.playerId" in df.columns:
        df["gameEvents.playerId"] = pd.to_numeric(df["gameEvents.playerId"], errors="coerce")
    if "gameEvents.teamId" in df.columns:
        df["gameEvents.teamId"] = pd.to_numeric(df["gameEvents.teamId"], errors="coerce")
    return df


def load_roster(path: str | Path, match_id: str | None = None) -> pd.DataFrame:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.json_normalize(raw)
    df = df.rename(
        columns={
            "player.id": "player_id",
            "player.nickname": "nickname",
            "team.id": "team_id",
            "team.name": "team",
            "positionGroupType": "position",
            "shirtNumber": "shirt_number",
        }
    )
    df["player_id"] = df["player_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    df["shirt_number"] = df["shirt_number"].astype(int)
    if match_id is not None:
        df["match_id"] = str(match_id)
    return df


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def side_for_team(meta: Mapping, team_id: str | int) -> str:
    home_id = str((meta.get("homeTeam") or {}).get("id"))
    return "home" if str(team_id) == home_id else "away"


def median_frame_dt(video_time_s: pd.Series) -> float:
    t = pd.to_numeric(video_time_s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) < 2:
        return 1.0 / 29.97
    dt = np.diff(np.sort(t))
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 1.0 / 29.97
    return float(np.median(dt))


# ---------------------------------------------------------------------------
# Method A — OTB intervals + team possession
# ---------------------------------------------------------------------------


def extract_tracking_game_event_index(
    tracking_path: str | Path,
    cache_path: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Scan tracking JSONL for per-game-event frame spans and tracking-only fields.

    Tracking Spec v2.2 attaches ``game_event`` (incl. ``home_ball``, ``sequence``,
    ``start_frame`` / ``end_frame``) only on frames inside the event. ``home_ball``
    does **not** appear in events.json — it is tracking-native.

    Returns one row per ``game_event_id`` with min/max frames and first metadata.
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    rows: dict[Any, dict] = {}
    for fr in iter_jsonl(tracking_path):
        gid = fr.get("game_event_id")
        ge = fr.get("game_event")
        if gid is None or not isinstance(ge, dict):
            continue
        gid_key = gid
        fnum = fr.get("frameNum")
        if fnum is None:
            continue
        fnum = int(fnum)
        if gid_key not in rows:
            rows[gid_key] = {
                "gameEventId": gid_key,
                "game_event_type": ge.get("game_event_type"),
                "frame0": fnum,
                "frame1": fnum,
                "home_ball": ge.get("home_ball"),
                "sequence_tracking": ge.get("sequence"),
                "team_id_tracking": ge.get("team_id"),
                "player_id_tracking": ge.get("player_id"),
                "start_frame_meta": ge.get("start_frame"),
                "end_frame_meta": ge.get("end_frame"),
            }
        else:
            r = rows[gid_key]
            r["frame0"] = min(r["frame0"], fnum)
            r["frame1"] = max(r["frame1"], fnum)

    out = pd.DataFrame(list(rows.values()))
    if not out.empty:
        # Prefer explicit start/end_frame from the event dict when present
        sf = pd.to_numeric(out["start_frame_meta"], errors="coerce")
        ef = pd.to_numeric(out["end_frame_meta"], errors="coerce")
        out["frame0"] = np.where(sf.notna(), sf, out["frame0"]).astype(int)
        out["frame1"] = np.where(ef.notna(), ef, out["frame1"]).astype(int)
        out["gameEventId"] = pd.to_numeric(out["gameEventId"], errors="coerce")
        out["sequence_tracking"] = pd.to_numeric(out["sequence_tracking"], errors="coerce")
        out["team_id_tracking"] = pd.to_numeric(out["team_id_tracking"], errors="coerce")
        out["player_id_tracking"] = pd.to_numeric(out["player_id_tracking"], errors="coerce")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def enrich_otb_with_tracking_index(otb: pd.DataFrame, ge_index: pd.DataFrame) -> pd.DataFrame:
    """Attach frame0/frame1/home_ball from tracking game-event index onto OTB intervals."""
    out = otb.copy()
    # Drop placeholders so merge does not create frame0_x / frame0_y.
    for col in ("frame0", "frame1", "home_ball", "sequence_tracking"):
        if col in out.columns:
            out = out.drop(columns=[col])
    if out.empty or ge_index is None or ge_index.empty:
        out["frame0"] = np.nan
        out["frame1"] = np.nan
        out["home_ball"] = np.nan
        return out
    keep = ge_index[
        ["gameEventId", "frame0", "frame1", "home_ball", "sequence_tracking"]
    ].drop_duplicates("gameEventId")
    keep = keep.copy()
    keep["gameEventId"] = pd.to_numeric(keep["gameEventId"], errors="coerce")
    out["gameEventId"] = pd.to_numeric(out["gameEventId"], errors="coerce")
    out = out.merge(keep, on="gameEventId", how="left")
    if "sequence" in out.columns:
        out["sequence"] = pd.to_numeric(out["sequence"], errors="coerce")
        out["sequence"] = out["sequence"].fillna(out["sequence_tracking"])
    return out


def build_otb_intervals(events: pd.DataFrame, epsilon_s: float | None = None) -> pd.DataFrame:
    """Collapse OTB rows by gameEventId into one interval per possession event."""
    if epsilon_s is None:
        epsilon_s = 1.0 / 29.97

    otb = events.loc[events["gameEvents.gameEventType"] == "OTB"].copy()
    if otb.empty:
        return pd.DataFrame(
            columns=[
                "gameEventId",
                "t0",
                "t1",
                "duration_pff",
                "is_instant_pff",
                "player_id",
                "team_id",
                "player_name",
                "team_name",
                "sequence",
                "period",
                "poss_types",
                "n_rows",
                "frame0",
                "frame1",
                "home_ball",
            ]
        )

    def _types(s: pd.Series) -> str:
        vals = [str(x) for x in s.dropna().unique()]
        return ",".join(vals)

    g = (
        otb.groupby("gameEventId", sort=False)
        .agg(
            t0=("startTime", "min"),
            t1=("endTime", "max"),
            duration_pff=("duration", "max"),
            player_id=("gameEvents.playerId", "first"),
            team_id=("gameEvents.teamId", "first"),
            player_name=("gameEvents.playerName", "first"),
            team_name=("gameEvents.teamName", "first"),
            sequence=("sequence", "first"),
            period=("gameEvents.period", "first"),
            poss_types=("possessionEvents.possessionEventType", _types),
            n_rows=("gameEventId", "size"),
        )
        .reset_index()
    )
    g["player_id"] = pd.to_numeric(g["player_id"], errors="coerce")
    g["team_id"] = pd.to_numeric(g["team_id"], errors="coerce")
    g["sequence"] = pd.to_numeric(g["sequence"], errors="coerce")
    g["t0"] = pd.to_numeric(g["t0"], errors="coerce")
    g["t1"] = pd.to_numeric(g["t1"], errors="coerce")
    g["duration_pff"] = pd.to_numeric(g["duration_pff"], errors="coerce")
    # Prefer end-start when duration missing
    span = g["t1"] - g["t0"]
    g["duration_pff"] = g["duration_pff"].fillna(span)
    g["is_instant_pff"] = (g["duration_pff"].fillna(0) <= 0) | (span.fillna(0) <= 0)
    g.loc[g["is_instant_pff"], "t1"] = g.loc[g["is_instant_pff"], "t0"] + float(epsilon_s)
    # Half-open intervals for positive-duration rows: keep [t0, t1)
    g["frame0"] = np.nan
    g["frame1"] = np.nan
    g["home_ball"] = np.nan
    g = g.dropna(subset=["t0", "t1"]).sort_values("t0").reset_index(drop=True)
    return g


def build_dead_markers(events: pd.DataFrame) -> pd.DataFrame:
    """Change points that force DEAD until the next OTB / kickoff."""
    types = {"OUT", "END"}
    mask = events["gameEvents.gameEventType"].isin(types)
    cols = ["gameEventId", "startTime", "endTime", "eventTime", "gameEvents.gameEventType", "gameEvents.period"]
    cols = [c for c in cols if c in events.columns]
    dead = events.loc[mask, cols].copy()
    dead = dead.rename(
        columns={
            "gameEvents.gameEventType": "event_type",
            "gameEvents.period": "period",
        }
    )
    dead["t_dead"] = pd.to_numeric(dead.get("eventTime", dead.get("startTime")), errors="coerce")
    if "startTime" in dead.columns:
        dead["t_dead"] = dead["t_dead"].fillna(pd.to_numeric(dead["startTime"], errors="coerce"))
    return dead.dropna(subset=["t_dead"]).sort_values("t_dead").reset_index(drop=True)


def _team_from_home_ball(
    home_ball: Any, home_team_id: float, focal_team_id: float
) -> str | None:
    """Map tracking-native home_ball → FOCAL/OPP relative to the analysis focal team."""
    if home_ball is None or (isinstance(home_ball, float) and np.isnan(home_ball)):
        return None
    if isinstance(home_ball, str):
        hb = home_ball.strip().lower() in ("1", "true", "t", "yes")
    else:
        hb = bool(home_ball)
    possessing_home = hb
    focal_is_home = float(focal_team_id) == float(home_team_id)
    if possessing_home == focal_is_home:
        return TEAM_FOCAL
    return TEAM_OPP


def resolve_otb_team_label(
    team_id: Any,
    home_ball: Any,
    focal_team_id: float,
    home_team_id: float | None,
) -> tuple[str, str]:
    """
    Resolve OTB team possession label.

    Prefer OTB ``team_id``; if missing, use tracking ``home_ball``. When both exist and
    disagree, keep ``team_id`` (player control) and note the conflict.
    """
    from_tid = None
    if team_id is not None and not (isinstance(team_id, float) and np.isnan(team_id)):
        from_tid = TEAM_FOCAL if float(team_id) == float(focal_team_id) else TEAM_OPP

    from_hb = None
    if home_team_id is not None:
        from_hb = _team_from_home_ball(home_ball, float(home_team_id), float(focal_team_id))

    if from_tid is None and from_hb is None:
        return TEAM_UNRESOLVED, "unresolved"
    if from_tid is None:
        return from_hb, "home_ball"
    if from_hb is None or from_hb == from_tid:
        return from_tid, "team_id"
    return from_tid, "team_id_over_home_ball_conflict"


def build_team_possession_spans(
    otb: pd.DataFrame,
    dead: pd.DataFrame,
    focal_team_id: int | float | str,
    t_min: float | None = None,
    t_max: float | None = None,
    home_team_id: int | float | str | None = None,
    respect_sequence: bool = True,
) -> pd.DataFrame:
    """
    Continuous team phase from OTB + dead markers.

    Same-team gaps stay FOCAL/OPP when still consistent with PFF ``sequence``
    (or when sequence is unknown). After OUT/END, DEAD until next OTB.

    ``home_ball`` (tracking-only) validates / fills team when ``team_id`` is missing;
    on conflict, ``team_id`` wins.
    """
    focal_team_id = float(focal_team_id)
    home_tid = float(home_team_id) if home_team_id is not None else None
    rows: list[dict] = []

    if otb.empty:
        return pd.DataFrame(columns=["t0", "t1", "team_possession", "reason"])

    otb = otb.sort_values("t0").reset_index(drop=True)
    dead_times = (
        pd.to_numeric(dead["t_dead"], errors="coerce").dropna().sort_values().to_numpy(dtype=float)
        if dead is not None and len(dead)
        else np.array([], dtype=float)
    )

    # Walk OTBs; insert DEAD after dead markers that fall between spans
    cur_team = None
    cur_t0 = None
    cur_seq = None
    prev_otb_t1 = None
    reason = None

    def _close(t1: float, why: str):
        nonlocal cur_team, cur_t0, cur_seq, reason
        if cur_team is None or cur_t0 is None:
            return
        if t1 > cur_t0:
            rows.append(
                {
                    "t0": float(cur_t0),
                    "t1": float(t1),
                    "team_possession": cur_team,
                    "reason": why if reason is None else reason,
                }
            )
        cur_team = None
        cur_t0 = None
        cur_seq = None
        reason = None

    def _seq_of(row) -> float | None:
        if "sequence" not in row.index or pd.isna(row["sequence"]):
            return None
        return float(row["sequence"])

    def _hb_of(row) -> Any:
        return row["home_ball"] if "home_ball" in row.index else None

    # Optional leading DEAD from t_min to first OTB if a dead marker exists before
    t0_first = float(otb.iloc[0]["t0"])
    if t_min is not None and t_min < t0_first:
        if len(dead_times) and dead_times[0] <= t0_first:
            rows.append(
                {
                    "t0": float(t_min),
                    "t1": t0_first,
                    "team_possession": TEAM_DEAD,
                    "reason": "pre_first_otb",
                }
            )

    for _, r in otb.iterrows():
        t0 = float(r["t0"])
        t1_otb = float(r["t1"]) if pd.notna(r["t1"]) else t0
        team, how = resolve_otb_team_label(r["team_id"], _hb_of(r), focal_team_id, home_tid)
        seq = _seq_of(r)

        if cur_team is not None and cur_t0 is not None:
            mids = dead_times[(dead_times > cur_t0) & (dead_times < t0)]
            if len(mids):
                td = float(mids[0])
                _close(td, "until_dead")
                rows.append(
                    {
                        "t0": td,
                        "t1": t0,
                        "team_possession": TEAM_DEAD,
                        "reason": "after_out_or_end",
                    }
                )
            elif cur_team != team:
                _close(t0, "until_opp_otb")
            elif respect_sequence and cur_seq is not None and seq is not None and cur_seq != seq:
                # Same team, different sequence: end at prior OTB end; leave the
                # inter-OTB gap unlabeled (maps to UNRESOLVED on frames).
                close_at = float(prev_otb_t1) if prev_otb_t1 is not None else t0
                _close(min(close_at, t0), "same_team_sequence_break")

        if cur_team is None:
            cur_team = team
            cur_t0 = t0
            cur_seq = seq
            reason = f"otb_start:{how}"
        elif cur_seq is None and seq is not None:
            cur_seq = seq

        prev_otb_t1 = t1_otb

    # Trailing: close at t_max or last OTB end
    t_end = float(otb["t1"].max()) if t_max is None else float(t_max)
    if cur_team is not None and cur_t0 is not None:
        mids = dead_times[(dead_times > cur_t0) & (dead_times < t_end)]
        if len(mids):
            td = float(mids[0])
            _close(td, "until_dead")
            rows.append(
                {
                    "t0": td,
                    "t1": t_end,
                    "team_possession": TEAM_DEAD,
                    "reason": "after_out_or_end",
                }
            )
        else:
            _close(t_end, "match_end")

    spans = pd.DataFrame(rows)
    if spans.empty:
        return spans
    # Merge adjacent identical labels
    spans = spans.sort_values("t0").reset_index(drop=True)
    merged = [spans.iloc[0].to_dict()]
    for i in range(1, len(spans)):
        prev = merged[-1]
        cur = spans.iloc[i].to_dict()
        if cur["team_possession"] == prev["team_possession"] and abs(cur["t0"] - prev["t1"]) < 1e-6:
            prev["t1"] = cur["t1"]
        else:
            merged.append(cur)
    return pd.DataFrame(merged)


def player_otb_label(
    player_id: float | int | None,
    team_id: float | int | None,
    focal_player_id: float | int | str,
    focal_team_id: float | int | str,
) -> str:
    if player_id is None or (isinstance(player_id, float) and np.isnan(player_id)):
        return PLAYER_NONE
    if float(player_id) == float(focal_player_id):
        return PLAYER_FOCAL
    if team_id is not None and not (isinstance(team_id, float) and np.isnan(team_id)):
        if float(team_id) == float(focal_team_id):
            return PLAYER_TEAMMATE
        return PLAYER_OPPONENT
    return PLAYER_NONE


def derive_possession_context(player_otb: str, team_possession: str) -> str:
    if team_possession == TEAM_DEAD:
        return CTX_DEAD
    if team_possession == TEAM_UNRESOLVED or pd.isna(team_possession):
        return CTX_UNRESOLVED
    if team_possession == TEAM_OPP:
        return CTX_OPPOSITION
    if team_possession == TEAM_FOCAL:
        if player_otb == PLAYER_FOCAL:
            return CTX_PLAYER_ON_BALL
        return CTX_TEAM_OFF_BALL
    return CTX_UNRESOLVED


def _label_intervals_on_times(
    times: np.ndarray,
    intervals: pd.DataFrame,
    t0_col: str,
    t1_col: str,
    value_col: str,
    default,
):
    """Half-open [t0, t1) labeling via searchsorted on sorted intervals."""
    out = np.array([default] * len(times), dtype=object)
    if intervals is None or intervals.empty:
        return out
    iv = intervals.sort_values(t0_col)
    t0 = iv[t0_col].to_numpy(dtype=float)
    t1 = iv[t1_col].to_numpy(dtype=float)
    vals = iv[value_col].to_numpy()
    idx = np.searchsorted(t0, times, side="right") - 1
    for i, j in enumerate(idx):
        if j >= 0 and times[i] < t1[j]:
            out[i] = vals[j]
    return out


def _label_intervals_on_frames(
    frames: np.ndarray,
    intervals: pd.DataFrame,
    f0_col: str,
    f1_col: str,
    value_col: str,
    default,
):
    """Inclusive [frame0, frame1] labeling (PFF start_frame..end_frame)."""
    out = np.array([default] * len(frames), dtype=object)
    if intervals is None or intervals.empty:
        return out
    iv = intervals.dropna(subset=[f0_col, f1_col]).sort_values(f0_col)
    if iv.empty:
        return out
    f0 = iv[f0_col].to_numpy(dtype=float)
    f1 = iv[f1_col].to_numpy(dtype=float)
    vals = iv[value_col].to_numpy()
    idx = np.searchsorted(f0, frames, side="right") - 1
    for i, j in enumerate(idx):
        if j >= 0 and frames[i] <= f1[j]:
            out[i] = vals[j]
    return out


def label_tracking_method_a(
    df: pd.DataFrame,
    otb: pd.DataFrame,
    team_spans: pd.DataFrame,
    focal_player_id: int | float | str,
    focal_team_id: int | float | str,
    time_col: str = "video_time_s",
    frame_col: str = "frameNum",
) -> pd.DataFrame:
    """
    Attach Method A columns to a focal-player tracking frame table.

    Prefers ``frame0``/``frame1`` from the tracking game-event index when present;
    otherwise falls back to ``video_time_s`` intervals ``[t0, t1)``.
    """
    out = df.copy()
    times = pd.to_numeric(out[time_col], errors="coerce").to_numpy(dtype=float)

    otb_lab = otb.copy()
    otb_lab["player_otb"] = [
        player_otb_label(pid, tid, focal_player_id, focal_team_id)
        for pid, tid in zip(otb_lab["player_id"], otb_lab["team_id"])
    ]

    use_frames = (
        frame_col in out.columns
        and "frame0" in otb_lab.columns
        and "frame1" in otb_lab.columns
        and otb_lab["frame0"].notna().any()
    )
    if use_frames:
        frames = pd.to_numeric(out[frame_col], errors="coerce").to_numpy(dtype=float)
        out["player_otb"] = _label_intervals_on_frames(
            frames, otb_lab, "frame0", "frame1", "player_otb", PLAYER_NONE
        )
        out["otb_is_instant"] = _label_intervals_on_frames(
            frames, otb_lab, "frame0", "frame1", "is_instant_pff", False
        )
    else:
        out["player_otb"] = _label_intervals_on_times(
            times, otb_lab, "t0", "t1", "player_otb", PLAYER_NONE
        )
        out["otb_is_instant"] = _label_intervals_on_times(
            times, otb_lab, "t0", "t1", "is_instant_pff", False
        )
    out["otb_is_instant"] = out["otb_is_instant"].astype(bool)

    out["team_possession"] = _label_intervals_on_times(
        times, team_spans, "t0", "t1", "team_possession", TEAM_UNRESOLVED
    )
    out["possession_context"] = [
        derive_possession_context(p, t) for p, t in zip(out["player_otb"], out["team_possession"])
    ]
    out["method"] = "A"
    return out


# ---------------------------------------------------------------------------
# Pass handoff validation
# ---------------------------------------------------------------------------


def pass_handoff_table(
    events: pd.DataFrame,
    otb: pd.DataFrame | None = None,
    max_gap_s: float = 5.0,
) -> pd.DataFrame:
    """Complete PA rows → next OTB of receiver within max_gap_s."""
    if otb is None:
        otb = build_otb_intervals(events)

    mask = (events["gameEvents.gameEventType"] == "OTB") & (
        events["possessionEvents.possessionEventType"] == "PA"
    )
    if "possessionEvents.passOutcomeType" in events.columns:
        # PFF codes: C=complete, D=incomplete, B=blocked, O=out, S=stolen/...
        mask = mask & (events["possessionEvents.passOutcomeType"].isin(["C", "COMPLETE", "complete"]))
    pa = events.loc[mask].copy()

    rows = []
    otb_sorted = otb.sort_values("t0")
    for _, r in pa.iterrows():
        t_raw = r.get("eventTime")
        if pd.isna(t_raw):
            t_raw = r.get("endTime")
        if pd.isna(t_raw):
            continue
        t_pa = float(t_raw)
        recv = pd.to_numeric(r.get("possessionEvents.receiverPlayerId"), errors="coerce")
        passer = pd.to_numeric(r.get("possessionEvents.passerPlayerId"), errors="coerce")
        if pd.isna(recv):
            continue
        nxt = otb_sorted.loc[(otb_sorted["t0"] >= t_pa - 1e-6) & (otb_sorted["t0"] <= t_pa + max_gap_s)]
        # Prefer receiver match
        hit = nxt.loc[nxt["player_id"] == float(recv)]
        if hit.empty:
            hit = nxt.head(1)
            match = False
            next_pid = float(hit.iloc[0]["player_id"]) if len(hit) else np.nan
            gap = float(hit.iloc[0]["t0"] - t_pa) if len(hit) else np.nan
        else:
            match = True
            next_pid = float(hit.iloc[0]["player_id"])
            gap = float(hit.iloc[0]["t0"] - t_pa)
        rows.append(
            {
                "t_pa": t_pa,
                "passer_id": float(passer) if pd.notna(passer) else np.nan,
                "receiver_id": float(recv),
                "next_otb_player_id": next_pid,
                "gap_s": gap,
                "receiver_match": match,
            }
        )
    cols = ["t_pa", "passer_id", "receiver_id", "next_otb_player_id", "gap_s", "receiver_match"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Summaries / segments
# ---------------------------------------------------------------------------


def value_share_table(series: pd.Series, name: str = "label") -> pd.DataFrame:
    s = series.fillna("NA")
    vc = s.value_counts(dropna=False)
    out = pd.DataFrame({name: vc.index, "n_frames": vc.values})
    out["pct"] = 100.0 * out["n_frames"] / max(len(s), 1)
    return out.reset_index(drop=True)


def contiguous_segments(
    times: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
) -> pd.DataFrame:
    times = np.asarray(pd.to_numeric(times, errors="coerce"), dtype=float)
    labels = np.asarray(labels, dtype=object)
    if len(times) == 0:
        return pd.DataFrame(columns=["label", "i0", "i1", "t0", "t1", "duration_s", "n_frames"])
    order = np.argsort(times)
    times = times[order]
    labels = labels[order]
    rows = []
    i0 = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[i0]:
            t0 = float(times[i0])
            t1 = float(times[i - 1])
            # duration ≈ last-first + median dt estimate
            dur = max(t1 - t0, 0.0)
            rows.append(
                {
                    "label": labels[i0],
                    "i0": int(i0),
                    "i1": int(i),
                    "t0": t0,
                    "t1": t1,
                    "duration_s": dur,
                    "n_frames": int(i - i0),
                }
            )
            i0 = i
    return pd.DataFrame(rows)


def segment_duration_stats(segments: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if segments is None or segments.empty:
        return pd.DataFrame(
            columns=["label", "n_segments", "median_s", "mean_s", "min_s", "max_s", "total_s"]
        )
    rows = []
    for lab, g in segments.groupby(label_col):
        d = g["duration_s"]
        rows.append(
            {
                "label": lab,
                "n_segments": int(len(g)),
                "median_s": float(d.median()),
                "mean_s": float(d.mean()),
                "min_s": float(d.min()),
                "max_s": float(d.max()),
                "total_s": float(d.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("total_s", ascending=False).reset_index(drop=True)


def agreement_table(a: pd.Series, b: pd.Series, name_a: str = "A", name_b: str = "B") -> pd.DataFrame:
    both = pd.DataFrame({name_a: a.astype(str).values, name_b: b.astype(str).values})
    ct = pd.crosstab(both[name_a], both[name_b], margins=True)
    agree = float((both[name_a] == both[name_b]).mean()) if len(both) else np.nan
    return ct, agree


# ---------------------------------------------------------------------------
# Method B — geometry + possession zone
# ---------------------------------------------------------------------------


def build_side_jersey_maps(
    roster: pd.DataFrame,
    meta: Mapping,
) -> dict[str, dict[str, dict]]:
    """
    Return {'home': {jersey: {player_id, team_id, nickname}}, 'away': {...}}.
    """
    home_id = str((meta.get("homeTeam") or {}).get("id"))
    maps: dict[str, dict[str, dict]] = {"home": {}, "away": {}}
    for _, r in roster.iterrows():
        side = "home" if str(r["team_id"]) == home_id else "away"
        jersey = str(int(r["shirt_number"]))
        maps[side][jersey] = {
            "player_id": str(r["player_id"]),
            "team_id": str(r["team_id"]),
            "nickname": r.get("nickname"),
        }
    return maps


def _ball_xy(rec: dict, prefer_raw: bool = True) -> tuple[float, float, str | None]:
    if prefer_raw:
        balls = rec.get("balls") or []
        if isinstance(balls, list) and balls:
            b = balls[0] or {}
            return (
                float(b["x"]) if b.get("x") is not None else np.nan,
                float(b["y"]) if b.get("y") is not None else np.nan,
                b.get("visibility"),
            )
    sm = rec.get("ballsSmoothed")
    if isinstance(sm, dict) and sm:
        return (
            float(sm["x"]) if sm.get("x") is not None else np.nan,
            float(sm["y"]) if sm.get("y") is not None else np.nan,
            sm.get("visibility"),
        )
    return np.nan, np.nan, None


def _nearest_on_side(
    players: list | None,
    bx: float,
    by: float,
) -> tuple[float, str | None]:
    if players is None or not np.isfinite(bx) or not np.isfinite(by):
        return np.nan, None
    best_d = np.inf
    best_j = None
    for p in players:
        x, y = p.get("x"), p.get("y")
        if x is None or y is None:
            continue
        d = float(np.hypot(float(x) - bx, float(y) - by))
        if d < best_d:
            best_d = d
            best_j = str(p.get("jerseyNum"))
    if best_j is None:
        return np.nan, None
    return best_d, best_j


def extract_frame_geometry(
    tracking_path: str | Path,
    cache_path: str | Path | None = None,
    prefer_raw_ball: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """
    One pass over tracking JSONL → per-frame min distance home/away to ball.
    """
    tracking_path = Path(tracking_path)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            return pd.read_parquet(cache_path)

    rows = []
    for rec in iter_jsonl(tracking_path):
        bx, by, bvis = _ball_xy(rec, prefer_raw=prefer_raw_ball)
        # Prefer smoothed player positions (Kalman) for geometry stability
        d_home, j_home = _nearest_on_side(rec.get("homePlayersSmoothed") or rec.get("homePlayers"), bx, by)
        d_away, j_away = _nearest_on_side(rec.get("awayPlayersSmoothed") or rec.get("awayPlayers"), bx, by)
        vtm = rec.get("videoTimeMs")
        rows.append(
            {
                "frameNum": rec.get("frameNum"),
                "period": rec.get("period"),
                "video_time_s": None if vtm is None else float(vtm) / 1000.0,
                "ball_x": bx,
                "ball_y": by,
                "ball_visibility": bvis,
                "d_home": d_home,
                "jersey_home": j_home,
                "d_away": d_away,
                "jersey_away": j_away,
            }
        )
    geo = pd.DataFrame(rows)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        geo.to_parquet(cache_path, index=False)
    return geo


def attach_team_distances(
    geo: pd.DataFrame,
    meta: Mapping,
    focal_team_id: int | float | str,
    jersey_maps: dict,
) -> pd.DataFrame:
    """Map home/away distances onto FOCAL / OPP and resolve player ids."""
    out = geo.copy()
    home_id = str((meta.get("homeTeam") or {}).get("id"))
    focal_is_home = str(focal_team_id) == home_id

    if focal_is_home:
        out["d_focal"] = out["d_home"]
        out["jersey_focal"] = out["jersey_home"]
        out["d_opp"] = out["d_away"]
        out["jersey_opp"] = out["jersey_away"]
        focal_side, opp_side = "home", "away"
    else:
        out["d_focal"] = out["d_away"]
        out["jersey_focal"] = out["jersey_away"]
        out["d_opp"] = out["d_home"]
        out["jersey_opp"] = out["jersey_home"]
        focal_side, opp_side = "away", "home"

    def _pid(side: str, jersey) -> float:
        if jersey is None or (isinstance(jersey, float) and np.isnan(jersey)):
            return np.nan
        info = jersey_maps.get(side, {}).get(str(jersey))
        if not info:
            return np.nan
        return float(info["player_id"])

    out["player_id_focal_nearest"] = [_pid(focal_side, j) for j in out["jersey_focal"]]
    out["player_id_opp_nearest"] = [_pid(opp_side, j) for j in out["jersey_opp"]]
    out["d_min"] = np.fmin(out["d_focal"].to_numpy(dtype=float), out["d_opp"].to_numpy(dtype=float))
    # which side wins
    foc = out["d_focal"].to_numpy(dtype=float)
    opp = out["d_opp"].to_numpy(dtype=float)
    nearest_side = np.where(foc <= opp, "FOCAL", "OPP")
    nearest_side = np.where(np.isfinite(foc) | np.isfinite(opp), nearest_side, None)
    out["nearest_side"] = nearest_side
    out["nearest_player_id"] = np.where(
        out["nearest_side"] == "FOCAL",
        out["player_id_focal_nearest"],
        out["player_id_opp_nearest"],
    )
    return out


def measure_actor_ball_distances_at_otb(
    tracking_path: str | Path,
    otb: pd.DataFrame,
    meta: Mapping,
    roster: pd.DataFrame,
    sync_s: float = 0.15,
    prefer_raw_ball: bool = True,
) -> pd.DataFrame:
    """
    For each OTB, find the actor's tracked distance to the ball near t0 (±sync_s).

    Uses the **minimum** distance in the window (not merely the nearest timestamp),
    preferring frames where the ball is VISIBLE.
    """
    jersey_maps = build_side_jersey_maps(roster, meta)
    pid_to_side_jersey: dict[float, tuple[str, str]] = {}
    for side, m in jersey_maps.items():
        for jersey, info in m.items():
            pid_to_side_jersey[float(info["player_id"])] = (side, jersey)

    targets = []
    for _, e in otb.iterrows():
        if pd.isna(e["player_id"]) or pd.isna(e["t0"]):
            continue
        pid = float(e["player_id"])
        if pid not in pid_to_side_jersey:
            continue
        side, jersey = pid_to_side_jersey[pid]
        targets.append(
            {
                "gameEventId": e["gameEventId"],
                "player_id": pid,
                "t0": float(e["t0"]),
                "side": side,
                "jersey": jersey,
                "is_instant_pff": bool(e.get("is_instant_pff", False)),
            }
        )
    if not targets:
        return pd.DataFrame()

    targets = sorted(targets, key=lambda r: r["t0"])
    # best[key] stores minimum visible (or any) distance in window
    best: dict[Any, dict] = {}
    n = len(targets)

    for rec in iter_jsonl(tracking_path):
        vtm = rec.get("videoTimeMs")
        if vtm is None:
            continue
        t = float(vtm) / 1000.0
        # binary search-ish: find targets with |t0 - t| <= sync_s
        # linear scan from left bound
        # maintain ti as first target with t0 >= t - sync_s
        # (recompute via search each frame is fine at 30 Hz * 180k)
        lo = 0
        hi = n
        # find first index with t0 >= t - sync_s
        while lo < hi:
            mid = (lo + hi) // 2
            if targets[mid]["t0"] < t - sync_s:
                lo = mid + 1
            else:
                hi = mid
        cand_idx = []
        j = lo
        while j < n and targets[j]["t0"] <= t + sync_s:
            cand_idx.append(j)
            j += 1
        if not cand_idx:
            continue
        bx, by, bvis = _ball_xy(rec, prefer_raw=prefer_raw_ball)
        if not (np.isfinite(bx) and np.isfinite(by)):
            continue
        visible = (bvis or "").upper() == "VISIBLE"
        for j in cand_idx:
            tgt = targets[j]
            pkey = "homePlayersSmoothed" if tgt["side"] == "home" else "awayPlayersSmoothed"
            alt = "homePlayers" if tgt["side"] == "home" else "awayPlayers"
            hit = None
            for p in rec.get(pkey) or rec.get(alt) or []:
                if str(p.get("jerseyNum")) == tgt["jersey"]:
                    hit = p
                    break
            if hit is None or hit.get("x") is None or hit.get("y") is None:
                continue
            d = float(np.hypot(float(hit["x"]) - bx, float(hit["y"]) - by))
            key = tgt["gameEventId"]
            prev = best.get(key)
            cand = {
                **tgt,
                "t_frame": t,
                "dist_m": d,
                "dt_s": t - tgt["t0"],
                "ball_visible": visible,
            }
            if prev is None:
                best[key] = cand
                continue
            # Prefer VISIBLE over non-visible; then smaller distance; then closer in time
            prev_vis = bool(prev.get("ball_visible"))
            better = False
            if visible and not prev_vis:
                better = True
            elif visible == prev_vis and d < float(prev["dist_m"]) - 1e-9:
                better = True
            elif visible == prev_vis and abs(d - float(prev["dist_m"])) <= 1e-9 and abs(t - tgt["t0"]) < abs(
                prev["t_frame"] - tgt["t0"]
            ):
                better = True
            if better:
                best[key] = cand

    return pd.DataFrame(list(best.values()))


def calibrate_from_actor_distances(
    actor_dists: pd.DataFrame,
    geo_with_ids: pd.DataFrame | None = None,
    otb: pd.DataFrame | None = None,
    sync_frames: int = 3,
    quantile: float = 0.95,
    outlier_cap_m: float = 8.0,
    rpz_floor_m: float = 1.0,
    rpz_ceil_m: float = 3.5,
) -> dict:
    """
    Build Rpz from actor–ball distances.

    Extreme distances (aerial / sync / tracking failures) are excluded from the
    quantile sample via outlier_cap_m; Rpz is then clipped to [floor, ceil] so a
    few bad OTB alignments cannot push the possession zone to tens of metres.
    """
    d_all = pd.to_numeric(actor_dists.get("dist_m"), errors="coerce").dropna().to_numpy(dtype=float)
    core = d_all[d_all <= float(outlier_cap_m)] if len(d_all) else d_all
    sample = core if len(core) >= 30 else d_all
    raw_q = float(np.quantile(sample, quantile)) if len(sample) else 1.5
    rpz = float(np.clip(raw_q, rpz_floor_m, rpz_ceil_m))
    p_nearest = np.nan
    if geo_with_ids is not None and otb is not None and len(actor_dists):
        g = geo_with_ids.dropna(subset=["video_time_s"]).sort_values("video_time_s")
        times = g["video_time_s"].to_numpy(dtype=float)
        ok = []
        for _, e in otb.iterrows():
            if pd.isna(e["player_id"]) or pd.isna(e["t0"]):
                continue
            j = int(np.searchsorted(times, float(e["t0"])))
            lo = max(0, j - sync_frames)
            hi = min(len(g), j + sync_frames + 1)
            window = g.iloc[lo:hi]
            if window.empty:
                continue
            k = int(np.argmin(np.abs(window["video_time_s"].to_numpy(dtype=float) - float(e["t0"]))))
            fr = window.iloc[k]
            ok.append(bool(fr.get("nearest_player_id") == float(e["player_id"])))
        if ok:
            p_nearest = float(np.mean(ok))
    return {
        "n_measured": int(len(d_all)),
        "n_core": int(len(core)),
        "distances": d_all,
        "distances_core": sample,
        "rpz_quantile": float(quantile),
        "rpz_raw_quantile_m": raw_q,
        "outlier_cap_m": float(outlier_cap_m),
        "rpz": rpz,
        "p_nearest_is_actor": p_nearest,
        "distance_stats": {
            "median": float(np.median(d_all)) if len(d_all) else np.nan,
            "p75": float(np.quantile(d_all, 0.75)) if len(d_all) else np.nan,
            "p90": float(np.quantile(d_all, 0.90)) if len(d_all) else np.nan,
            "p95": float(np.quantile(d_all, 0.95)) if len(d_all) else np.nan,
            "mean": float(np.mean(d_all)) if len(d_all) else np.nan,
            "core_p95": float(np.quantile(core, 0.95)) if len(core) else np.nan,
        },
    }


def label_geometry_method_b(
    geo: pd.DataFrame,
    r_pz: float,
    r_dz: float | None = None,
    dead_spans: pd.DataFrame | None = None,
    fallback_team_spans: pd.DataFrame | None = None,
    focal_player_id: int | float | str | None = None,
    focal_team_id: int | float | str | None = None,
) -> pd.DataFrame:
    """
    Frame-wise CONTROL / DUEL / FREE / DEAD and team / player layers for Method B.
    """
    if r_dz is None:
        r_dz = float(r_pz)
    out = geo.copy()
    d_f = out["d_focal"].to_numpy(dtype=float)
    d_o = out["d_opp"].to_numpy(dtype=float)
    ball_ok = np.isfinite(out["ball_x"].to_numpy(dtype=float)) & np.isfinite(
        out["ball_y"].to_numpy(dtype=float)
    )

    focal_in = ball_ok & np.isfinite(d_f) & (d_f < r_pz)
    opp_in = ball_ok & np.isfinite(d_o) & (d_o < r_pz)
    # Duel: both sides have a player within R_dz
    duel = ball_ok & np.isfinite(d_f) & np.isfinite(d_o) & (d_f < r_dz) & (d_o < r_dz)

    state = np.full(len(out), FREE, dtype=object)
    state[~ball_ok] = FREE  # missing ball → free/no control (event dead handled below)
    state[duel] = DUEL
    control_mask = (focal_in | opp_in) & ~duel
    state[control_mask] = CONTROL
    out["ball_state"] = state

    # Controlling player / team
    ctrl_team = np.array([None] * len(out), dtype=object)
    ctrl_pid = np.full(len(out), np.nan)
    for i in range(len(out)):
        if state[i] == CONTROL:
            if focal_in[i] and (not opp_in[i] or d_f[i] <= d_o[i]):
                ctrl_team[i] = TEAM_FOCAL
                ctrl_pid[i] = out["player_id_focal_nearest"].iloc[i]
            elif opp_in[i]:
                ctrl_team[i] = TEAM_OPP
                ctrl_pid[i] = out["player_id_opp_nearest"].iloc[i]
        elif state[i] == DUEL:
            ctrl_team[i] = None
            ctrl_pid[i] = np.nan
    out["control_team"] = ctrl_team
    out["control_player_id"] = ctrl_pid

    # Dead spans from events override
    times = pd.to_numeric(out["video_time_s"], errors="coerce").to_numpy(dtype=float)
    event_dead = np.zeros(len(out), dtype=bool)
    if dead_spans is not None and len(dead_spans):
        dead_lab = _label_intervals_on_times(
            times, dead_spans, "t0", "t1", "team_possession", None
        )
        event_dead = np.array([x == TEAM_DEAD for x in dead_lab], dtype=bool)
        out.loc[event_dead, "ball_state"] = DEAD

    fallback = None
    if fallback_team_spans is not None and len(fallback_team_spans):
        fallback = _label_intervals_on_times(
            times, fallback_team_spans, "t0", "t1", "team_possession", TEAM_UNRESOLVED
        )

    # Team possession: persist last CONTROL team through FREE/DUEL; event DEAD resets
    team: list[str] = []
    last = TEAM_UNRESOLVED
    for i in range(len(out)):
        if event_dead[i]:
            last = TEAM_DEAD
            team.append(TEAM_DEAD)
            continue
        st = out["ball_state"].iloc[i]
        if st == CONTROL and ctrl_team[i] is not None:
            last = ctrl_team[i]
            team.append(last)
        else:
            if last in (TEAM_FOCAL, TEAM_OPP):
                team.append(last)
            elif fallback is not None and fallback[i] in (TEAM_FOCAL, TEAM_OPP, TEAM_DEAD):
                last = fallback[i]
                team.append(last)
            else:
                team.append(TEAM_UNRESOLVED)
    out["team_possession"] = team

    # Player OTB-like layer relative to focal player
    focal_player_id_f = float(focal_player_id) if focal_player_id is not None else None
    potb = []
    for i in range(len(out)):
        st = out["ball_state"].iloc[i]
        if st == DUEL:
            potb.append(PLAYER_DUEL)
        elif st == CONTROL and np.isfinite(ctrl_pid[i]):
            if ctrl_team[i] == TEAM_FOCAL:
                potb.append(
                    PLAYER_FOCAL
                    if focal_player_id_f is not None and float(ctrl_pid[i]) == focal_player_id_f
                    else PLAYER_TEAMMATE
                )
            else:
                potb.append(PLAYER_OPPONENT)
        else:
            potb.append(PLAYER_NONE)
    out["player_otb"] = potb
    out["possession_context"] = [
        derive_possession_context(p if p != PLAYER_DUEL else PLAYER_NONE, t)
        for p, t in zip(out["player_otb"], out["team_possession"])
    ]
    # During duel keep team context but player not on ball
    out["method"] = "B"
    return out


def recover_instant_control_windows(
    otb: pd.DataFrame,
    geo_b: pd.DataFrame,
    r_pz: float,
    search_s: float = 0.5,
) -> pd.DataFrame:
    """
    For zero-duration PFF OTBs, find contiguous frames where the actor is inside Rpz.
    """
    inst = otb.loc[otb["is_instant_pff"]].copy()
    g = geo_b.dropna(subset=["video_time_s"]).sort_values("video_time_s")
    times = g["video_time_s"].to_numpy(dtype=float)
    rows = []
    for _, e in inst.iterrows():
        pid = float(e["player_id"]) if pd.notna(e["player_id"]) else np.nan
        t = float(e["t0"])
        if not np.isfinite(pid):
            continue
        m = (times >= t - search_s) & (times <= t + search_s)
        win = g.loc[m]
        if win.empty:
            continue
        # actor distance series
        d = []
        for _, fr in win.iterrows():
            if fr.get("player_id_focal_nearest") == pid:
                d.append(float(fr["d_focal"]))
            elif fr.get("player_id_opp_nearest") == pid:
                d.append(float(fr["d_opp"]))
            else:
                d.append(np.nan)
        d = np.asarray(d, dtype=float)
        inside = np.isfinite(d) & (d < r_pz)
        if not inside.any():
            rows.append(
                {
                    "gameEventId": e["gameEventId"],
                    "player_id": pid,
                    "t_event": t,
                    "recovered": False,
                    "t0": np.nan,
                    "t1": np.nan,
                    "n_frames": 0,
                    "min_dist": float(np.nanmin(d)) if np.isfinite(d).any() else np.nan,
                }
            )
            continue
        # longest contiguous True run containing closest-to-event frame or first run
        idx = np.where(inside)[0]
        # expand around frame nearest t among inside
        t_win = win["video_time_s"].to_numpy(dtype=float)
        nearest_inside = idx[np.argmin(np.abs(t_win[idx] - t))]
        lo = hi = int(nearest_inside)
        while lo - 1 >= 0 and inside[lo - 1]:
            lo -= 1
        while hi + 1 < len(inside) and inside[hi + 1]:
            hi += 1
        rows.append(
            {
                "gameEventId": e["gameEventId"],
                "player_id": pid,
                "t_event": t,
                "recovered": True,
                "t0": float(t_win[lo]),
                "t1": float(t_win[hi]),
                "n_frames": int(hi - lo + 1),
                "min_dist": float(np.nanmin(d[lo : hi + 1])),
            }
        )
    return pd.DataFrame(rows)


def label_tracking_method_b(
    focal_df: pd.DataFrame,
    geo_labeled: pd.DataFrame,
    on: str = "frameNum",
) -> pd.DataFrame:
    """Join Method B frame labels onto the focal player's tracking table."""
    cols = [
        on,
        "ball_state",
        "player_otb",
        "team_possession",
        "possession_context",
        "d_focal",
        "d_opp",
        "d_min",
        "nearest_player_id",
        "control_player_id",
        "ball_x",
        "ball_y",
        "method",
    ]
    cols = [c for c in cols if c in geo_labeled.columns or c == on]
    right = geo_labeled[cols].drop_duplicates(on)
    out = focal_df.merge(right, on=on, how="left", suffixes=("", "_b"))
    if "method" in out.columns:
        out["method"] = "B"
    return out


# ---------------------------------------------------------------------------
# Focal player tracking extract (lightweight)
# ---------------------------------------------------------------------------


def extract_outfield_tracking(
    tracking_path: str | Path,
    roster: pd.DataFrame,
    meta: Mapping,
    cache_path: str | Path | None = None,
    force: bool = False,
    prefer_smoothed_player: bool = True,
) -> pd.DataFrame:
    """
    One JSONL pass → long table of all non-GK players (smoothed x,y).

    Columns: frameNum, period, video_time_s, periodGameClockTime, player_id,
    team_id, side, shirt_number, x, y, visibility
    """
    tracking_path = Path(tracking_path)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            return pd.read_parquet(cache_path)

    home_id = str((meta.get("homeTeam") or {}).get("id"))
    # jersey -> (player_id, team_id) per side
    side_maps: dict[str, dict[str, tuple[str, str]]] = {"home": {}, "away": {}}
    for _, r in roster.iterrows():
        if str(r.get("position", "")).upper() == "GK":
            continue
        side = "home" if str(r["team_id"]) == home_id else "away"
        jersey = str(int(r["shirt_number"]))
        side_maps[side][jersey] = (str(r["player_id"]), str(r["team_id"]))

    rows: list[dict] = []
    for rec in iter_jsonl(tracking_path):
        vtm = rec.get("videoTimeMs")
        video_time_s = None if vtm is None else float(vtm) / 1000.0
        base = {
            "frameNum": rec.get("frameNum"),
            "period": rec.get("period"),
            "video_time_s": video_time_s,
            "periodGameClockTime": rec.get("periodGameClockTime"),
        }
        for side, jmap in side_maps.items():
            if prefer_smoothed_player:
                pkey = "homePlayersSmoothed" if side == "home" else "awayPlayersSmoothed"
                alt = "homePlayers" if side == "home" else "awayPlayers"
            else:
                pkey = "homePlayers" if side == "home" else "awayPlayers"
                alt = pkey
            for p in rec.get(pkey) or rec.get(alt) or []:
                jersey = str(p.get("jerseyNum"))
                if jersey not in jmap:
                    continue
                if p.get("x") is None or p.get("y") is None:
                    continue
                pid, tid = jmap[jersey]
                rows.append(
                    {
                        **base,
                        "player_id": pid,
                        "team_id": tid,
                        "side": side,
                        "shirt_number": int(jersey),
                        "x": float(p["x"]),
                        "y": float(p["y"]),
                        "visibility": p.get("visibility"),
                    }
                )

    out = pd.DataFrame(rows)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def extract_focal_tracking(
    tracking_path: str | Path,
    side: str,
    shirt_number: int | str,
    prefer_smoothed_player: bool = True,
) -> pd.DataFrame:
    """Extract one player's frames + raw ball (for speed / plots)."""
    pkey = "homePlayers" if side == "home" else "awayPlayers"
    skey = "homePlayersSmoothed" if side == "home" else "awayPlayersSmoothed"
    jersey = str(shirt_number)
    rows = []
    for rec in iter_jsonl(tracking_path):
        hit = None
        src = rec.get(skey if prefer_smoothed_player else pkey) or rec.get(pkey) or []
        for p in src:
            if str(p.get("jerseyNum")) == jersey:
                hit = p
                break
        if hit is None:
            continue
        bx, by, bvis = _ball_xy(rec, prefer_raw=True)
        vtm = rec.get("videoTimeMs")
        rows.append(
            {
                "frameNum": rec.get("frameNum"),
                "period": rec.get("period"),
                "video_time_s": None if vtm is None else float(vtm) / 1000.0,
                "periodGameClockTime": rec.get("periodGameClockTime"),
                "periodElapsedTime": rec.get("periodElapsedTime"),
                "x": hit.get("x"),
                "y": hit.get("y"),
                "visibility": hit.get("visibility"),
                "ball_x": bx,
                "ball_y": by,
                "ball_visibility": bvis,
            }
        )
    return pd.DataFrame(rows)


def add_simple_speed(df: pd.DataFrame, time_col: str = "video_time_s") -> pd.DataFrame:
    """Forward difference speed from consecutive frames (validation plots)."""
    out = df.sort_values(["period", time_col, "frameNum"]).reset_index(drop=True).copy()
    out["speed"] = np.nan
    for _, sub in out.groupby("period", sort=False):
        t = pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(sub["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(dtype=float)
        if len(t) < 2:
            continue
        dt = np.diff(t)
        dx = np.diff(x)
        dy = np.diff(y)
        sp = np.full(len(t), np.nan)
        good = dt > 0
        sp[1:] = np.where(good, np.hypot(dx, dy) / np.where(good, dt, np.nan), np.nan)
        if len(sp) > 1:
            sp[0] = sp[1]
        out.loc[sub.index, "speed"] = sp
    out.loc[out["speed"] > 15.0, "speed"] = np.nan
    return out


def build_match_possession(
    match_id: str,
    match_dir: str | Path,
    focal_player_id: int | float | str,
    focal_team_id: int | float | str,
    focal_shirt: int,
    focal_side: str,
    cache_dir: str | Path | None = None,
    force_geometry: bool = False,
    rpz_quantile: float = 0.95,
) -> dict[str, Any]:
    """
    End-to-end Method A + B for one match. Caches geometry and focal tracking under cache_dir.
    """
    match_dir = Path(match_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else match_dir / "possession_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(match_dir / f"{match_id}_events.json")
    roster = load_roster(match_dir / f"{match_id}_roster.json", match_id)
    meta = load_json(match_dir / f"{match_id}_metadata.json")
    tracking_path = match_dir / f"{match_id}_tracking.jsonl"

    focal_path = cache_dir / f"{match_id}_focal_{focal_player_id}.parquet"
    geo_path = cache_dir / f"{match_id}_frame_geometry.parquet"
    actor_path = cache_dir / f"{match_id}_otb_actor_dist.parquet"

    if focal_path.exists() and not force_geometry:
        focal = pd.read_parquet(focal_path)
    else:
        focal = extract_focal_tracking(tracking_path, focal_side, focal_shirt)
        focal = add_simple_speed(focal)
        focal.to_parquet(focal_path, index=False)

    eps = median_frame_dt(focal["video_time_s"])
    otb = build_otb_intervals(events, epsilon_s=eps)
    ge_index_path = cache_dir / f"{match_id}_game_event_index.parquet"
    ge_index = extract_tracking_game_event_index(
        tracking_path, cache_path=ge_index_path, force=force_geometry
    )
    otb = enrich_otb_with_tracking_index(otb, ge_index)
    dead = build_dead_markers(events)
    t_min = float(focal["video_time_s"].min())
    t_max = float(focal["video_time_s"].max())
    home_team_id = (meta.get("homeTeam") or {}).get("id")
    team_spans = build_team_possession_spans(
        otb,
        dead,
        focal_team_id,
        t_min=t_min,
        t_max=t_max,
        home_team_id=home_team_id,
        respect_sequence=True,
    )

    labeled_a = label_tracking_method_a(
        focal, otb, team_spans, focal_player_id, focal_team_id
    )

    geo = extract_frame_geometry(
        tracking_path, cache_path=geo_path, prefer_raw_ball=True, force=force_geometry
    )
    jmaps = build_side_jersey_maps(roster, meta)
    geo = attach_team_distances(geo, meta, focal_team_id, jmaps)

    if actor_path.exists() and not force_geometry:
        actor_dists = pd.read_parquet(actor_path)
    else:
        actor_dists = measure_actor_ball_distances_at_otb(
            tracking_path, otb, meta, roster, sync_s=0.15, prefer_raw_ball=True
        )
        actor_dists.to_parquet(actor_path, index=False)

    calib = calibrate_from_actor_distances(
        actor_dists, geo_with_ids=geo, otb=otb, quantile=rpz_quantile
    )
    r_pz = float(calib["rpz"])
    r_dz = float(r_pz)  # start equal; report as such

    # Dead spans only (for Method B event-dead overlay)
    dead_spans = team_spans.loc[team_spans["team_possession"] == TEAM_DEAD].copy()
    geo_b = label_geometry_method_b(
        geo,
        r_pz=r_pz,
        r_dz=r_dz,
        dead_spans=dead_spans,
        fallback_team_spans=team_spans,
        focal_player_id=focal_player_id,
        focal_team_id=focal_team_id,
    )
    labeled_b = label_tracking_method_b(focal, geo_b)

    recovered = recover_instant_control_windows(otb, geo_b, r_pz=r_pz, search_s=0.5)
    handoff = pass_handoff_table(events, otb)

    return {
        "match_id": str(match_id),
        "events": events,
        "roster": roster,
        "meta": meta,
        "otb": otb,
        "dead": dead,
        "team_spans": team_spans,
        "focal": focal,
        "labeled_a": labeled_a,
        "labeled_b": labeled_b,
        "geo_b": geo_b,
        "calib": calib,
        "r_pz": r_pz,
        "r_dz": r_dz,
        "actor_dists": actor_dists,
        "recovered_instants": recovered,
        "handoff": handoff,
        "epsilon_s": eps,
        "cache_dir": cache_dir,
        "ge_index": ge_index,
    }
