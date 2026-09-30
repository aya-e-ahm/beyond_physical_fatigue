"""
World Cup 2022 position profiles — stream one match at a time.

Regulation minutes only (1–90). Mean and sample SD of player-minute
speed, |acceleration|, distance, and |jerk|, at 3-class and detailed
position-code level. Goalkeepers are ignored.
"""
from __future__ import annotations

import json
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import match_player_analysis as mpa

N_MINUTES = 90
EXPECTED_MATCHES = 64
COVERAGE_FRAC = 0.70
METRICS = ("speed", "acceleration", "distance", "jerk")
COARSE_ORDER = ("Defender", "Midfielder", "Attacker")
COARSE_MAP = {
    "LB": "Defender",
    "LCB": "Defender",
    "RCB": "Defender",
    "RB": "Defender",
    "DM": "Midfielder",
    "CM": "Midfielder",
    "AM": "Midfielder",
    "LW": "Attacker",
    "RW": "Attacker",
    "CF": "Attacker",
}
DETAIL_ORDER = ("LB", "LCB", "RCB", "RB", "DM", "CM", "AM", "LW", "RW", "CF")


@dataclass
class MatchFiles:
    match_id: str
    tracking: Path
    roster: Path
    metadata: Path


@dataclass
class ProfileResult:
    table: pd.DataFrame
    unknown_codes: list[str] = field(default_factory=list)
    n_matches: int = 0
    errors: list[str] = field(default_factory=list)


class _Buf:
    __slots__ = ("t", "clock", "x", "y", "period", "frame")

    def __init__(self):
        self.t = array("d")
        self.clock = array("d")
        self.x = array("d")
        self.y = array("d")
        self.period = array("h")
        self.frame = array("i")

    def add(self, t, clock, x, y, period, frame):
        self.t.append(float(t))
        self.clock.append(float(clock))
        self.x.append(float(x))
        self.y.append(float(y))
        self.period.append(int(period))
        self.frame.append(int(frame) if frame is not None else 0)

    def __len__(self):
        return len(self.t)


class _Running:
    """Online n / sum / sumsq for (group, minute, metric)."""

    def __init__(self):
        self._n: dict[str, np.ndarray] = {}
        self._sum: dict[str, np.ndarray] = {}
        self._sumsq: dict[str, np.ndarray] = {}

    def add(self, group: str, minute_idx: int, values: np.ndarray):
        if group not in self._n:
            self._n[group] = np.zeros((N_MINUTES, len(METRICS)), dtype=np.int64)
            self._sum[group] = np.zeros((N_MINUTES, len(METRICS)), dtype=np.float64)
            self._sumsq[group] = np.zeros((N_MINUTES, len(METRICS)), dtype=np.float64)
        ok = np.isfinite(values)
        if not ok.any():
            return
        v = np.where(ok, values, 0.0)
        self._n[group][minute_idx, ok] += 1
        self._sum[group][minute_idx, ok] += v[ok]
        self._sumsq[group][minute_idx, ok] += v[ok] * v[ok]

    def to_frame(self, level: str) -> pd.DataFrame:
        rows = []
        for group, n in self._n.items():
            s = self._sum[group]
            ss = self._sumsq[group]
            mean = np.full(n.shape, np.nan)
            sd = np.full(n.shape, np.nan)
            good = n > 0
            mean[good] = s[good] / n[good]
            enough = n >= 2
            var = np.zeros(n.shape, dtype=np.float64)
            var[enough] = (ss[enough] - (s[enough] ** 2) / n[enough]) / (n[enough] - 1)
            sd[enough] = np.sqrt(np.maximum(var[enough], 0.0))
            for mi in range(N_MINUTES):
                for k, metric in enumerate(METRICS):
                    rows.append(
                        {
                            "level": level,
                            "group": group,
                            "minute": mi + 1,
                            "metric": metric,
                            "mean": float(mean[mi, k]) if np.isfinite(mean[mi, k]) else np.nan,
                            "sd": float(sd[mi, k]) if np.isfinite(sd[mi, k]) else np.nan,
                            "n": int(n[mi, k]),
                        }
                    )
        return pd.DataFrame(rows)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if (
        isinstance(data, list)
        and len(data) == 1
        and isinstance(data[0], dict)
        and "homeTeam" in data[0]
    ):
        return data[0]
    return data


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _first_existing_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    return None


def _match_id_from_tracking_name(path: Path) -> str | None:
    name = path.name

    suffixes = [
        "_tracking.jsonl.bz2",
        "_tracking.jsonl",
        "_tracking.json",
        ".jsonl.bz2",
        ".jsonl",
        ".json",
    ]

    for suffix in suffixes:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]

    return None


def _pair_sidecars(match_id: str, tracking: Path, roster_dir: Path | None, meta_dir: Path | None) -> MatchFiles | None:
    roster_candidates = []
    meta_candidates = []
    if roster_dir is not None:
        roster_candidates.extend(
            [
                roster_dir / f"{match_id}.json",
                roster_dir / f"{match_id}_roster.json",
            ]
        )
    roster_candidates.extend(
        [
            tracking.with_name(f"{match_id}_roster.json"),
            tracking.with_name(f"{match_id}.json"),
        ]
    )
    if meta_dir is not None:
        meta_candidates.extend(
            [
                meta_dir / f"{match_id}.json",
                meta_dir / f"{match_id}_metadata.json",
            ]
        )
    meta_candidates.extend(
        [
            tracking.with_name(f"{match_id}_metadata.json"),
            tracking.with_name(f"{match_id}.json"),
        ]
    )
    roster = next((p for p in roster_candidates if p.is_file() and p != tracking), None)
    metadata = next((p for p in meta_candidates if p.is_file() and p != tracking), None)
    if roster is None or metadata is None:
        return None
    return MatchFiles(match_id, tracking, roster, metadata)

def debug_discover_matches(root):
    root = Path(root)

    tracking_dir = _first_existing_dir(
        root, ("Tracking Data", "Tracking", "tracking")
    )
    roster_dir = _first_existing_dir(
        root, ("Rosters", "Roster", "rosters")
    )
    meta_dir = _first_existing_dir(
        root, ("Metadata", "metadata")
    )

    print("tracking_dir:", tracking_dir)
    print("roster_dir:", roster_dir)
    print("meta_dir:", meta_dir)

    tracking_files = []

    if tracking_dir is not None:
        tracking_files.extend(sorted(tracking_dir.glob("*.jsonl")))
        tracking_files.extend(sorted(tracking_dir.glob("*.jsonl.bz2")))
        tracking_files.extend(sorted(tracking_dir.glob("*.json")))

    print("\nTracking files found:", len(tracking_files))

    for path in tracking_files[:10]:
        print("\nFILE:", path.name)

        mid = _match_id_from_tracking_name(path)
        print("  match_id:", mid)

        if mid is None:
            print("  FAIL: could not extract match ID")
            continue

        paired = _pair_sidecars(
            mid,
            path,
            roster_dir,
            meta_dir
        )

        print("  paired:", paired)

        if paired is None:
            print("  FAIL: roster/metadata sidecar not found")

def discover_matches(root: Path | str) -> list[MatchFiles]:
    """Find match triples under the Drive layout or the local ``{id}_tracking.jsonl`` layout."""
    root = Path(root)
    tracking_dir = _first_existing_dir(root, ("Tracking Data", "Tracking", "tracking"))
    roster_dir = _first_existing_dir(root, ("Rosters", "Roster", "rosters"))
    meta_dir = _first_existing_dir(root, ("Metadata", "metadata"))

    tracking_files: list[Path] = []

    if tracking_dir is not None:
        tracking_files.extend(sorted(tracking_dir.glob("*.jsonl")))
        tracking_files.extend(sorted(tracking_dir.glob("*.jsonl.bz2")))
        tracking_files.extend(sorted(tracking_dir.glob("*.json")))
    else:
        tracking_files.extend(sorted(root.rglob("*_tracking.jsonl")))
        tracking_files.extend(sorted(root.rglob("*_tracking.jsonl.bz2")))

    seen: set[str] = set()
    out: list[MatchFiles] = []
    for path in tracking_files:
        if path.name.lower().endswith(".json") and path.stat().st_size < 4096:
            continue
        mid = _match_id_from_tracking_name(path)
        if mid is None or mid in seen:
            continue
        paired = _pair_sidecars(mid, path, roster_dir, meta_dir)
        if paired is None:
            continue
        seen.add(mid)
        out.append(paired)
    out.sort(key=lambda m: m.match_id)
    return out


def roster_inventory(matches: list[MatchFiles]) -> pd.DataFrame:
    """Count roster appearances per detailed code and 3-class group (GK excluded from 3-class)."""
    rows = []
    for m in matches:
        raw = _load_json(m.roster)
        if isinstance(raw, dict):
            raw = raw.get("players") or raw.get("roster") or []
        for rec in raw:
            code = str(rec.get("positionGroupType") or "").upper()
            if not code:
                continue
            rows.append({"match_id": m.match_id, "code": code, "coarse": COARSE_MAP.get(code)})
    if not rows:
        return pd.DataFrame(columns=["code", "coarse", "n_roster_rows", "n_matches"])
    df = pd.DataFrame(rows)
    g = (
        df.groupby(["code", "coarse"], dropna=False)
        .agg(n_roster_rows=("match_id", "size"), n_matches=("match_id", "nunique"))
        .reset_index()
    )
    return g.sort_values(["coarse", "code"], na_position="last")


def _video_seconds(rec: dict) -> float | None:
    if rec.get("videoTimeMs") is not None:
        return float(rec["videoTimeMs"]) / 1000.0
    if rec.get("video_time_s") is not None:
        return float(rec["video_time_s"])
    if rec.get("videoTime") is not None:
        v = float(rec["videoTime"])
        return v / 1000.0 if v > 10000 else v
    return None


def _regulation_minute(period: int, clock_s: float) -> int | None:
    if period == 1 and 0.0 <= clock_s < 45.0 * 60.0:
        return int(clock_s // 60.0)
    if period == 2 and 45.0 * 60.0 <= clock_s < 90.0 * 60.0:
        return int(clock_s // 60.0)
    return None


def _load_roster_index(match: MatchFiles) -> dict[tuple[str, str], str]:
    meta = _load_json(match.metadata)
    home_id = str((meta.get("homeTeam") or {}).get("id"))
    raw = _load_json(match.roster)
    if isinstance(raw, dict):
        raw = raw.get("players") or raw.get("roster") or []
    index: dict[tuple[str, str], str] = {}
    for rec in raw:
        code = str(rec.get("positionGroupType") or "").upper()
        if not code or code == "GK":
            continue
        team_id = str((rec.get("team") or {}).get("id"))
        side = "home" if team_id == home_id else "away"
        jersey = str(rec.get("shirtNumber"))
        index[(side, jersey)] = code
    return index


def _player_minute_values(df: pd.DataFrame) -> np.ndarray:
    """Return (90, 4) array; NaN where the minute fails the coverage gate."""
    out = np.full((N_MINUTES, len(METRICS)), np.nan)
    if df.empty:
        return out
    t = df["video_time_s"].to_numpy(dtype=float)
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[dt > 0]
    med_dt = float(np.median(dt)) if len(dt) else (1.0 / 30.0)
    expected = max(1.0, 60.0 / med_dt)
    min_frames = max(3, int(COVERAGE_FRAC * expected))

    clock = df["periodGameClockTime"].to_numpy(dtype=float)
    period = df["period"].to_numpy(dtype=int)
    minute = np.array(
        [_regulation_minute(int(p), float(c)) if np.isfinite(c) else None for p, c in zip(period, clock)],
        dtype=object,
    )
    speed = df["speed"].to_numpy(dtype=float)
    acc = df["acc_use"].to_numpy(dtype=float)
    jerk = df["jerk_use"].to_numpy(dtype=float)
    step = df["step_dist"].to_numpy(dtype=float)

    for mi in range(N_MINUTES):
        m = np.array([v == mi for v in minute], dtype=bool)
        n_obs = int(m.sum())
        if n_obs < min_frames:
            continue
        sp = speed[m]
        ac = acc[m]
        jk = jerk[m]
        dist = step[m]
        sp_ok = sp[np.isfinite(sp)]
        ac_ok = ac[np.isfinite(ac)]
        jk_ok = jk[np.isfinite(jk)]
        dist_ok = dist[np.isfinite(dist)]
        out[mi, 0] = float(np.mean(sp_ok)) if len(sp_ok) >= min_frames else np.nan
        out[mi, 1] = float(np.mean(ac_ok)) if len(ac_ok) >= min_frames else np.nan
        out[mi, 2] = float(np.sum(dist_ok)) if len(dist_ok) else np.nan
        out[mi, 3] = float(np.mean(jk_ok)) if len(jk_ok) >= min_frames else np.nan
    return out


def _fold_player(buf: _Buf, code: str, coarse: _Running, detailed: _Running, unknown: set[str]):
    if len(buf) < 10:
        return
    df = pd.DataFrame(
        {
            "video_time_s": np.asarray(buf.t, dtype=float),
            "periodGameClockTime": np.asarray(buf.clock, dtype=float),
            "x": np.asarray(buf.x, dtype=float),
            "y": np.asarray(buf.y, dtype=float),
            "period": np.asarray(buf.period, dtype=int),
            "frameNum": np.asarray(buf.frame, dtype=int),
        }
    )
    df = mpa.add_stride_kinematics(df)
    df = mpa.add_step_distance(df)
    vals = _player_minute_values(df)
    coarse_name = COARSE_MAP.get(code)
    if coarse_name is None:
        unknown.add(code)
    for mi in range(N_MINUTES):
        row = vals[mi]
        if not np.isfinite(row).any():
            continue
        detailed.add(code, mi, row)
        if coarse_name is not None:
            coarse.add(coarse_name, mi, row)


def _stream_match(match: MatchFiles, index: dict[tuple[str, str], str]) -> dict[tuple[str, str], _Buf]:
    bufs = {key: _Buf() for key in index}
    for rec in _iter_jsonl(match.tracking):
        try:
            period = int(rec.get("period"))
        except (TypeError, ValueError):
            continue
        if period not in (1, 2):
            continue
        clock = rec.get("periodGameClockTime")
        t = _video_seconds(rec)
        if clock is None or t is None:
            continue
        frame = rec.get("frameNum")
        for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
            for p in rec.get(key) or []:
                ident = (side, str(p.get("jerseyNum")))
                buf = bufs.get(ident)
                if buf is None:
                    continue
                x, y = p.get("x"), p.get("y")
                if x is None or y is None:
                    continue
                buf.add(t, clock, x, y, period, frame)
    return bufs


def _progress(total: int, desc: str, position: int = 0):
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, position=position, leave=(position == 0), unit="match" if position == 0 else "player")
    except ImportError:
        return None


def accumulate_profiles(matches: list[MatchFiles], expected_matches: int = EXPECTED_MATCHES) -> ProfileResult:
    """Stream every match. Progress: matches finished / 64, and players in the current match."""
    n_found = len(matches)
    if n_found != expected_matches:
        print(
            f"Note: found {n_found} match files; progress is finished/{expected_matches} "
            f"(tournament size)."
        )
    coarse = _Running()
    detailed = _Running()
    unknown: set[str] = set()
    errors: list[str] = []

    outer_total = expected_matches
    match_bar = _progress(outer_total, "matches 0/%d" % expected_matches, position=0)
    done = 0
    for match in matches:
        label = f"matches {done}/{expected_matches}"
        if match_bar is not None:
            match_bar.set_description(label + f"  |  {match.match_id}")
        else:
            print(f"{label}  |  starting {match.match_id}")
        try:
            index = _load_roster_index(match)
            bufs = _stream_match(match, index)
            keys = list(index.keys())
            player_bar = _progress(len(keys), f"players 0/{len(keys)}  {match.match_id}", position=1)
            for i, key in enumerate(keys, start=1):
                _fold_player(bufs.pop(key), index[key], coarse, detailed, unknown)
                if player_bar is not None:
                    player_bar.update(1)
                    player_bar.set_description(f"players {i}/{len(keys)}  {match.match_id}")
                else:
                    print(f"  players {i}/{len(keys)}  {match.match_id}")
            if player_bar is not None:
                player_bar.close()
            del bufs
        except Exception as exc:  # keep going through the rest of the tournament
            errors.append(f"{match.match_id}: {exc}")
            print(f"  skipped {match.match_id}: {exc}")
        done += 1
        if match_bar is not None:
            match_bar.update(1)
            match_bar.set_description(f"matches {done}/{expected_matches}")
        else:
            print(f"matches {done}/{expected_matches} finished {match.match_id}")
    if match_bar is not None:
        match_bar.close()

    table = pd.concat(
        [coarse.to_frame("coarse"), detailed.to_frame("detailed")],
        ignore_index=True,
    )
    return ProfileResult(
        table=table,
        unknown_codes=sorted(unknown),
        n_matches=done,
        errors=errors,
    )


def player_minute_profile(match: MatchFiles, player_id: str | int) -> pd.DataFrame:
    """One player's regulation-minute speed / |a| / distance / |jerk|.

    Same coverage gate and clock windows as the tournament profiles.
    Streams the match, keeps one jersey, discards the rest.
    """
    meta = _load_json(match.metadata)
    home_id = str((meta.get("homeTeam") or {}).get("id"))
    raw = _load_json(match.roster)
    if isinstance(raw, dict):
        raw = raw.get("players") or raw.get("roster") or []
    hit = None
    for rec in raw:
        pid = str((rec.get("player") or {}).get("id"))
        if pid == str(player_id):
            hit = rec
            break
    if hit is None:
        raise KeyError(f"player_id {player_id} not on roster {match.roster}")
    code = str(hit.get("positionGroupType") or "").upper()
    team_id = str((hit.get("team") or {}).get("id"))
    side = "home" if team_id == home_id else "away"
    jersey = str(hit.get("shirtNumber"))
    nick = str((hit.get("player") or {}).get("nickname") or player_id)
    index = {(side, jersey): code}
    bufs = _stream_match(match, index)
    buf = bufs[(side, jersey)]
    if len(buf) < 10:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "video_time_s": np.asarray(buf.t, dtype=float),
            "periodGameClockTime": np.asarray(buf.clock, dtype=float),
            "x": np.asarray(buf.x, dtype=float),
            "y": np.asarray(buf.y, dtype=float),
            "period": np.asarray(buf.period, dtype=int),
            "frameNum": np.asarray(buf.frame, dtype=int),
        }
    )
    del bufs
    df = mpa.add_stride_kinematics(df)
    df = mpa.add_step_distance(df)
    vals = _player_minute_values(df)
    rows = []
    for mi in range(N_MINUTES):
        for k, metric in enumerate(METRICS):
            rows.append(
                {
                    "match_id": match.match_id,
                    "player_id": str(player_id),
                    "nickname": nick,
                    "position": code,
                    "coarse": COARSE_MAP.get(code),
                    "minute": mi + 1,
                    "metric": metric,
                    "value": float(vals[mi, k]) if np.isfinite(vals[mi, k]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_profiles(table: pd.DataFrame, level: str, save_dir: Path | None = None):
    """Mean ± SD vs regulation minute. One figure per metric."""
    import matplotlib.pyplot as plt

    sub = table.loc[table["level"] == level].copy()
    if sub.empty:
        raise ValueError(f"no rows for level={level!r}")
    if level == "coarse":
        groups = [g for g in COARSE_ORDER if g in set(sub["group"])]
        colors = {"Defender": "#1f4e79", "Midfielder": "#b9770e", "Attacker": "#922b21"}
    else:
        present = list(dict.fromkeys(sub["group"]))
        groups = [g for g in DETAIL_ORDER if g in present] + [g for g in present if g not in DETAIL_ORDER]
        cmap = plt.colormaps["tab10"]
        colors = {g: cmap(i % 10) for i, g in enumerate(groups)}

    figs = []
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(11.5, 4.6), layout="constrained")
        block = sub.loc[sub["metric"] == metric]
        for g in groups:
            sl = block.loc[block["group"] == g].sort_values("minute")
            if sl.empty:
                continue
            x = sl["minute"].to_numpy()
            y = sl["mean"].to_numpy(dtype=float)
            sd = sl["sd"].to_numpy(dtype=float)
            c = colors[g]
            ax.plot(x, y, color=c, lw=1.6, label=g)
            ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.18, linewidth=0)
        ax.axvline(45.5, color="#888888", lw=0.8, ls="--")
        ax.set_xlim(1, 90)
        ax.set_xlabel("regulation minute")
        ylab = {
            "speed": "mean speed (m/s)",
            "acceleration": r"mean $|a|$ (m/s$^2$)",
            "distance": "mean distance (m)",
            "jerk": r"mean $|j|$",
        }[metric]
        ax.set_ylabel(ylab)
        ax.set_title(f"{level}  |  {metric}  |  mean ± SD across player-minutes")
        ax.legend(ncol=min(5, max(len(groups), 1)), fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"wc_profile_{level}_{metric}.png", dpi=140, bbox_inches="tight")
        figs.append(fig)
    return figs
