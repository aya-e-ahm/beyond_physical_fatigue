"""Fixed-weight mixer v2: 7 active series -> cognitive / kinematic / both.

Active series (defaults from player_timeline_suite_nb mixer v2):
  cognitive: tight_w3 (1), pass_C (1), standing (1.5)
  kinematic: ols_distance (1), tm_cum (1), op_cum (1), speed_phi (1)

Normalization:
  nearest C clipped at +/-350 m -> (C+350)/700
  OLS: invert roll_A to degrees, clip +/-42.5 -> (theta+42.5)/85
  speed_phi / pass_C / tight_w3 already on [0,1]
  standing benefit = 1 - worst_vulnerability_5min_norm

  python -m beyond_physical_fatigue.mixer_fixed --folder marcos_acuna_3835
  python -m beyond_physical_fatigue.mixer_fixed --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
WEIGHTS = json.loads((PKG / "weights.json").read_text(encoding="utf-8"))

NEAREST_CLIP_M = float(WEIGHTS["normalization"]["nearest_clip_m"])
OLS_CLIP_DEG = float(WEIGHTS["normalization"]["ols_clip_deg"])
SERIES_WEIGHTS = {k: float(v) for k, v in WEIGHTS["series_weights"].items()}
GROUP_WEIGHTS = {k: float(v) for k, v in WEIGHTS["group_weights"].items()}
COGNITIVE = list(WEIGHTS["groups"]["cognitive"])
KINEMATIC = list(WEIGHTS["groups"]["kinematic"])


def clip_unit(values, half_range: float) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    r = float(half_range)
    return np.clip((y + r) / (2.0 * r), 0.0, 1.0)


def ols_from_roll_a(roll_a, half_deg: float = OLS_CLIP_DEG) -> np.ndarray:
    a = pd.to_numeric(roll_a, errors="coerce").to_numpy(dtype=float)
    theta_deg = np.degrees(a * np.pi - np.pi / 2.0)
    return clip_unit(theta_deg, half_deg)


def series_frame(df: pd.DataFrame, minute_col: str, value) -> pd.DataFrame:
    minute = pd.to_numeric(df[minute_col], errors="coerce")
    if isinstance(value, str):
        y = pd.to_numeric(df[value], errors="coerce")
    else:
        y = pd.Series(np.asarray(value, dtype=float), index=df.index)
    if "period" in df.columns:
        period = pd.to_numeric(df["period"], errors="coerce")
    else:
        m = minute.to_numpy(dtype=float)
        period = pd.Series(
            np.where(m <= 45, 1, np.where(m <= 90, 2, np.where(m <= 105, 3, 4))),
            index=df.index,
        )
    out = pd.DataFrame({"period": period, "minute": minute, "value": y})
    return out.dropna(subset=["period", "minute", "value"])


def interp_on(minutes: np.ndarray, src: pd.DataFrame) -> np.ndarray:
    if src.empty:
        return np.full(len(minutes), np.nan)
    src = src.sort_values("minute")
    x = pd.to_numeric(src["minute"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(src["value"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) == 0:
        return np.full(len(minutes), np.nan)
    return np.interp(minutes, x, y, left=np.nan, right=np.nan)


def load_active_pieces(source_dir: Path) -> dict[str, pd.DataFrame]:
    pieces: dict[str, pd.DataFrame] = {}

    twin = source_dir / "01f_pc_synergy_m_80_1450_timewin.csv"
    if twin.exists():
        df = pd.read_csv(twin)
        if "m_w3" in df.columns:
            pieces["tight_w3"] = series_frame(df, "minute", "m_w3")

    pas = source_dir / "05_pass_shot_quality_w5.csv"
    if pas.exists():
        df = pd.read_csv(pas)
        if "C" in df.columns:
            pieces["pass_C"] = series_frame(df, "t_min", "C")

    standing = source_dir / "06_standing_position.csv"
    if standing.exists():
        df = pd.read_csv(standing)
        if "worst_vulnerability_5min_norm" in df.columns:
            benefit = 1.0 - pd.to_numeric(df["worst_vulnerability_5min_norm"], errors="coerce")
            pieces["standing"] = series_frame(df, "minute", benefit)

    ols = source_dir / "02_ols_line_angle.csv"
    if ols.exists():
        df = pd.read_csv(ols)
        part = df[df["y_metric"] == "distance_m"].copy()
        if not part.empty and "roll_A" in part.columns:
            unit = ols_from_roll_a(part["roll_A"])
            pieces["ols_distance"] = series_frame(part, "minute", unit)

    near = source_dir / "03_self_nearest_distance_lead.csv"
    if near.exists():
        df = pd.read_csv(near)
        for ref, tag in (("teammates", "tm"), ("opponents", "op")):
            part = df[df["reference"] == ref].copy()
            if part.empty:
                continue
            unit = clip_unit(part["C"], NEAREST_CLIP_M)
            pieces[f"{tag}_cum"] = series_frame(part, "minute", unit)

    phi = source_dir / "04_speed_roll10_phi.csv"
    if phi.exists():
        df = pd.read_csv(phi)
        if "phi_roll10" in df.columns:
            pieces["speed_phi"] = series_frame(df, "minute", "phi_roll10")

    return pieces


def build_grid(pieces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    periods = sorted({int(p) for frame in pieces.values() for p in frame["period"].dropna().unique()})
    rows = []
    for period in periods:
        observed = [
            frame.loc[frame["period"] == period, "minute"].to_numpy(dtype=float)
            for frame in pieces.values()
        ]
        observed = [x for x in observed if len(x)]
        if not observed:
            continue
        lo = float(np.min([x.min() for x in observed]))
        hi = float(np.max([x.max() for x in observed]))
        grid = np.arange(np.floor(lo), np.ceil(hi) + 1e-9, 1.0)
        block = {"period": period, "minute": grid}
        for name, frame in pieces.items():
            part = frame[frame["period"] == period]
            block[name] = interp_on(grid, part)
        rows.append(pd.DataFrame(block))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def weighted_mean(frame: pd.DataFrame, names: list[str], weights: dict[str, float]) -> np.ndarray:
    used = [name for name in names if weights.get(name, 0) > 0 and name in frame.columns]
    if not used:
        return np.full(len(frame), np.nan)
    vals = np.column_stack([frame[name].to_numpy(dtype=float) for name in used])
    w = np.array([weights[name] for name in used], dtype=float)
    finite = np.isfinite(vals)
    weighted = np.where(finite, vals * w, 0.0)
    wsum = (finite * w).sum(axis=1)
    return np.divide(weighted.sum(axis=1), wsum, out=np.full(len(frame), np.nan), where=wsum > 0)


def score(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["cognitive"] = weighted_mean(scored, COGNITIVE, SERIES_WEIGHTS)
    scored["kinematic"] = weighted_mean(scored, KINEMATIC, SERIES_WEIGHTS)
    both_v = np.column_stack(
        [
            scored["cognitive"].to_numpy(dtype=float),
            scored["kinematic"].to_numpy(dtype=float),
        ]
    )
    both_w = np.array([GROUP_WEIGHTS["cognitive"], GROUP_WEIGHTS["kinematic"]], dtype=float)
    active = both_w > 0
    finite = np.isfinite(both_v[:, active])
    w = both_w[active]
    weighted = np.where(finite, both_v[:, active] * w, 0.0)
    wsum = (finite * w).sum(axis=1)
    scored["both"] = np.divide(
        weighted.sum(axis=1), wsum, out=np.full(len(scored), np.nan), where=wsum > 0
    )
    return scored


def run_folder(folder: str) -> pd.DataFrame:
    source = DATA / "players" / folder / "source"
    if not source.exists():
        raise SystemExit(f"missing source dir: {source}")
    pieces = load_active_pieces(source)
    missing = [k for k in SERIES_WEIGHTS if k not in pieces]
    if missing:
        print(f"warning {folder}: missing series {missing}")
    grid = build_grid(pieces)
    return score(grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", help="one player_match folder name")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="optional output dir for scored CSVs")
    args = ap.parse_args()
    cases = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))
    folders = [c["folder"] for c in cases]
    if args.folder:
        folders = [args.folder]
    elif not args.all:
        raise SystemExit("pass --folder NAME or --all")

    out_root = args.out
    for folder in folders:
        scored = run_folder(folder)
        print(f"{folder}: n={len(scored)} cols={list(scored.columns)}")
        if out_root is not None:
            dest = out_root / folder
            dest.mkdir(parents=True, exist_ok=True)
            scored.to_csv(dest / "curves_scored.csv", index=False)


if __name__ == "__main__":
    main()
