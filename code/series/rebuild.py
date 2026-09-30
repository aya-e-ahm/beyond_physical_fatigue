"""Rebuild mixer source curves from upstream tables (no pitch-control synergy).

XY / speed / sprint / nearest kernels: ``beyond_physical_fatigue/code/lib/match_player_analysis.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SERIES_DIR = Path(__file__).resolve().parent
SCALES = json.loads((SERIES_DIR / "ols_global_scales.json").read_text(encoding="utf-8"))

WORST_VULN_MEDIAN = 0.0256378650665283
WORST_VULN_P95 = 0.11229975186288345
PC_TIGHT_SX = 80.0
PC_TIGHT_SY = 1450.0
TIME_WINDOWS_MIN = (1.0, 3.0, 5.0, 10.0)


def centered_minute_mean(minutes, values, window_min: float) -> np.ndarray:
    t = np.asarray(minutes, dtype=float)
    y = np.asarray(values, dtype=float)
    half = float(window_min) / 2.0
    inside = np.abs(t[:, None] - t[None, :]) <= half + 1e-6
    finite = np.isfinite(y)
    use = inside & finite[None, :]
    acc = (use * np.where(finite, y, 0.0)).sum(axis=1)
    count = use.sum(axis=1)
    return np.divide(acc, count, out=np.full(len(y), np.nan), where=count > 0)


def centered_minute_mean_by_period(df: pd.DataFrame, window_min: float, value_col: str = "m") -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("period", sort=False).groups.items():
        idx = list(idx)
        out.loc[idx] = centered_minute_mean(df.loc[idx, "minute"], df.loc[idx, value_col], window_min)
    return out


def normalize_worst_vulnerability(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    med, p95 = WORST_VULN_MEDIAN, WORST_VULN_P95
    low = 0.5 * np.maximum(v, 0.0) / med
    high = 0.5 + 0.5 * (v - med) / (p95 - med)
    out = np.where(v <= med, low, high)
    out = np.clip(out, 0.0, 1.0)
    return np.where(np.isfinite(v), out, np.nan)


def capped_synergy_score(individual, team, sx: float = PC_TIGHT_SX, sy: float = PC_TIGHT_SY):
    x = np.asarray(individual, dtype=float)
    y = np.asarray(team, dtype=float)
    u = 0.5 * (np.clip(x / sx, -1.0, 1.0) + 1.0)
    v = 0.5 * (np.clip(y / sy, -1.0, 1.0) + 1.0)
    return u, v, 0.5 * (u + v)


def lwrd_lead_from_residuals(tab: pd.DataFrame) -> pd.DataFrame:
    """1-min distance residuals → cumulative LWRD lead (suite ``03_``)."""
    rows = []
    for period, g in tab.groupby(pd.to_numeric(tab["period"], errors="coerce")):
        g = g.sort_values("t_centre_min")
        t = pd.to_numeric(g["t_centre_min"], errors="coerce")
        for col, name in (("d_resid_tm", "teammates"), ("d_resid_op", "opponents")):
            y = pd.to_numeric(g[col], errors="coerce")
            ok = np.isfinite(t) & np.isfinite(y)
            c = np.full(len(g), np.nan)
            if ok.any():
                c[np.flatnonzero(ok.to_numpy())] = np.cumsum(y[ok].to_numpy())
            rows.append(
                pd.DataFrame(
                    {
                        "section": "self_nearest_distance_lead",
                        "period": int(period),
                        "minute": t,
                        "reference": name,
                        "r": y,
                        "ma7": centered_minute_mean(t.to_numpy(dtype=float), y.to_numpy(dtype=float), 7.0),
                        "C": c,
                        "N": np.clip((c + 600.0) / 1200.0, 0.0, 1.0),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def ols_curve_from_fitted(fitted_distance: pd.DataFrame, fitted_peak: pd.DataFrame | None = None) -> pd.DataFrame:
    frames = []
    for ycol, fitted in (("distance_m", fitted_distance), ("peak_speed", fitted_peak)):
        if fitted is None or fitted.empty:
            continue
        minute_col = "start_clock_min" if "start_clock_min" in fitted.columns else "minute"
        frames.append(
            pd.DataFrame(
                {
                    "section": "ols_line_angle",
                    "y_metric": ycol,
                    "period": pd.to_numeric(fitted["period"], errors="coerce"),
                    "minute": pd.to_numeric(fitted[minute_col], errors="coerce"),
                    "roll_A": pd.to_numeric(fitted["roll_A"], errors="coerce"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rsp_curve_from_residual(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "section": "speed_roll10",
            "minute": pd.to_numeric(df["minute"], errors="coerce"),
            "phi_roll10": pd.to_numeric(df["phi_roll10"], errors="coerce"),
            "roll10_z": pd.to_numeric(df["roll10_z"], errors="coerce")
            if "roll10_z" in df.columns
            else np.nan,
        }
    )


def standing_curve_from_enriched(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.sort_values(["period", "minute"]).copy()
    out["trailing_median_5min"] = pd.to_numeric(out["best_regret_5min"], errors="coerce")
    out["trailing_median_5min_norm"] = np.minimum(out["trailing_median_5min"] + 0.5, 1.0)
    if "worst_vulnerability_5min" in out.columns:
        out["worst_vulnerability_5min_norm"] = normalize_worst_vulnerability(out["worst_vulnerability_5min"])
    out.insert(0, "section", "standing_position")
    return out


def synergy_tight_timewin_from_events(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process only: cap ±80/±1450 and centered windows on saved synergy events.

    Input must already contain individual/team synergy (or ``m_80_1450``).
    Does **not** recompute pitch-control synergy from tracking.
    """
    out = df.copy()
    if "minute" not in out.columns and "game_clock_s" in out.columns:
        out["minute"] = pd.to_numeric(out["game_clock_s"], errors="coerce") / 60.0
    out["period"] = pd.to_numeric(out["period"], errors="coerce")
    out = out.sort_values(["period", "minute"]).reset_index(drop=True)
    if "m_80_1450" not in out.columns:
        _, _, m = capped_synergy_score(out["individual_synergy"], out["team_synergy"])
        out["m_80_1450"] = m
    work = out.assign(m=out["m_80_1450"])
    curve = pd.DataFrame(
        {
            "section": "pc_synergy_80_1450_timewin",
            "period": out["period"],
            "minute": out["minute"],
            "m": out["m_80_1450"],
        }
    )
    for width in TIME_WINDOWS_MIN:
        curve[f"m_w{width:g}"] = centered_minute_mean_by_period(work, width).to_numpy()
    return curve


def ols_scales() -> dict:
    return SCALES
