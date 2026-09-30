"""
Match-player analysis for Morocco–Croatia games (3820, 10516).

Provides preprocessing + five figure families used by
`player_match_analysis_nb.ipynb` and `run_player_match_analysis.py`.

Match-clock design choice
-------------------------
PFF `periodGameClockTime` runs P1 past 45:00 into stoppage, then P2 *restarts*
at 45:00. Mixing both periods on one clock axis would merge P1 stoppage with
the start of P2.

Policy for this project (MAR–CRO 3820 / 10516; tracking is periods 1–2 only):
  * Plot **each period separately** (P1 and P2 figures).
  * Keep **extra time / stoppage inside each period** on that period's plot
    (P1 past 45:00, P2 past 90:00). No dropping of stoppage frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import possession_states as ps

# ---------------------------------------------------------------------------
# Matches (Morocco vs Croatia only)
# ---------------------------------------------------------------------------

def _resolve_football_root() -> Path:
    """Same rules as ``beyond_physical_fatigue.code.paths.football_root``."""
    import os

    env = os.environ.get("FOOTBALL_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    markers = ("3820_MAR_CRO", "10516_CRO_MAR", "possession_cache", "kinematic_profiles")
    # Prefer: package parent (…/football) when this file lives in …/beyond_physical_fatigue/code/lib
    for cand in (here.parent.parent.parent, here.parent.parent, here, *here.parents):
        if any((cand / m).exists() for m in markers):
            return cand.resolve()
    return here.parent.parent.parent.resolve()


ROOT = _resolve_football_root()

MATCHES = {
    "early": {
        "match_id": "3820",
        "dir": ROOT / "3820_MAR_CRO",
        "label": "early | 3820 MAR–CRO",
        "when": "early",
    },
    "mid": {
        "match_id": "10506",
        "dir": ROOT / "10506_JPN_CRO",
        "label": "mid | 10506 JPN–CRO",
        "when": "mid",
    },
    "late": {
        "match_id": "10516",
        "dir": ROOT / "10516_CRO_MAR",
        "label": "late | 10516 CRO–MAR",
        "when": "late",
    },
}

# Matches whose raw player points include vendor ``speed`` (for PFF vs stride-12 compare).
# MAR–CRO early/late (3820 / 10516) do **not** store that field.
VENDOR_SPEED_MATCHES = {
    "10510": {
        "match_id": "10510",
        "dir": ROOT / "10510_CRO_BRZ",
        "label": "CRO–BRZ | 10510",
    },
    "10506": {
        "match_id": "10506",
        "dir": ROOT / "10506_JPN_CRO",
        "label": "JPN–CRO | 10506",
    },
    "3812": {
        "match_id": "3812",
        "dir": ROOT / "3812_SEN_NED",
        "label": "SEN–NED | 3812",
    },
}

CRO_TEAM_ID = "371"
MAR_TEAM_ID = "374"

STRIDE = 12
ACC_MA_USE = 16
ACC_MA_WINDOWS = (16,)
CAP_SPEED_AT_15 = True
SPEED_CAP_MPS = 15.0

ROLL_WIN_MIN = 3.0
ROLL_STEP_MIN = 2.0  # 1 min overlap (step = win − overlap)
ROLL_MIN_FRAMES = 30


def set_rolling_window(
    win_min: float,
    *,
    overlap_s: float | None = None,
    step_min: float | None = None,
) -> tuple[float, float]:
    """Set module-level rolling window length and step (minutes).

    Prefer ``overlap_s`` (seconds of overlap between consecutive windows) or
    pass ``step_min`` directly. Returns ``(ROLL_WIN_MIN, ROLL_STEP_MIN)``.
    """
    global ROLL_WIN_MIN, ROLL_STEP_MIN
    win = float(win_min)
    if step_min is not None:
        step = float(step_min)
    elif overlap_s is not None:
        step = win - float(overlap_s) / 60.0
    else:
        raise ValueError("pass overlap_s= or step_min=")
    if step <= 0 or step > win:
        raise ValueError(f"invalid step {step} for win {win}")
    ROLL_WIN_MIN = win
    ROLL_STEP_MIN = step
    return ROLL_WIN_MIN, ROLL_STEP_MIN


def rolling_overlap_min() -> float:
    return float(ROLL_WIN_MIN) - float(ROLL_STEP_MIN)


def safe_name(s: str) -> str:
    """Filesystem-safe slug for player nicknames in output folder names."""
    import re

    out = re.sub(r"[^\w\-]+", "_", str(s).strip(), flags=re.UNICODE)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "player"


SPEED_BIN = 0.2
X_Q = (0.00, 0.995)
Y_Q = (0.01, 0.99)
AS_MIN_BIN = 15
AS_SCATTER_CAP = 12000

# Sequential colormap for A–S time colouring (one period per figure → clear progression).
TIME_CMAP = plt.colormaps["plasma"]

# Display order for A–S-by-class panels (4 classes; DEAD + UNRESOLVED merged).
AS_CLASS_PANELS = [
    (ps.CTX_PLAYER_ON_BALL, "player on ball"),
    (ps.CTX_TEAM_OFF_BALL, "team in possession (off-ball)"),
    (ps.CTX_OPPOSITION, "out of possession"),
    ("DEAD_OR_UNRESOLVED", "dead / unresolved"),
]
# Back-compat alias
CTX_ORDER = [c for c, _ in AS_CLASS_PANELS]

# Window-level team possession (relative to focal player's team): Method A team layer.
POS_IN = "IN_POSSESSION"
POS_OUT = "OUT_OF_POSSESSION"
POS_NEUTRAL = "DEAD_OR_UNRESOLVED"
POS_COLORS = {
    POS_IN: "#1a7f37",
    POS_OUT: "#b03a2e",
    POS_NEUTRAL: "#95a5a6",
}
POS_LABELS = {
    POS_IN: "in possession (focal team)",
    POS_OUT: "out of possession",
    POS_NEUTRAL: "dead / unresolved",
}
POS_SHADE_ALPHA = 0.20
SERIES_LINE_COLOR = "#1c2833"

# Timeline strip colours (Method A possession_context).
CTX4_TIMELINE = [
    (ps.CTX_PLAYER_ON_BALL, "player on ball", "#1f618d"),
    (ps.CTX_TEAM_OFF_BALL, "team in possession (off-ball)", "#1a7f37"),
    (ps.CTX_OPPOSITION, "out of possession", "#b03a2e"),
    ("DEAD_OR_UNRESOLVED", "dead / unresolved", "#95a5a6"),
]
CTX3_TIMELINE = [
    (POS_IN, "in possession (player on-ball ∪ team)", "#1a7f37"),
    (POS_OUT, "out of possession", "#b03a2e"),
    (POS_NEUTRAL, "dead / unresolved", "#95a5a6"),
]


def _as_class_mask(ctx_series: pd.Series, class_key: str) -> pd.Series:
    s = ctx_series.astype(str)
    if class_key == "DEAD_OR_UNRESOLVED":
        return s.isin([ps.CTX_DEAD, ps.CTX_UNRESOLVED])
    return s == class_key


def collapse_team_poss_class(team_possession: pd.Series) -> pd.Series:
    """Map Method A/B team_possession → IN / OUT / NEUTRAL (focal-team relative)."""
    s = team_possession.astype(str)
    out = pd.Series(np.full(len(s), POS_NEUTRAL, dtype=object), index=s.index)
    out.loc[s == ps.TEAM_FOCAL] = POS_IN
    out.loc[s == ps.TEAM_OPP] = POS_OUT
    return out


def collapse_context_4(possession_context: pd.Series) -> pd.Series:
    """4 analysis classes: on-ball / team-off / out / dead+unresolved."""
    s = possession_context.astype(str)
    out = s.copy()
    out.loc[s.isin([ps.CTX_DEAD, ps.CTX_UNRESOLVED])] = "DEAD_OR_UNRESOLVED"
    return out


def collapse_context_3(possession_context: pd.Series) -> pd.Series:
    """3 classes: focal-team in possession (on-ball ∪ off-ball) / out / dead+unresolved."""
    s = possession_context.astype(str)
    out = pd.Series(np.full(len(s), POS_NEUTRAL, dtype=object), index=s.index)
    out.loc[s.isin([ps.CTX_PLAYER_ON_BALL, ps.CTX_TEAM_OFF_BALL])] = POS_IN
    out.loc[s == ps.CTX_OPPOSITION] = POS_OUT
    out.loc[s.isin([ps.CTX_DEAD, ps.CTX_UNRESOLVED])] = POS_NEUTRAL
    return out


def attach_window_possession(
    tab: pd.DataFrame,
    labeled_tracking: pd.DataFrame,
    win_min: float | None = None,
) -> pd.DataFrame:
    """Majority Method-A team possession inside each rolling window → poss_class."""
    if win_min is None:
        win_min = ROLL_WIN_MIN
    out = tab.copy()
    if out.empty:
        out["poss_class"] = pd.Series(dtype=object)
        return out
    d = labeled_tracking.copy()
    d["_min"] = period_clock_minutes(d)
    d["_cls"] = collapse_team_poss_class(d["team_possession"])
    labels = []
    for tc in out["t_centre_min"].to_numpy(dtype=float):
        t0 = float(tc) - 0.5 * win_min
        t1 = float(tc) + 0.5 * win_min
        w = d.loc[(d["_min"] >= t0) & (d["_min"] < t1), "_cls"]
        if len(w) == 0:
            labels.append(POS_NEUTRAL)
        else:
            labels.append(str(w.value_counts().index[0]))
    out["poss_class"] = labels
    return out


def window_possession_proportions(
    labeled_tracking: pd.DataFrame,
    win_min: float,
    step_min: float,
    period: int | None = None,
    min_frames: int = ROLL_MIN_FRAMES,
) -> pd.DataFrame:
    """Rolling-window Method-A team possession *proportions* (not only majority).

    Returns one row per window with ``p_ip``, ``p_oop``, ``p_dead``, ``p_max``,
    majority ``poss_class``, and frame counts. Times are period-clock minutes.
    """
    d = labeled_tracking.copy()
    if period is not None:
        d = slice_period(d, int(period))
    if d.empty or "team_possession" not in d.columns:
        return pd.DataFrame()

    d["_min"] = period_clock_minutes(d)
    d["_cls"] = collapse_team_poss_class(d["team_possession"])
    t = d["_min"].to_numpy(dtype=float)
    ok = np.isfinite(t)
    if not ok.any():
        return pd.DataFrame()

    starts = window_starts(t[ok], win_min=win_min, step_min=step_min)
    rows = []
    for t0 in starts:
        t1 = float(t0) + float(win_min)
        w = d.loc[(d["_min"] >= t0) & (d["_min"] < t1), "_cls"]
        n = int(len(w))
        if n < min_frames:
            continue
        vc = w.value_counts(normalize=True)
        p_ip = float(vc.get(POS_IN, 0.0))
        p_oop = float(vc.get(POS_OUT, 0.0))
        p_dead = float(vc.get(POS_NEUTRAL, 0.0))
        # renormalize in case of float drift
        s = p_ip + p_oop + p_dead
        if s > 0:
            p_ip, p_oop, p_dead = p_ip / s, p_oop / s, p_dead / s
        p_max = max(p_ip, p_oop, p_dead)
        if p_ip >= p_oop and p_ip >= p_dead:
            maj = POS_IN
        elif p_oop >= p_ip and p_oop >= p_dead:
            maj = POS_OUT
        else:
            maj = POS_NEUTRAL
        rows.append(
            {
                "period": int(period) if period is not None else (
                    int(d["period"].mode().iloc[0]) if "period" in d.columns else np.nan
                ),
                "t0_min": float(t0),
                "t1_min": float(t1),
                "t_centre_min": float(t0) + 0.5 * float(win_min),
                "win_min": float(win_min),
                "step_min": float(step_min),
                "n_frames": n,
                "p_ip": p_ip,
                "p_oop": p_oop,
                "p_dead": p_dead,
                "p_max": p_max,
                "poss_class": maj,
            }
        )
    return pd.DataFrame(rows)


def possession_window_sensitivity(
    bundle: dict,
    win_mins: tuple[float, ...] = (0.5, 1.0, 3.0),
    step_min: float = 0.5,
    dominance_thresholds: tuple[float, ...] = (0.50, 0.60, 0.70),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample sensitivity: state proportions + P_max across window lengths.

    Uses Method A team possession (focal-relative 3-class). Fixed ``step_min``
    across lengths so coverage along the match is comparable.
    Returns ``(windows_long, summary)``.
    """
    labeled = _focal_poss_tracking(bundle)
    parts = []
    for p in periods_in(bundle["tracking"]):
        for w in win_mins:
            tab = window_possession_proportions(
                labeled, win_min=float(w), step_min=float(step_min), period=int(p)
            )
            if not tab.empty:
                parts.append(tab)
    windows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if windows.empty:
        return windows, pd.DataFrame()

    rows = []
    for w, g in windows.groupby("win_min", sort=True):
        row = {
            "win_min": float(w),
            "win_s": float(w) * 60.0,
            "n_windows": int(len(g)),
            "p_max_mean": float(g["p_max"].mean()),
            "p_max_median": float(g["p_max"].median()),
            "p_max_p25": float(g["p_max"].quantile(0.25)),
            "p_max_p75": float(g["p_max"].quantile(0.75)),
            "mean_p_ip": float(g["p_ip"].mean()),
            "mean_p_oop": float(g["p_oop"].mean()),
            "mean_p_dead": float(g["p_dead"].mean()),
        }
        for thr in dominance_thresholds:
            clean = g["p_max"] >= float(thr)
            row[f"frac_pmax_ge_{int(round(thr * 100))}"] = float(clean.mean())
            # among clean: share of each majority class
            sub = g.loc[clean]
            if len(sub):
                row[f"frac_clean_ip_{int(round(thr * 100))}"] = float((sub["poss_class"] == POS_IN).mean())
                row[f"frac_clean_oop_{int(round(thr * 100))}"] = float((sub["poss_class"] == POS_OUT).mean())
                row[f"frac_mixed_{int(round(thr * 100))}"] = float((~clean).mean())
            else:
                row[f"frac_clean_ip_{int(round(thr * 100))}"] = np.nan
                row[f"frac_clean_oop_{int(round(thr * 100))}"] = np.nan
                row[f"frac_mixed_{int(round(thr * 100))}"] = 1.0
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("win_min").reset_index(drop=True)
    return windows, summary


def plot_possession_window_sensitivity(
    bundle: dict,
    windows: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
    dominance_thresholds: tuple[float, ...] = (0.50, 0.60, 0.70),
    save_path: Path | None = None,
):
    """Two-panel figure: P_max distributions by window length + clean-label fractions."""
    if windows is None or summary is None:
        windows, summary = possession_window_sensitivity(
            bundle, dominance_thresholds=dominance_thresholds
        )
    if windows.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_title("no windows")
        return fig

    win_levels = sorted(windows["win_min"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")

    # --- left: P_max histogram / KDE-ish step hist ---
    ax = axes[0]
    colors = {0.5: "#1f4e79", 1.0: "#b03a2e", 3.0: "#1a7f37"}
    for w in win_levels:
        g = windows.loc[windows["win_min"] == w, "p_max"].to_numpy(dtype=float)
        lab = f"{w*60:g}s  (med={np.nanmedian(g):.2f})"
        ax.hist(
            g,
            bins=np.linspace(0.3, 1.0, 15),
            density=True,
            histtype="step",
            lw=2.0,
            color=colors.get(float(w), "#333333"),
            label=lab,
        )
    for thr in dominance_thresholds:
        ax.axvline(thr, color="#7f8c8d", ls=":", lw=1.0, alpha=0.8)
    ax.set_xlabel(r"$P_{\max}=\max(p_{\mathrm{IP}},p_{\mathrm{OOP}},p_{\mathrm{dead}})$")
    ax.set_ylabel("density")
    ax.set_title("Dominance of the leading state inside each window")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    # --- right: fraction meeting dominance threshold ---
    ax = axes[1]
    x = np.arange(len(dominance_thresholds))
    width = 0.25
    for i, w in enumerate(win_levels):
        row = summary.loc[summary["win_min"] == w]
        if row.empty:
            continue
        row = row.iloc[0]
        ys = [row.get(f"frac_pmax_ge_{int(round(thr * 100))}", np.nan) for thr in dominance_thresholds]
        ax.bar(
            x + (i - (len(win_levels) - 1) / 2) * width,
            ys,
            width=width,
            color=colors.get(float(w), "#333333"),
            label=f"{w*60:g}s",
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{int(round(t*100))}%" for t in dominance_thresholds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of windows")
    ax.set_xlabel("minimum dominance for a clean categorical label")
    ax.set_title("How often would a dominance rule keep a label?")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  "
        f"window possession sensitivity (Method A team 3-class)",
        fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def _draw_poss_series(
    ax, t, y, poss, *, win_min: float | None = None, lw=1.6, ms=4.0, legend=False
):
    """Neutral line + markers; translucent axvspan per window by possession class."""
    from matplotlib.patches import Patch

    if win_min is None:
        win_min = ROLL_WIN_MIN
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    poss = np.asarray(poss, dtype=object)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y, poss = t[ok], y[ok], poss[ok]
    if len(t) == 0:
        return
    half = 0.5 * float(win_min)
    for ti, cls in zip(t, poss):
        c = POS_COLORS.get(cls, POS_COLORS[POS_NEUTRAL])
        ax.axvspan(ti - half, ti + half, color=c, alpha=POS_SHADE_ALPHA, lw=0, zorder=0)
    ax.plot(t, y, color=SERIES_LINE_COLOR, lw=lw, zorder=2)
    ax.scatter(t, y, c=SERIES_LINE_COLOR, s=ms ** 2, zorder=3, edgecolors="none")
    if legend:
        ax.legend(
            handles=[
                Patch(
                    facecolor=POS_COLORS[cls],
                    alpha=min(1.0, POS_SHADE_ALPHA * 2.5),
                    edgecolor="none",
                    label=POS_LABELS[cls],
                )
                for cls in (POS_IN, POS_OUT, POS_NEUTRAL)
            ],
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )


def _poss_class_avgs(y, poss, fmt=".1f") -> str:
    """Average of y within IN and OUT windows (neutral omitted from avgs)."""
    y = np.asarray(y, dtype=float)
    poss = np.asarray(poss, dtype=object)
    parts = []
    for cls, short in ((POS_IN, "IN"), (POS_OUT, "OUT")):
        m = (poss == cls) & np.isfinite(y)
        if not m.any():
            parts.append(f"{short} avg=n/a")
        else:
            parts.append(f"{short} avg={format(float(np.nanmean(y[m])), fmt)}")
    return "  |  ".join(parts)


def _focal_poss_tracking(bundle: dict) -> pd.DataFrame:
    """Method A labeled tracking for the focal player (team_possession is focal-relative)."""
    return bundle["tracking_a"]


def _label_runs_minutes(minutes: np.ndarray, labels: np.ndarray) -> list[tuple[str, float, float]]:
    """Contiguous (label, t0_min, t1_min) runs on a sorted period-clock axis."""
    minutes = np.asarray(minutes, dtype=float)
    labels = np.asarray(labels, dtype=object)
    ok = np.isfinite(minutes)
    minutes, labels = minutes[ok], labels[ok]
    if len(minutes) == 0:
        return []
    order = np.argsort(minutes)
    minutes, labels = minutes[order], labels[order]
    runs = []
    i0 = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[i0]:
            runs.append((str(labels[i0]), float(minutes[i0]), float(minutes[i - 1])))
            i0 = i
    return runs


def _plot_timeline_blocks(ax, minutes, labels, class_specs, title: str):
    """Horizontal block rows: one row per class."""
    color_map = {k: c for k, _, c in class_specs}
    y_labels = [lab for _, lab, _ in class_specs]
    key_to_y = {k: i for i, (k, _, _) in enumerate(class_specs)}
    runs = _label_runs_minutes(minutes, labels)
    for lab, t0, t1 in runs:
        if lab not in key_to_y:
            continue
        yi = key_to_y[lab]
        width = max(t1 - t0, 1.0 / 60.0)
        ax.broken_barh(
            [(t0, width)],
            (yi - 0.4, 0.8),
            facecolors=color_map[lab],
            edgecolors="none",
        )
    ax.set_yticks(range(len(class_specs)))
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.8, len(class_specs) - 0.2)
    ax.set_xlabel("period clock (min)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)


def plot_possession_timelines(
    bundle: dict,
    save_path_4: Path | None = None,
    save_path_3: Path | None = None,
    period: int | None = None,
):
    """Two timeline figures per period: 4-class context and 3-class (on-ball⊂team IN)."""
    if period is None:
        figs4, figs3 = {}, {}
        src = _focal_poss_tracking(bundle)
        for p in periods_in(src):
            f4, f3 = plot_possession_timelines(
                bundle,
                save_path_4=period_save_path(save_path_4, p),
                save_path_3=period_save_path(save_path_3, p),
                period=p,
            )
            figs4[p], figs3[p] = f4, f3
        return figs4, figs3

    df = slice_period(_focal_poss_tracking(bundle), period)
    minutes = period_clock_minutes(df).to_numpy(dtype=float)
    ctx = df["possession_context"]
    lab4 = collapse_context_4(ctx).to_numpy(dtype=object)
    lab3 = collapse_context_3(ctx).to_numpy(dtype=object)

    fig4, ax4 = plt.subplots(figsize=(12.0, 3.6), layout="constrained")
    _plot_timeline_blocks(
        ax4,
        minutes,
        lab4,
        CTX4_TIMELINE,
        "Method A · 4 classes · player on-ball separated",
    )
    fig4.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  P{period}  |  "
        f"possession timeline (focal={bundle['team_name']})",
        fontsize=11,
    )
    if save_path_4:
        fig4.savefig(save_path_4, dpi=140, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(12.0, 3.0), layout="constrained")
    _plot_timeline_blocks(
        ax3,
        minutes,
        lab3,
        CTX3_TIMELINE,
        "Method A · 3 classes · player on-ball counted as team in possession",
    )
    fig3.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  P{period}  |  "
        f"team possession timeline (focal={bundle['team_name']})",
        fontsize=11,
    )
    if save_path_3:
        fig3.savefig(save_path_3, dpi=140, bbox_inches="tight")

    return fig4, fig3

HEAT_AGG_S = 1.0
# Fixed y-ranges / bin counts shared across all players (1-s aggregates).
# Chosen to cover ~p999 of WC outfield kinematics while keeping rare spikes off-scale.
HEAT_N_BINS = 40
HEAT_HALF_BIN_FACTOR = 1.4
HEAT_JERK_TIGHT_RANGE_FACTOR = 4.0  # show ymax / 4
HEAT_JERK_TIGHT_BIN_FACTOR = 2.5  # bin width = full_bin_width / 2.5
HEAT_YMAX = {
    "speed": 10.0,  # m/s
    "acc_use": 5.0,  # m/s^2
    "jerk_use": 8.0,  # m/s^3
    "distance_1s": 10.0,  # m per 1-s bin
}


# ---------------------------------------------------------------------------
# Kinematics (same as profiles notebook)
# ---------------------------------------------------------------------------


def _segment_bounds(t, max_gap_factor=2.5):
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n < 2:
        return [(0, n)]
    dt = np.diff(t)
    h = np.nanmedian(dt)
    if not np.isfinite(h) or h <= 0:
        return [(0, n)]
    cuts = np.where(dt > max_gap_factor * h)[0]
    bounds, start = [], 0
    for cut in cuts:
        bounds.append((start, cut + 1))
        start = cut + 1
    bounds.append((start, n))
    return bounds


def _strided_central_diff_1d(x, h, stride):
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = int(stride)
    v = np.full(n, np.nan)
    a = np.full(n, np.nan)
    j = np.full(n, np.nan)
    hs = s * h
    if n > 2 * s and hs > 0:
        v[s:-s] = (x[2 * s :] - x[: -2 * s]) / (2.0 * hs)
        a[s:-s] = (x[2 * s :] - 2.0 * x[s:-s] + x[: -2 * s]) / (hs ** 2)
    if n > 4 * s and hs > 0:
        j[2 * s : -2 * s] = (
            x[4 * s :] - 2.0 * x[3 * s : -s] + 2.0 * x[s : -3 * s] - x[: -4 * s]
        ) / (2.0 * hs ** 3)
    return v, a, j


def central_moving_average(x, window):
    w = int(window)
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x.copy()
    return (
        pd.Series(x)
        .rolling(window=w, center=True, min_periods=w)
        .mean()
        .to_numpy(dtype=float)
    )


def add_stride_kinematics(df, t_col="video_time_s", period_col="period"):
    df = df.sort_values([period_col, t_col, "frameNum"]).copy()
    kin_cols = ["vx", "vy", "speed", "ax", "ay", "acc", "jx", "jy", "jerk", "a_tan"]
    for w in ACC_MA_WINDOWS:
        kin_cols.extend(
            [
                f"ax_ma_{w}",
                f"ay_ma_{w}",
                f"acc_ma_{w}",
                f"jx_ma_{w}",
                f"jy_ma_{w}",
                f"jerk_ma_{w}",
                f"a_tan_ma_{w}",
            ]
        )
    for col in kin_cols:
        df[col] = np.nan
    for period, grp in df.groupby(period_col, sort=False):
        order = grp.index.to_numpy()
        t = grp[t_col].to_numpy(dtype=float)
        for start, end in _segment_bounds(t):
            sl = slice(start, end)
            idx = order[sl]
            tt = t[sl]
            if len(tt) < 2 * STRIDE + 1:
                continue
            h = float(np.median(np.diff(tt)))
            x = pd.to_numeric(grp["x"], errors="coerce").to_numpy(dtype=float)[sl]
            y = pd.to_numeric(grp["y"], errors="coerce").to_numpy(dtype=float)[sl]
            vx, ax, jx = _strided_central_diff_1d(x, h, STRIDE)
            vy, ay, jy = _strided_central_diff_1d(y, h, STRIDE)
            speed = np.sqrt(vx ** 2 + vy ** 2)
            a_tan, _, _ = _strided_central_diff_1d(speed, h, STRIDE)
            df.loc[idx, "vx"] = vx
            df.loc[idx, "vy"] = vy
            df.loc[idx, "speed"] = speed
            df.loc[idx, "a_tan"] = a_tan
            df.loc[idx, "ax"] = ax
            df.loc[idx, "ay"] = ay
            df.loc[idx, "acc"] = np.sqrt(ax ** 2 + ay ** 2)
            df.loc[idx, "jx"] = jx
            df.loc[idx, "jy"] = jy
            df.loc[idx, "jerk"] = np.sqrt(jx ** 2 + jy ** 2)
            for w in ACC_MA_WINDOWS:
                ax_ma = central_moving_average(ax, w)
                ay_ma = central_moving_average(ay, w)
                jx_ma, _, _ = _strided_central_diff_1d(ax_ma, h, STRIDE)
                jy_ma, _, _ = _strided_central_diff_1d(ay_ma, h, STRIDE)
                df.loc[idx, f"ax_ma_{w}"] = ax_ma
                df.loc[idx, f"ay_ma_{w}"] = ay_ma
                df.loc[idx, f"acc_ma_{w}"] = np.sqrt(ax_ma ** 2 + ay_ma ** 2)
                df.loc[idx, f"jx_ma_{w}"] = jx_ma
                df.loc[idx, f"jy_ma_{w}"] = jy_ma
                df.loc[idx, f"jerk_ma_{w}"] = np.sqrt(jx_ma ** 2 + jy_ma ** 2)
                df.loc[idx, f"a_tan_ma_{w}"] = central_moving_average(a_tan, w)
    if CAP_SPEED_AT_15:
        too_fast = df["speed"] > SPEED_CAP_MPS
        df.loc[too_fast, ["vx", "vy", "speed"]] = np.nan
    df["acc_use"] = df[f"acc_ma_{ACC_MA_USE}"]
    df["jerk_use"] = df[f"jerk_ma_{ACC_MA_USE}"]
    df["a_tan_use"] = df[f"a_tan_ma_{ACC_MA_USE}"]
    return df


def add_step_distance(df):
    df = df.sort_values(["period", "video_time_s", "frameNum"]).reset_index(drop=True).copy()
    df["step_dist"] = np.nan
    for _, sub in df.groupby("period", sort=False):
        x = pd.to_numeric(sub["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(sub["video_time_s"], errors="coerce").to_numpy(dtype=float)
        d = np.full(len(sub), np.nan)
        if len(sub) >= 2:
            step = np.hypot(np.diff(x), np.diff(y))
            dt = np.diff(t)
            step = np.where(dt > 0, step, np.nan)
            d[1:] = step
        df.loc[sub.index, "step_dist"] = d
    return df


def period_clock_minutes(df):
    """Period game-clock minutes (includes stoppage within that period)."""
    return pd.to_numeric(df["periodGameClockTime"], errors="coerce") / 60.0


# Back-compat alias used by rolling / peer helpers
continuous_match_minutes = period_clock_minutes


def periods_in(df) -> list[int]:
    if df is None or len(df) == 0 or "period" not in df.columns:
        return []
    vals = pd.to_numeric(df["period"], errors="coerce").dropna().unique()
    return sorted(int(v) for v in vals)


def slice_period(df: pd.DataFrame, period: int) -> pd.DataFrame:
    p = pd.to_numeric(df["period"], errors="coerce")
    return df.loc[p == int(period)].copy()


def mark_substitution_on_ax(
    ax,
    clock_min: float | None,
    label: str = "sub",
    *,
    vline: bool = True,
    color: str = "#6c3483",
) -> None:
    """Draw a clock event. ``vline=True`` when match time is the x-axis."""
    if clock_min is None or not np.isfinite(float(clock_min)):
        return
    text = str(label)
    if vline:
        x0, x1 = ax.get_xlim()
        clock = float(clock_min)
        if clock > x1:
            if (x1 - x0) <= 1.1 and x0 <= 0.05:
                ax.set_xlim(max(0.0, clock - 12.0), clock + 1.5)
            else:
                ax.set_xlim(x0, clock + max(0.6, 0.04 * max(clock - x0, 1.0)))
        ax.axvline(clock, color=color, ls=":", lw=1.2, zorder=4)
        y0, y1 = ax.get_ylim()
        x0, x1 = ax.get_xlim()
        near_right = clock > x0 + 0.72 * (x1 - x0)
        ax.text(
            clock - 0.08 if near_right else clock + 0.08,
            y0 + 0.94 * (y1 - y0),
            text,
            color=color,
            fontsize=8,
            va="top",
            ha="right" if near_right else "left",
            clip_on=False,
        )
        return
    ax.text(
        0.98,
        0.03,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=color,
    )


def period_save_path(save_path: Path | None, period: int) -> Path | None:
    if save_path is None:
        return None
    save_path = Path(save_path)
    return save_path.with_name(f"{save_path.stem}_P{int(period)}{save_path.suffix}")


def window_starts(t_minutes, win_min=None, step_min=None):
    if win_min is None:
        win_min = ROLL_WIN_MIN
    if step_min is None:
        step_min = ROLL_STEP_MIN
    t_lo = float(np.nanmin(t_minutes))
    t_hi = float(np.nanmax(t_minutes))
    start0 = np.floor(t_lo)
    return np.arange(start0, t_hi - win_min + 1e-9, step_min)


# ---------------------------------------------------------------------------
# Resolve player / load match data
# ---------------------------------------------------------------------------


def list_outfield_players(when: str) -> pd.DataFrame:
    spec = MATCHES[when]
    roster = ps.load_roster(spec["dir"] / f"{spec['match_id']}_roster.json", spec["match_id"])
    out = roster.loc[roster["position"].astype(str).str.upper() != "GK"].copy()
    meta = ps.load_json(spec["dir"] / f"{spec['match_id']}_metadata.json")
    home_id = str((meta.get("homeTeam") or {}).get("id"))
    out["side"] = np.where(out["team_id"].astype(str) == home_id, "home", "away")
    out["when"] = when
    out["match_id"] = spec["match_id"]
    return out.reset_index(drop=True)


def find_player(needle: str, when: str | None = None) -> pd.DataFrame:
    """Match by nickname substring or player_id across one or both matches."""
    whens = [when] if when else list(MATCHES)
    rows = []
    for w in whens:
        r = list_outfield_players(w)
        mask = r["nickname"].str.contains(needle, case=False, na=False)
        if str(needle).isdigit():
            mask = mask | (r["player_id"].astype(str) == str(needle))
        hit = r.loc[mask].copy()
        rows.append(hit)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_player_match(
    when: str,
    player_id: str | int,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Load tracking + kinematics + Method A/B possession labels for one player–match."""
    spec = MATCHES[when]
    mid = spec["match_id"]
    cache_dir = Path(cache_dir) if cache_dir else ROOT / "possession_cache" / mid
    cache_dir.mkdir(parents=True, exist_ok=True)

    roster = ps.load_roster(spec["dir"] / f"{mid}_roster.json", mid)
    meta = ps.load_json(spec["dir"] / f"{mid}_metadata.json")
    row = roster.loc[roster["player_id"].astype(str) == str(player_id)]
    if row.empty:
        raise ValueError(f"player_id {player_id} not in roster for match {mid}")
    row = row.iloc[0]
    if str(row["position"]).upper() == "GK":
        raise ValueError("Goalkeepers are excluded from this analysis")

    team_id = str(row["team_id"])
    side = ps.side_for_team(meta, team_id)
    shirt = int(row["shirt_number"])
    nickname = str(row["nickname"])
    opp_team = "Morocco" if team_id == CRO_TEAM_ID else "Croatia"

    # Possession (reuses geometry caches)
    poss = ps.build_match_possession(
        match_id=mid,
        match_dir=spec["dir"],
        focal_player_id=player_id,
        focal_team_id=team_id,
        focal_shirt=shirt,
        focal_side=side,
        cache_dir=cache_dir,
        force_geometry=force,
    )

    # Prefer labeled_a as base tracking; add kinematics
    kin_path = cache_dir / f"{mid}_focal_{player_id}_kin.parquet"
    if kin_path.exists() and not force:
        tracking = pd.read_parquet(kin_path)
    else:
        tracking = poss["focal"].copy()
        tracking = add_stride_kinematics(tracking)
        tracking = add_step_distance(tracking)
        tracking.to_parquet(kin_path, index=False)

    # Attach possession labels by frameNum
    a = poss["labeled_a"][["frameNum", "player_otb", "team_possession", "possession_context", "otb_is_instant"]].drop_duplicates(
        "frameNum"
    )
    b = poss["labeled_b"][
        ["frameNum", "player_otb", "team_possession", "possession_context", "ball_state", "d_focal", "d_opp"]
    ].drop_duplicates("frameNum")
    tracking_a = tracking.merge(a, on="frameNum", how="left", suffixes=("", "_a"))
    tracking_b = tracking.merge(b, on="frameNum", how="left", suffixes=("", "_b"))

    return {
        "when": when,
        "match_id": mid,
        "label": spec["label"],
        "player_id": str(player_id),
        "nickname": nickname,
        "team_id": team_id,
        "team_name": str(row["team"]),
        "opp_team_name": opp_team,
        "side": side,
        "shirt": shirt,
        "tracking": tracking,
        "tracking_a": tracking_a,
        "tracking_b": tracking_b,
        "poss": poss,
        "meta": meta,
        "roster": roster,
        "r_pz": poss["r_pz"],
    }


# ---------------------------------------------------------------------------
# Shared A–S helpers
# ---------------------------------------------------------------------------


def _as_percentiles(spd, y):
    vmax = float(np.nanmax(spd))
    edges = np.arange(0, vmax + SPEED_BIN, SPEED_BIN)
    cat = pd.cut(spd, edges, include_lowest=True)
    g = pd.Series(y).groupby(cat, observed=False)
    centres = np.array([(iv.left + iv.right) / 2 for iv in g.mean().index])
    p90 = g.quantile(0.90).to_numpy()
    p10 = g.quantile(0.10).to_numpy()
    mean = g.mean().to_numpy()
    med = g.median().to_numpy()
    ok = g.size().to_numpy() >= AS_MIN_BIN
    return centres[ok], p90[ok], p10[ok], mean[ok], med[ok]


def _as_arrays(df, positive_only=False):
    spd = pd.to_numeric(df["speed"], errors="coerce")
    y = pd.to_numeric(df["a_tan_use"], errors="coerce")
    clock = pd.to_numeric(df["periodGameClockTime"], errors="coerce")
    m = spd.notna() & y.notna() & clock.notna() & (spd > 0.2)
    if positive_only:
        m = m & (y > 0)
    return (
        spd[m].to_numpy(dtype=float),
        y[m].to_numpy(dtype=float),
        (clock[m] / 60.0).to_numpy(dtype=float),
    )


def _scatter_as_on_ax(
    ax, spd, y, minutes, t_vmin, t_vmax, xlim, ylim, title, *, draw_profiles: bool = True
):
    if len(spd) == 0:
        ax.set_title(title + " (no frames)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        return None
    rng = np.random.default_rng(0)
    if len(spd) > AS_SCATTER_CAP:
        take = rng.choice(len(spd), size=AS_SCATTER_CAP, replace=False)
        sx, sy, st = spd[take], y[take], minutes[take]
    else:
        sx, sy, st = spd, y, minutes
    order = np.argsort(st)
    sx, sy, st = sx[order], sy[order], st[order]
    sc = ax.scatter(
        sx, sy, c=st, s=6, cmap=TIME_CMAP, alpha=0.55, linewidths=0, vmin=t_vmin, vmax=t_vmax
    )
    if draw_profiles:
        centres, p90, p10, mean, med = _as_percentiles(spd, y)
        if len(centres):
            ax.plot(centres, p90, color="#555555", lw=1.4, ls="-")
            ax.plot(centres, p10, color="#555555", lw=1.4, ls="--")
            ax.plot(centres, mean, color="#777777", lw=1.1, ls="-.")
            ax.plot(centres, med, color="#777777", lw=1.1, ls=":")
    ax.axhline(0, color="#888888", lw=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel(r"$a_{\tan}=d|v|/dt$ MA16")
    return sc


def plot_as_time_colored(bundle: dict, save_path: Path | None = None, period: int | None = None):
    """2-panel figure: all a_tan | positive-only, time-coloured (one period)."""
    if period is None:
        figs = {}
        for p in periods_in(bundle["tracking"]):
            figs[p] = plot_as_time_colored(bundle, period_save_path(save_path, p), period=p)
        return figs

    df = slice_period(bundle["tracking"], period)
    spd, y, minutes = _as_arrays(df, False)
    spd_p, y_p, minutes_p = _as_arrays(df, True)
    if len(spd) == 0:
        fig, ax = plt.subplots()
        ax.set_title(f"P{period}: no A–S frames")
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
        return fig
    # Shared axis limits from full match for comparable period panels
    spd_all, y_all, _ = _as_arrays(bundle["tracking"], False)
    x0, x1 = np.quantile(spd_all if len(spd_all) else spd, X_Q)
    y0, y1 = np.quantile(y_all if len(y_all) else y, Y_Q)
    xlim = (max(0.0, x0 - 0.3), x1 + 0.3)
    ylim = (y0 - 0.3, y1 + 0.3)
    t0, t1 = float(np.nanmin(minutes)), float(np.nanmax(minutes))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, layout="constrained")
    sc = _scatter_as_on_ax(
        axes[0], spd, y, minutes, t0, t1, xlim, ylim, "a_tan all (accel + decel)"
    )
    _scatter_as_on_ax(
        axes[1], spd_p, y_p, minutes_p, t0, t1, xlim, ylim, r"$a_{\tan}>0$ only"
    )
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  period {period}  |  "
        f"A–S coloured by time within this period  |  stride {STRIDE}, MA {ACC_MA_USE}",
        fontsize=11,
    )
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.02, fraction=0.03)
        cbar.set_label("period clock (min) · plasma (early→late)")
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_as_by_possession(
    bundle: dict, method: str = "A", save_path: Path | None = None, period: int | None = None
):
    """2×2 A–S panels by possession class for one method (one period).

    Classes: player on ball, team in possession (off-ball), out of possession,
    dead/unresolved. Colour = period clock within this period. Each panel title
    reports that class's share of period frames (%). Scatter points only —
    no mean / median / percentile line profiles.
    """
    if period is None:
        figs = {}
        src = bundle["tracking_a"] if method == "A" else bundle["tracking_b"]
        for p in periods_in(src):
            figs[p] = plot_as_by_possession(
                bundle, method=method, save_path=period_save_path(save_path, p), period=p
            )
        return figs

    df = slice_period(bundle["tracking_a"] if method == "A" else bundle["tracking_b"], period)
    spd_all, y_all, minutes_all = _as_arrays(slice_period(bundle["tracking"], period), False)
    if len(spd_all) == 0:
        fig, ax = plt.subplots()
        ax.set_title(f"P{period}: no frames")
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
        return fig
    x0, x1 = np.quantile(spd_all, X_Q)
    y0, y1 = np.quantile(y_all, Y_Q)
    xlim = (max(0.0, x0 - 0.3), x1 + 0.3)
    ylim = (y0 - 0.3, y1 + 0.3)
    t0, t1 = float(np.nanmin(minutes_all)), float(np.nanmax(minutes_all))
    n_period = int(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), sharex=True, sharey=True, layout="constrained")
    sc = None
    for ax, (ctx_key, label) in zip(axes.ravel(), AS_CLASS_PANELS):
        mask = _as_class_mask(df["possession_context"], ctx_key)
        n_cls = int(mask.sum())
        pct = 100.0 * n_cls / n_period if n_period else 0.0
        sub = df.loc[mask]
        spd, y, minutes = _as_arrays(sub, False)
        title = f"{label}\n{pct:.1f}% of period frames  (n={n_cls:,} / {n_period:,})"
        sc_i = _scatter_as_on_ax(
            ax, spd, y, minutes, t0, t1, xlim, ylim, title, draw_profiles=False
        )
        if sc_i is not None:
            sc = sc_i
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  period {period}  |  "
        f"Method {method} A–S by class  |  colour = time within this period",
        fontsize=11,
    )
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.02, fraction=0.025)
        cbar.set_label("period clock (min) · plasma (early→late)")
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Rolling max
# ---------------------------------------------------------------------------


def rolling_peak_table(df):
    t = continuous_match_minutes(df).to_numpy(dtype=float)
    spd = pd.to_numeric(df["speed"], errors="coerce").to_numpy(dtype=float)
    acc = pd.to_numeric(df["acc_use"], errors="coerce").to_numpy(dtype=float)
    jerk = pd.to_numeric(df["jerk_use"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t)
    if not ok.any():
        return pd.DataFrame()
    rows = []
    for t0 in window_starts(t[ok]):
        t1 = t0 + ROLL_WIN_MIN
        m = ok & (t >= t0) & (t < t1)
        if int(m.sum()) < ROLL_MIN_FRAMES:
            continue

        def _nanmax(x):
            xx = x[m]
            xx = xx[np.isfinite(xx)]
            return float(np.max(xx)) if len(xx) else np.nan

        rows.append(
            {
                "t_centre_min": t0 + 0.5 * ROLL_WIN_MIN,
                "max_speed": _nanmax(spd),
                "max_acc": _nanmax(acc),
                "max_jerk": _nanmax(jerk),
            }
        )
    return pd.DataFrame(rows)


def _shared_ylim(*series_list, pad_frac: float = 0.06, floor0: bool = False):
    """Y-limits spanning one or more series (for matching P1/P2 axes)."""
    chunks = []
    for s in series_list:
        if s is None:
            continue
        a = np.asarray(s, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if len(a):
            chunks.append(a)
    if not chunks:
        return None
    vals = np.concatenate(chunks)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = hi - lo
    pad = (0.1 if span <= 0 else span * pad_frac)
    y0 = 0.0 if floor0 and lo >= 0 else lo - pad
    y1 = hi + pad
    if y1 <= y0:
        y1 = y0 + 1.0
    return (y0, y1)


def _rolling_max_tabs(bundle: dict) -> dict[int, pd.DataFrame]:
    tabs = {}
    for p in periods_in(bundle["tracking"]):
        tab = rolling_peak_table(slice_period(bundle["tracking"], p))
        tabs[p] = attach_window_possession(tab, slice_period(_focal_poss_tracking(bundle), p))
    return tabs


def plot_rolling_max(bundle: dict, save_path: Path | None = None, period: int | None = None):
    tabs = _rolling_max_tabs(bundle)
    specs = (
        ("max_speed", "max speed (m/s)", ".2f"),
        ("max_acc", f"max |a| MA{ACC_MA_USE}", ".2f"),
        ("max_jerk", f"max |j| from MA{ACC_MA_USE}", ".2f"),
    )
    ylims = {
        col: _shared_ylim(*(t[col] for t in tabs.values() if not t.empty), floor0=True)
        for col, _, _ in specs
    }

    def _one(p: int, path: Path | None):
        tab = tabs.get(p, pd.DataFrame())
        fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.4), sharex=True)
        if tab.empty:
            axes[0].set_title("No windows")
        else:
            for i, (ax, (col, ylab, fmt)) in enumerate(zip(axes, specs)):
                _draw_poss_series(
                    ax,
                    tab["t_centre_min"],
                    tab[col],
                    tab["poss_class"],
                    legend=(i == 0),
                )
                ax.set_ylabel(ylab)
                ax.set_title(_poss_class_avgs(tab[col], tab["poss_class"], fmt=fmt), fontsize=9)
                if ylims.get(col) is not None:
                    ax.set_ylim(*ylims[col])
                ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("period clock (min)")
        fig.suptitle(
            f"{bundle['nickname']}  |  {bundle['label']}  |  P{p}  |  rolling max  |  "
            f"{ROLL_WIN_MIN:g} min / {rolling_overlap_min():g} min overlap  |  "
            f"shared y vs other period  |  "
            f"window shade = Method A team possession (focal={bundle['team_name']})",
            fontsize=10,
        )
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=140, bbox_inches="tight")
        return fig

    if period is None:
        return {p: _one(p, period_save_path(save_path, p)) for p in tabs}
    return _one(int(period), save_path)


# ---------------------------------------------------------------------------
# Distance progression (new)
# ---------------------------------------------------------------------------


def distance_window_table(df):
    t = continuous_match_minutes(df).to_numpy(dtype=float)
    step = pd.to_numeric(df["step_dist"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t)
    rows = []
    for t0 in window_starts(t[ok]):
        t1 = t0 + ROLL_WIN_MIN
        m = ok & (t >= t0) & (t < t1)
        if int(m.sum()) < ROLL_MIN_FRAMES:
            continue
        rows.append(
            {
                "t_centre_min": t0 + 0.5 * ROLL_WIN_MIN,
                "window_distance_m": float(np.nansum(step[m])),
            }
        )
    return pd.DataFrame(rows)


def _distance_tabs(bundle: dict) -> dict[int, pd.DataFrame]:
    tabs = {}
    for p in periods_in(bundle["tracking"]):
        df = slice_period(bundle["tracking"], p)
        tabs[p] = attach_window_possession(
            distance_window_table(df), slice_period(_focal_poss_tracking(bundle), p)
        )
    return tabs


def plot_distance_progression(bundle: dict, save_path: Path | None = None, period: int | None = None):
    tabs = _distance_tabs(bundle)
    ylim = _shared_ylim(
        *(t["window_distance_m"] for t in tabs.values() if not t.empty), floor0=True
    )

    def _one(p: int, path: Path | None):
        tab = tabs.get(p, pd.DataFrame())
        fig, ax = plt.subplots(figsize=(11.5, 4.0))
        if not tab.empty:
            _draw_poss_series(
                ax, tab["t_centre_min"], tab["window_distance_m"], tab["poss_class"], legend=True
            )
            avg_txt = _poss_class_avgs(tab["window_distance_m"], tab["poss_class"], fmt=".1f")
        else:
            avg_txt = "no windows"
        ax.set_ylabel("distance in window (m)")
        ax.set_xlabel("period clock (min)")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(
            f"{bundle['nickname']}  |  {bundle['label']}  |  P{p}  |  "
            f"distance / {ROLL_WIN_MIN:g}-min window  |  shared y vs other period  |  {avg_txt}\n"
            f"window shade = Method A team possession (focal={bundle['team_name']})",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=140, bbox_inches="tight")
        return fig

    if period is None:
        return {p: _one(p, period_save_path(save_path, p)) for p in tabs}
    return _one(int(period), save_path)


# ---------------------------------------------------------------------------
# Peer percentiles
# ---------------------------------------------------------------------------


def _percentile_inclusive(focal_val, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(focal_val) or len(values) == 0:
        return np.nan
    return float(100.0 * np.mean(values <= focal_val))


def load_outfield_kin(when: str, force: bool = False) -> pd.DataFrame:
    spec = MATCHES[when]
    mid = spec["match_id"]
    cache = ROOT / "possession_cache" / "outfield_kin"
    cache.mkdir(parents=True, exist_ok=True)
    kin_path = cache / f"{mid}_outfield_kin.parquet"
    if kin_path.exists() and not force:
        return pd.read_parquet(kin_path)

    roster = ps.load_roster(spec["dir"] / f"{mid}_roster.json", mid)
    meta = ps.load_json(spec["dir"] / f"{mid}_metadata.json")
    xy_path = cache / f"{mid}_outfield_xy.parquet"
    xy = ps.extract_outfield_tracking(
        spec["dir"] / f"{mid}_tracking.jsonl",
        roster,
        meta,
        cache_path=xy_path,
        force=force,
    )
    pieces = []
    for pid, g in xy.groupby("player_id", sort=False):
        g = add_stride_kinematics(g.copy())
        g = add_step_distance(g)
        pieces.append(
            g[
                [
                    "frameNum",
                    "period",
                    "video_time_s",
                    "periodGameClockTime",
                    "player_id",
                    "team_id",
                    "speed",
                    "acc_use",
                    "jerk_use",
                    "step_dist",
                ]
            ]
        )
    kin = pd.concat(pieces, ignore_index=True)
    kin.to_parquet(kin_path, index=False)
    return kin


def peer_percentile_table(
    kin: pd.DataFrame, focal_player_id: str, focal_team_id: str, period: int | None = None
):
    from collections import defaultdict

    kin = kin.copy()
    if period is not None:
        kin = slice_period(kin, period)
    kin["match_min"] = period_clock_minutes(kin)
    kin["player_id"] = kin["player_id"].astype(str)
    kin["team_id"] = kin["team_id"].astype(str)
    focal_player_id = str(focal_player_id)
    focal_team_id = str(focal_team_id)

    rows = []
    diag = {
        "n_windows": 0,
        "focal_missing": 0,
        "short_team": 0,
        "short_all": 0,
        "n_team_hist": defaultdict(int),
        "n_all_hist": defaultdict(int),
        "period": period,
    }
    if kin.empty or not np.isfinite(kin["match_min"]).any():
        diag["n_team_hist"] = dict(diag["n_team_hist"])
        diag["n_all_hist"] = dict(diag["n_all_hist"])
        return pd.DataFrame(), diag

    for t0 in window_starts(kin["match_min"].to_numpy(dtype=float)):
        t1 = t0 + ROLL_WIN_MIN
        w = kin.loc[(kin["match_min"] >= t0) & (kin["match_min"] < t1)]
        present = []
        for pid, g in w.groupby("player_id"):
            if len(g) < ROLL_MIN_FRAMES:
                continue
            present.append(
                {
                    "player_id": pid,
                    "team_id": str(g["team_id"].iloc[0]),
                    "mean_speed": float(np.nanmean(pd.to_numeric(g["speed"], errors="coerce"))),
                    "mean_acc": float(np.nanmean(pd.to_numeric(g["acc_use"], errors="coerce"))),
                    "mean_jerk": float(np.nanmean(pd.to_numeric(g["jerk_use"], errors="coerce"))),
                    "total_distance": float(np.nansum(pd.to_numeric(g["step_dist"], errors="coerce"))),
                }
            )
        if not present:
            continue
        pdf = pd.DataFrame(present)
        if focal_player_id not in set(pdf["player_id"]):
            diag["focal_missing"] += 1
            continue
        team_pdf = pdf.loc[pdf["team_id"] == focal_team_id]
        n_team, n_all = len(team_pdf), len(pdf)
        diag["n_team_hist"][n_team] += 1
        diag["n_all_hist"][n_all] += 1
        if n_team < 10:
            diag["short_team"] += 1
        if n_all < 20:
            diag["short_all"] += 1
        focal = pdf.loc[pdf["player_id"] == focal_player_id].iloc[0]
        row = {"t_centre_min": t0 + 0.5 * ROLL_WIN_MIN, "n_team": n_team, "n_all": n_all, "period": period}
        for m in ("mean_speed", "mean_acc", "mean_jerk", "total_distance"):
            fv = float(focal[m])
            row[f"{m}_team_pct"] = _percentile_inclusive(fv, team_pdf[m].to_numpy())
            row[f"{m}_all_pct"] = _percentile_inclusive(fv, pdf[m].to_numpy())
        rows.append(row)
    diag["n_windows"] = len(rows)
    diag["n_team_hist"] = dict(diag["n_team_hist"])
    diag["n_all_hist"] = dict(diag["n_all_hist"])
    return pd.DataFrame(rows), diag


def plot_peer_percentiles(bundle: dict, save_path: Path | None = None, period: int | None = None):
    kin = load_outfield_kin(bundle["when"])
    # Build both periods so figures stay comparable; percentile y is fixed 0–100.
    all_periods = periods_in(kin)
    tabs, diags = {}, {}
    for p in all_periods:
        tab, diag = peer_percentile_table(kin, bundle["player_id"], bundle["team_id"], period=p)
        tabs[p] = attach_window_possession(tab, slice_period(_focal_poss_tracking(bundle), p))
        diags[p] = diag
    # Percentile axis is intrinsically shared; keep fixed limits on every period figure.
    y_pct = (-2.0, 102.0)

    def _one(p: int, path: Path | None):
        tab = tabs.get(p, pd.DataFrame())
        diag = diags.get(p, {})
        fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.4), sharex=True)
        panels = [
            ("mean_speed", "mean speed pct vs team"),
            ("mean_acc", "mean |a| pct vs team"),
            ("mean_jerk", "mean |j| pct vs team"),
            ("total_distance", "total distance pct vs team"),
        ]
        for i, (ax, (key, ylab)) in enumerate(zip(axes, panels)):
            col = f"{key}_team_pct"
            if not tab.empty:
                _draw_poss_series(
                    ax, tab["t_centre_min"], tab[col], tab["poss_class"], legend=(i == 0)
                )
                ax.set_title(_poss_class_avgs(tab[col], tab["poss_class"], fmt=".1f"), fontsize=9)
            ax.axhline(50, color="#888", ls=":", lw=1)
            ax.set_ylim(*y_pct)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("period clock (min)")
        fig.suptitle(
            f"{bundle['nickname']}  |  {bundle['label']}  |  P{p}  |  "
            f"peer percentiles vs teammate outfield (~10)  |  shared y 0–100  |  "
            f"window shade = Method A team possession (focal={bundle['team_name']})\n"
            f"windows={diag.get('n_windows', 0)} short_team={diag.get('short_team', 0)} "
            f"focal_missing={diag.get('focal_missing', 0)}",
            fontsize=10,
        )
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=140, bbox_inches="tight")
        return fig, tab, diag

    if period is None:
        figs, out_tabs, out_diags = {}, {}, {}
        for p in all_periods:
            fig, tab, diag = _one(p, period_save_path(save_path, p))
            figs[p], out_tabs[p], out_diags[p] = fig, tab, diag
        return figs, out_tabs, out_diags

    return _one(int(period), save_path)


# ---------------------------------------------------------------------------
# Spectrogram-like kinematic heatmaps (from profiles_as cell)
# ---------------------------------------------------------------------------


def prepare_1s_data(df, interval_s=HEAT_AGG_S):
    d = df.copy().sort_values(["period", "periodGameClockTime"]).reset_index(drop=True)
    for col in ["periodGameClockTime", "x", "y", "speed", "acc_use", "jerk_use", "step_dist"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    if "step_dist" not in d.columns:
        dx = d.groupby("period")["x"].diff()
        dy = d.groupby("period")["y"].diff()
        d["step_dist"] = np.sqrt(dx ** 2 + dy ** 2)
    d["time_bin"] = np.floor(d["periodGameClockTime"] / interval_s).astype("Int64")
    sec = (
        d.groupby(["period", "time_bin"], as_index=False)
        .agg(
            periodGameClockTime=("periodGameClockTime", "mean"),
            speed=("speed", "mean"),
            acc_use=("acc_use", "mean"),
            jerk_use=("jerk_use", "mean"),
            distance_1s=("step_dist", "sum"),
            n_frames=("speed", "count"),
        )
    )
    return sec


def sliding_histogram_matrix(df, value_col, bin_edges, window_s=180.0, step_s=120.0):
    d = df[["periodGameClockTime", value_col]].dropna()
    if d.empty:
        raise ValueError(f"No valid values for {value_col}")
    time = d["periodGameClockTime"].to_numpy(dtype=float)
    values = d[value_col].to_numpy(dtype=float)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    t_min, t_max = float(np.nanmin(time)), float(np.nanmax(time))
    starts = np.arange(t_min, t_max - window_s + step_s, step_s)
    histograms, time_centres = [], []
    for start in starts:
        end = start + window_s
        vals = values[(time >= start) & (time < end)]
        counts, _ = np.histogram(vals, bins=bin_edges)
        histograms.append(counts)
        time_centres.append((start + end) / 2.0)
    H = np.asarray(histograms).T
    return H, np.asarray(time_centres), bin_centres


def _heatmap_metrics():
    return {
        "speed": ("speed", "1-s mean speed (m/s)"),
        "acceleration": ("acc_use", f"1-s mean |a| MA{ACC_MA_USE}"),
        "jerk": ("jerk_use", f"1-s mean |j| MA{ACC_MA_USE}"),
        "distance": ("distance_1s", "distance per second (m)"),
    }


def _heatmap_scale(zoom: str = "full"):
    """Return (y_max_by_col, n_bins, zoom_label) for a heatmap variant."""
    zoom = zoom.lower()
    if zoom == "full":
        ymax = dict(HEAT_YMAX)
        n_bins = HEAT_N_BINS
        label = f"full fixed range, {n_bins} bins"
    elif zoom == "half":
        ymax = {k: v / 2.0 for k, v in HEAT_YMAX.items()}
        n_bins = int(round(HEAT_N_BINS * HEAT_HALF_BIN_FACTOR))
        label = f"lower half of fixed range, {n_bins} bins ({HEAT_HALF_BIN_FACTOR:g}×)"
    elif zoom in ("jerk_tight", "tight"):
        ymax = {"jerk_use": HEAT_YMAX["jerk_use"] / HEAT_JERK_TIGHT_RANGE_FACTOR}
        n_bins = int(round(HEAT_N_BINS * HEAT_JERK_TIGHT_BIN_FACTOR))
        label = (
            f"jerk lower 1/{HEAT_JERK_TIGHT_RANGE_FACTOR:g} of fixed range, "
            f"{n_bins} bins (bin width ÷{HEAT_JERK_TIGHT_BIN_FACTOR:g})"
        )
    else:
        raise ValueError(f"Unknown heatmap zoom={zoom!r}; use full|half|jerk_tight")
    return ymax, n_bins, label


def _pcolormesh_hist(ax, H, times_min, bin_centres):
    if len(times_min) > 1:
        dt = float(np.median(np.diff(times_min)))
    else:
        dt = ROLL_STEP_MIN
    if len(bin_centres) > 1:
        dy = float(np.median(np.diff(bin_centres)))
    else:
        dy = 1.0
    t_edges = np.concatenate(
        [
            [times_min[0] - dt / 2],
            times_min[:-1] + np.diff(times_min) / 2,
            [times_min[-1] + dt / 2],
        ]
    )
    y_edges = np.concatenate(
        [
            [bin_centres[0] - dy / 2],
            bin_centres[:-1] + np.diff(bin_centres) / 2,
            [bin_centres[-1] + dy / 2],
        ]
    )
    return ax.pcolormesh(t_edges, y_edges, H, shading="auto", cmap="viridis")


def plot_kinematic_heatmaps(
    bundle: dict,
    save_path: Path | None = None,
    zoom: str = "full",
    period: int | None = None,
):
    """
    2×2 sliding-histogram heatmaps with fixed shared y-limits (one period).

    zoom:
      - "full": HEAT_YMAX, HEAT_N_BINS
      - "half": ymax/2, ~1.4× bins (finer view of the dense lower half)
    Values above ymax are omitted from the histogram (not shown on the y-axis).
    """
    if period is None:
        figs = {}
        for p in periods_in(bundle["tracking"]):
            figs[p] = plot_kinematic_heatmaps(
                bundle, period_save_path(save_path, p), zoom=zoom, period=p
            )
        return figs

    if zoom not in ("full", "half"):
        raise ValueError("plot_kinematic_heatmaps zoom must be 'full' or 'half'")
    sec = prepare_1s_data(slice_period(bundle["tracking"], period))
    metrics = _heatmap_metrics()
    ymax_map, n_bins, zoom_label = _heatmap_scale(zoom)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), layout="constrained")
    window_s = ROLL_WIN_MIN * 60.0
    step_s = ROLL_STEP_MIN * 60.0
    for ax, (name, (col, ylab)) in zip(axes.ravel(), metrics.items()):
        ymax = float(ymax_map[col])
        edges = np.linspace(0.0, ymax, n_bins + 1)
        try:
            H, times, y_bins = sliding_histogram_matrix(sec, col, edges, window_s, step_s)
        except ValueError:
            ax.set_title(f"{name}: no data")
            continue
        pcm = _pcolormesh_hist(ax, H, times / 60.0, y_bins)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="count")
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel("period clock (min)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{name}  |  0–{ymax:g}")
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  P{period}  |  sliding hist  |  {zoom_label}  |  "
        f"{ROLL_WIN_MIN:g} min / {rolling_overlap_min():g} min overlap (1-s agg)",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_jerk_heatmap_tight(
    bundle: dict, save_path: Path | None = None, period: int | None = None
):
    """Single-panel jerk heatmap on the lower 1/4 of HEAT_YMAX with 2.5× more bins."""
    if period is None:
        figs = {}
        for p in periods_in(bundle["tracking"]):
            figs[p] = plot_jerk_heatmap_tight(bundle, period_save_path(save_path, p), period=p)
        return figs

    sec = prepare_1s_data(slice_period(bundle["tracking"], period))
    ymax_map, n_bins, zoom_label = _heatmap_scale("jerk_tight")
    ymax = float(ymax_map["jerk_use"])
    edges = np.linspace(0.0, ymax, n_bins + 1)
    window_s = ROLL_WIN_MIN * 60.0
    step_s = ROLL_STEP_MIN * 60.0
    fig, ax = plt.subplots(figsize=(12.0, 4.8), layout="constrained")
    try:
        H, times, y_bins = sliding_histogram_matrix(
            sec, "jerk_use", edges, window_s, step_s
        )
        pcm = _pcolormesh_hist(ax, H, times / 60.0, y_bins)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="count")
        ax.set_ylim(0.0, ymax)
        ax.set_title(f"jerk  |  0–{ymax:g}  |  {n_bins} bins")
    except ValueError:
        ax.set_title("jerk: no data")
    ax.set_xlabel("period clock (min)")
    ax.set_ylabel(f"1-s mean |j| MA{ACC_MA_USE}")
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  P{period}  |  {zoom_label}",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Phase-duration histograms (3-class Method A)
# ---------------------------------------------------------------------------


PHASE3_HIST_SPECS = [
    (POS_IN, "in possession", POS_COLORS[POS_IN]),
    (POS_OUT, "out of possession", POS_COLORS[POS_OUT]),
    (POS_NEUTRAL, "dead / unresolved", POS_COLORS[POS_NEUTRAL]),
]


def _nice_bin_width_seconds(durations_s: np.ndarray) -> float:
    """Freedman–Diaconis bin width, snapped to a readable second grid."""
    d = np.asarray(durations_s, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 2:
        return 5.0
    edges = np.histogram_bin_edges(d, bins="fd")
    raw = float(np.median(np.diff(edges))) if len(edges) > 1 else 5.0
    if not np.isfinite(raw) or raw <= 0:
        raw = float(np.std(d) / 2) if len(d) else 5.0
    candidates = np.array([0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300], dtype=float)
    return float(candidates[np.argmin(np.abs(candidates - raw))])


def phase_duration_table(bundle: dict, period: int | None = None) -> pd.DataFrame:
    """Contiguous Method-A 3-class phase lengths (seconds) for the focal player."""
    df = _focal_poss_tracking(bundle)
    if period is not None:
        df = slice_period(df, period)
    if df.empty:
        return pd.DataFrame(columns=["label", "t0", "t1", "duration_s", "duration_min", "period"])

    times = pd.to_numeric(df["video_time_s"], errors="coerce").to_numpy(dtype=float)
    labels = collapse_context_3(df["possession_context"]).to_numpy(dtype=object)
    segs = ps.contiguous_segments(times, labels)
    if segs.empty:
        return segs

    finite = times[np.isfinite(times)]
    dt = float(np.nanmedian(np.diff(np.sort(finite)))) if len(finite) > 1 else 0.04
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.04
    # contiguous_segments uses last-first; add one sample so duration matches frame span
    segs = segs.copy()
    segs["duration_s"] = (segs["t1"] - segs["t0"] + dt).clip(lower=dt)
    segs["duration_min"] = segs["duration_s"] / 60.0
    if period is not None:
        segs["period"] = int(period)
    elif "period" in df.columns:
        # majority period of frames in segment (approx via t mid → nearest frames)
        segs["period"] = np.nan
    return segs


def plot_phase_duration_histograms(
    bundle: dict, save_path: Path | None = None, period: int | None = None, unit: str = "auto"
):
    """Histograms of continuous 3-class phase lengths (Method A).

    unit: 's' | 'min' | 'auto' (minutes if pooled median duration ≥ 45 s).
    When period is None, one figure spanning all periods in the match (still one match).
    """
    if period is None and save_path is not None:
        # Also emit per-period copies when saving a stem path? Keep single whole-match figure
        # plus optional per-period if caller loops. Here: one combined figure for the match.
        pass

    pieces = []
    periods = periods_in(_focal_poss_tracking(bundle)) if period is None else [int(period)]
    for p in periods:
        tab = phase_duration_table(bundle, period=p)
        if not tab.empty:
            tab = tab.copy()
            tab["period"] = p
            pieces.append(tab)
    segs = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True, layout="constrained")
    if segs.empty:
        for ax in axes:
            ax.set_title("no phases")
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
        return fig, segs

    all_dur = segs["duration_s"].to_numpy(dtype=float)
    use_min = (unit == "min") or (unit == "auto" and float(np.nanmedian(all_dur)) >= 45.0)
    if unit == "s":
        use_min = False
    scale = 1.0 / 60.0 if use_min else 1.0
    xlab = "phase length (min)" if use_min else "phase length (s)"
    bin_w_s = _nice_bin_width_seconds(all_dur)
    bin_w = bin_w_s * scale
    x_max = float(np.nanmax(all_dur)) * scale
    edges = np.arange(0.0, x_max + bin_w, bin_w)
    if len(edges) < 2:
        edges = np.array([0.0, bin_w])

    for ax, (key, title, color) in zip(axes, PHASE3_HIST_SPECS):
        d = segs.loc[segs["label"] == key, "duration_s"].to_numpy(dtype=float) * scale
        d = d[np.isfinite(d)]
        if len(d):
            ax.hist(d, bins=edges, color=color, edgecolor="white", linewidth=0.4, alpha=0.9)
            ax.axvline(float(np.median(d)), color="#1c2833", ls="--", lw=1.2, label=f"median={np.median(d):.2g}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{title}\nn={len(d):,}", fontsize=10)
        ax.set_xlabel(xlab)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("count (phases)")

    per = f"P{period}" if period is not None else "P1+P2"
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  {per}  |  "
        f"Method A phase lengths (3-class)  |  bin={bin_w_s:g}s"
        + (f" ({bin_w:g} min)" if use_min else ""),
        fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, segs


# ---------------------------------------------------------------------------
# Self-nearest scatter (1-min windows; dynamic spatial NN)
# ---------------------------------------------------------------------------

SELF_NEAREST_K = 2
SELF_NEAREST_WIN_S = 60.0
SELF_NEAREST_MIN_COVERAGE = 0.75  # require ≥75% of expected frames in a 1-min bin
SELF_NEAREST_FPS_FALLBACK = 29.97
SELF_NEAREST_ROLL_MEDIAN_MIN = 9.0  # centered rolling median on 1-min residual series
# Single-hue darkening (light → dark gray) so time progression is obvious within a period
SELF_NEAREST_CMAP = LinearSegmentedColormap.from_list(
    "self_nearest_time",
    ["#F0F0F0", "#B0B0B0", "#606060", "#101010"],
)


def _align_outfield_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    if "team_id" in out.columns:
        out["team_id"] = out["team_id"].astype(str)
    out["frameNum"] = pd.to_numeric(out["frameNum"], errors="coerce")
    out["period"] = pd.to_numeric(out["period"], errors="coerce")
    return out


def load_outfield_panel(when: str, force: bool = False) -> pd.DataFrame:
    """Outfield panel with x, y, speed, step_dist (for spatial nearest-neighbour work)."""
    kin = _align_outfield_keys(load_outfield_kin(when, force=force))
    need = {"x", "y", "speed", "step_dist"}
    if need.issubset(set(kin.columns)):
        return kin.drop_duplicates(["frameNum", "period", "player_id"], keep="first")

    spec = MATCHES[when]
    mid = spec["match_id"]
    cache = ROOT / "possession_cache" / "outfield_kin"
    xy_path = cache / f"{mid}_outfield_xy.parquet"
    roster = ps.load_roster(spec["dir"] / f"{mid}_roster.json", mid)
    meta = ps.load_json(spec["dir"] / f"{mid}_metadata.json")
    xy = _align_outfield_keys(
        ps.extract_outfield_tracking(
            spec["dir"] / f"{mid}_tracking.jsonl",
            roster,
            meta,
            cache_path=xy_path,
            force=force,
        )
    )
    cols = ["frameNum", "period", "player_id", "x", "y"]
    kin = kin.drop_duplicates(["frameNum", "period", "player_id"], keep="first")
    xy = xy.drop_duplicates(["frameNum", "period", "player_id"], keep="first")
    return kin.merge(xy[cols], on=["frameNum", "period", "player_id"], how="left")


def _estimate_tracking_fps(video_time_s: np.ndarray | pd.Series) -> float:
    t = pd.to_numeric(pd.Series(video_time_s), errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) < 3:
        return SELF_NEAREST_FPS_FALLBACK
    t = np.unique(np.sort(t))
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return SELF_NEAREST_FPS_FALLBACK
    med = float(np.median(dt))
    if not np.isfinite(med) or med <= 0:
        return SELF_NEAREST_FPS_FALLBACK
    return float(1.0 / med)


def self_nearest_minute_table(bundle: dict, k: int = SELF_NEAREST_K) -> pd.DataFrame:
    """1-min aggregates: focal vs dynamic nearest-k teammates / opponents.

    At each frame, pick the k spatially nearest teammates and k nearest opponents
    (Euclidean pitch distance), average their speed and step_dist, then aggregate
    into non-overlapping 1-minute bins on periodGameClockTime (per period).

    Distance columns (`d_tm`, `d_op`) are **local-neighbour distance output**: the
    accumulated per-frame mean displacement of whichever k neighbours are nearest
    at each instant (identities may change within the minute) — not the path of
    two fixed players.
    """
    panel = load_outfield_panel(bundle["when"])
    focal_id = str(bundle["player_id"])
    team_id = str(bundle["team_id"])
    panel = panel.copy()
    panel["player_id"] = panel["player_id"].astype(str)
    panel["team_id"] = panel["team_id"].astype(str)
    panel["period"] = pd.to_numeric(panel["period"], errors="coerce")
    panel["frameNum"] = pd.to_numeric(panel["frameNum"], errors="coerce")
    panel = panel.drop_duplicates(["period", "frameNum", "player_id"], keep="first")

    for col in ("x", "y", "speed", "step_dist", "periodGameClockTime", "frameNum", "video_time_s"):
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    focal = panel.loc[
        panel["player_id"] == focal_id,
        ["frameNum", "period", "periodGameClockTime", "video_time_s", "x", "y", "speed", "step_dist"],
    ].rename(
        columns={"x": "fx", "y": "fy", "speed": "s_player", "step_dist": "d_player"}
    )
    if focal.empty:
        return pd.DataFrame()

    fps = _estimate_tracking_fps(focal["video_time_s"])
    min_frames = int(np.ceil(SELF_NEAREST_MIN_COVERAGE * fps * SELF_NEAREST_WIN_S))

    others = panel.loc[
        panel["player_id"] != focal_id,
        ["frameNum", "period", "player_id", "team_id", "x", "y", "speed", "step_dist"],
    ]
    # Full-frame merge of every outfield row onto the focal exploded memory on
    # longer WC tracking files. Join in period + frame chunks and keep only
    # the k nearest teammates / opponents per frame.
    chunk_frames = 400
    tm_parts: list[pd.DataFrame] = []
    op_parts: list[pd.DataFrame] = []
    focal = focal.drop_duplicates(["period", "frameNum"], keep="first")
    others = others.drop_duplicates(["period", "frameNum", "player_id"], keep="first")
    for period, o_per in others.groupby("period", sort=False):
        f_per = focal.loc[focal["period"] == period]
        if f_per.empty or o_per.empty:
            continue
        frame_ids = f_per["frameNum"].drop_duplicates().to_numpy()
        for i0 in range(0, len(frame_ids), chunk_frames):
            keep = set(frame_ids[i0 : i0 + chunk_frames].tolist())
            m = o_per.loc[o_per["frameNum"].isin(keep)].merge(
                f_per.loc[
                    f_per["frameNum"].isin(keep),
                    ["frameNum", "period", "periodGameClockTime", "fx", "fy", "s_player", "d_player"],
                ],
                on=["frameNum", "period"],
                how="inner",
            )
            if m.empty:
                continue
            m["dist"] = np.hypot(m["x"] - m["fx"], m["y"] - m["fy"])
            m = m.loc[
                np.isfinite(m["dist"])
                & np.isfinite(m["speed"])
                & np.isfinite(m["s_player"])
                & np.isfinite(m["step_dist"])
                & np.isfinite(m["d_player"])
            ]
            if m.empty:
                continue
            m["is_tm"] = m["team_id"] == team_id
            tm_c = (
                m.loc[m["is_tm"]]
                .sort_values(["period", "frameNum", "dist"])
                .groupby(["period", "frameNum"], sort=False)
                .head(k)
            )
            op_c = (
                m.loc[~m["is_tm"]]
                .sort_values(["period", "frameNum", "dist"])
                .groupby(["period", "frameNum"], sort=False)
                .head(k)
            )
            if not tm_c.empty:
                tm_parts.append(tm_c)
            if not op_c.empty:
                op_parts.append(op_c)
    if not tm_parts or not op_parts:
        return pd.DataFrame()
    tm = pd.concat(tm_parts, ignore_index=True)
    op = pd.concat(op_parts, ignore_index=True)
    tm_agg = tm.groupby(["period", "frameNum"], as_index=False).agg(
        s_tm=("speed", "mean"),
        d_tm=("step_dist", "mean"),
        n_tm=("speed", "size"),
    )
    op_agg = op.groupby(["period", "frameNum"], as_index=False).agg(
        s_op=("speed", "mean"),
        d_op=("step_dist", "mean"),
        n_op=("speed", "size"),
    )
    frames = focal.merge(tm_agg, on=["period", "frameNum"], how="inner").merge(
        op_agg, on=["period", "frameNum"], how="inner"
    )
    frames = frames.loc[(frames["n_tm"] >= k) & (frames["n_op"] >= k)].copy()
    if frames.empty:
        return pd.DataFrame()

    frames["minute_bin"] = np.floor(
        pd.to_numeric(frames["periodGameClockTime"], errors="coerce") / SELF_NEAREST_WIN_S
    )
    rows = []
    for (period, minute_bin), g in frames.groupby(["period", "minute_bin"], sort=True):
        if not np.isfinite(minute_bin):
            continue
        if len(g) < min_frames:
            continue
        s_player = float(np.nanmean(g["s_player"]))
        s_tm = float(np.nanmean(g["s_tm"]))
        s_op = float(np.nanmean(g["s_op"]))
        d_player = float(np.nansum(g["d_player"]))
        d_tm = float(np.nansum(g["d_tm"]))  # local-neighbour distance output
        d_op = float(np.nansum(g["d_op"]))
        rows.append(
            {
                "period": int(period),
                "minute_bin": float(minute_bin),
                "t_centre_min": float(minute_bin) + 0.5,
                "s_player": s_player,
                "s_tm": s_tm,
                "s_op": s_op,
                "d_player": d_player,
                "d_tm": d_tm,
                "d_op": d_op,
                "s_resid_tm": s_player - s_tm,
                "s_resid_op": s_player - s_op,
                "d_resid_tm": d_player - d_tm,
                "d_resid_op": d_player - d_op,
                "n_frames": int(len(g)),
                "coverage": float(len(g) / (fps * SELF_NEAREST_WIN_S)),
                "fps_est": fps,
                "min_frames_req": min_frames,
            }
        )
    return pd.DataFrame(rows)


def plot_self_nearest_scatter(
    bundle: dict, save_path: Path | None = None, period: int | None = None
):
    """Self-nearest 2×2 scatters — one figure per period (stoppage kept).

    Colour = period clock within that period (light→dark gray, early→late). Shared x/y
    limits across P1/P2 for each metric family. Returns dict[period]→fig when
    period is None, else (fig, tab_period).
    """
    tab_all = self_nearest_minute_table(bundle)
    periods = (
        sorted(int(p) for p in tab_all["period"].dropna().unique())
        if not tab_all.empty
        else periods_in(bundle.get("tracking"))
    )
    if not periods:
        periods = [1, 2]

    # Shared axis limits across periods (comparable halves)
    def _lims(cols):
        if tab_all.empty:
            return None
        vals = np.concatenate([tab_all[c].to_numpy(dtype=float) for c in cols])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return None
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.05 * (hi - lo + 1e-9)
        return (lo - pad, hi + pad)

    speed_lim = _lims(["s_player", "s_tm", "s_op"])
    dist_lim = _lims(["d_player", "d_tm", "d_op"])

    panels_spec = [
        ("s_tm", "s_player", "nearest-2 teammates speed (m/s)", "player speed (m/s)", "Speed vs teammates", "speed"),
        ("s_op", "s_player", "nearest-2 opponents speed (m/s)", "player speed (m/s)", "Speed vs opponents", "speed"),
        ("d_tm", "d_player", "local-neighbour distance output · TM (m)", "player distance (m)", "Distance vs teammates", "dist"),
        ("d_op", "d_player", "local-neighbour distance output · OP (m)", "player distance (m)", "Distance vs opponents", "dist"),
    ]

    def _one(p: int, path: Path | None):
        tab = tab_all.loc[tab_all["period"] == int(p)].copy() if not tab_all.empty else pd.DataFrame()
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.0), layout="constrained")
        if tab.empty:
            axes[0, 0].set_title(f"P{p}: no self-nearest windows")
            fig.suptitle(
                f"{bundle['nickname']}  |  {bundle['label']}  |  P{p}  |  self-nearest",
                fontsize=11,
            )
            if path:
                fig.savefig(path, dpi=140, bbox_inches="tight")
            return fig, tab

        t = tab["t_centre_min"].to_numpy(dtype=float)
        t_ok = t[np.isfinite(t)]
        t0 = float(np.nanmin(t_ok)) if len(t_ok) else 0.0
        t1 = float(np.nanmax(t_ok)) if len(t_ok) else 1.0
        if t1 <= t0:
            t1 = t0 + 1.0

        sc = None
        for ax, (xcol, ycol, xlab, ylab, title, kind) in zip(axes.ravel(), panels_spec):
            x = tab[xcol].to_numpy(dtype=float)
            y = tab[ycol].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
            if not ok.any():
                ax.set_title(f"{title}\n(no data)")
                continue
            sc = ax.scatter(
                x[ok],
                y[ok],
                c=t[ok],
                cmap=SELF_NEAREST_CMAP,
                vmin=t0,
                vmax=t1,
                s=36,
                alpha=0.85,
                edgecolors="none",
            )
            lim = speed_lim if kind == "speed" else dist_lim
            if lim is None:
                lo = float(np.nanmin([x[ok].min(), y[ok].min()]))
                hi = float(np.nanmax([x[ok].max(), y[ok].max()]))
                pad = 0.05 * (hi - lo + 1e-9)
                lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, color="#666666", ls="--", lw=1.0, zorder=0)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.25)

        if sc is not None:
            cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
            cbar.set_label("period clock (min) · gray light→dark (early→late)")
        cov = float(tab["coverage"].median()) if "coverage" in tab.columns else np.nan
        fig.suptitle(
            f"{bundle['nickname']}  |  {bundle['label']}  |  P{p}  |  self-nearest scatter  |  "
            f"dynamic spatial top-{SELF_NEAREST_K}  |  1-min bins (ET kept)  |  "
            f"coverage≥{SELF_NEAREST_MIN_COVERAGE:.0%}  |  n={len(tab)}  |  "
            f"median coverage={cov:.0%}",
            fontsize=10,
        )
        if path:
            fig.savefig(path, dpi=140, bbox_inches="tight")
        return fig, tab

    if period is None:
        figs, tabs = {}, {}
        for p in periods:
            fig, tab = _one(p, period_save_path(save_path, p))
            figs[p], tabs[p] = fig, tab
        return figs, tabs

    return _one(int(period), save_path)


def _rolling_window_1d(
    t: np.ndarray, y: np.ndarray, half_win_min: float, how: str = "median"
) -> np.ndarray:
    """Centered rolling statistic on irregular 1-min samples (half window in minutes)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    for i in range(len(t)):
        if not np.isfinite(t[i]):
            continue
        m = np.isfinite(t) & np.isfinite(y) & (np.abs(t - t[i]) <= float(half_win_min))
        if not m.any():
            continue
        vals = y[m]
        out[i] = float(np.median(vals) if how == "median" else np.mean(vals))
    return out


def _rolling_median_1d(t: np.ndarray, y: np.ndarray, half_win_min: float) -> np.ndarray:
    """Centered rolling median on irregular 1-min samples (half window in minutes)."""
    return _rolling_window_1d(t, y, half_win_min, how="median")


def _rolling_mean_1d(t: np.ndarray, y: np.ndarray, half_win_min: float) -> np.ndarray:
    """Centered moving average on irregular 1-min samples (half window in minutes)."""
    return _rolling_window_1d(t, y, half_win_min, how="mean")


def _self_nearest_phase(period: int, t_min: float) -> str:
    """Early / middle / late within a period (period-clock bands; P2 starts ~45)."""
    t = float(t_min)
    if int(period) == 1:
        if t < 15.0:
            return "early"
        if t < 30.0:
            return "middle"
        return "late"
    # period 2 (and any later): clock typically resumes at 45
    if t < 60.0:
        return "early"
    if t < 75.0:
        return "middle"
    return "late"


def enrich_self_nearest_possession(bundle: dict, tab: pd.DataFrame) -> pd.DataFrame:
    """Majority Method-A team possession inside each 1-min self-nearest bin."""
    if tab is None or tab.empty:
        return tab
    labeled = _focal_poss_tracking(bundle)
    out = attach_window_possession(tab.copy(), labeled, win_min=SELF_NEAREST_WIN_S / 60.0)
    out["phase"] = [
        _self_nearest_phase(int(r.period), float(r.t_centre_min)) for r in out.itertuples()
    ]
    return out


def plot_self_nearest_residual_progression(
    bundle: dict,
    save_path: Path | None = None,
    metric: str = "speed",
    roll_median_min: float = SELF_NEAREST_ROLL_MEDIAN_MIN,
    tab: pd.DataFrame | None = None,
    sub_clock_min: float | None = None,
    sub_period: int | None = None,
    sub_label: str = "sub",
    sub_color: str = "#6c3483",
):
    """Main residual view: time vs residual, colour=possession, TM+OP on one panel.

    Overlay centered rolling median (default 9 min). P1|P2 side by side.
    ``metric``: ``speed`` → s_resid_*; ``distance`` → d_resid_*.
    Pass ``tab`` to reuse a precomputed minute table.
    """
    if metric not in ("speed", "distance"):
        raise ValueError("metric must be 'speed' or 'distance'")
    prefix = "s" if metric == "speed" else "d"
    col_tm, col_op = f"{prefix}_resid_tm", f"{prefix}_resid_op"
    ylab = (
        r"$r=v_{\mathrm{player}}-v_{\mathrm{nearby}}$ (m/s)"
        if metric == "speed"
        else r"$r=d_{\mathrm{player}}-d_{\mathrm{nearby}}$ (m)"
    )

    if tab is None:
        tab = enrich_self_nearest_possession(bundle, self_nearest_minute_table(bundle))
    periods = (
        sorted(int(p) for p in tab["period"].dropna().unique()) if not tab.empty else [1, 2]
    )
    n = max(len(periods), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.4), sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]

    half = 0.5 * float(roll_median_min)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    y_all = []
    for ax, p in zip(axes, periods):
        g = tab.loc[tab["period"] == int(p)].sort_values("t_centre_min") if not tab.empty else pd.DataFrame()
        if g.empty:
            ax.set_title(f"P{p}: no data")
            ax.axhline(0.0, color="#666", ls="--", lw=1.0)
            continue
        t = g["t_centre_min"].to_numpy(dtype=float)
        for col, marker, ms, label in (
            (col_tm, "o", 28, "vs teammates"),
            (col_op, "s", 22, "vs opponents"),
        ):
            y = g[col].to_numpy(dtype=float)
            ok = np.isfinite(t) & np.isfinite(y)
            if not ok.any():
                continue
            y_all.append(y[ok])
            colors = [POS_COLORS.get(c, POS_COLORS[POS_NEUTRAL]) for c in g.loc[ok, "poss_class"]]
            ax.scatter(
                t[ok],
                y[ok],
                c=colors,
                marker=marker,
                s=ms,
                alpha=0.75,
                edgecolors="none",
                zorder=2,
                label=label,
            )
            med = _rolling_median_1d(t, y, half)
            ax.plot(
                t,
                med,
                color="#1c2833" if col == col_tm else "#5d6d7e",
                lw=2.2 if col == col_tm else 1.8,
                ls="-" if col == col_tm else "--",
                zorder=3,
                label=f"rolling median · {label}",
            )
        ax.axhline(0.0, color="#666666", ls=":", lw=1.1, zorder=1)
        ax.set_xlabel("period clock (min)")
        ax.set_title(f"P{p}", fontsize=11)
        ax.grid(True, alpha=0.25)

    if y_all:
        vals = np.concatenate(y_all)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = 0.08 * (hi - lo + 1e-9)
        m = max(abs(lo - pad), abs(hi + pad), 0.2)
        axes[0].set_ylim(-m, m)
    for ax, p in zip(axes, periods):
        if sub_period is None or int(sub_period) == int(p):
            mark_substitution_on_ax(
                ax, sub_clock_min, sub_label, vline=True, color=sub_color
            )
    axes[0].set_ylabel(ylab)

    handles = [
        Patch(facecolor=POS_COLORS[POS_IN], edgecolor="none", label=POS_LABELS[POS_IN]),
        Patch(facecolor=POS_COLORS[POS_OUT], edgecolor="none", label=POS_LABELS[POS_OUT]),
        Patch(facecolor=POS_COLORS[POS_NEUTRAL], edgecolor="none", label=POS_LABELS[POS_NEUTRAL]),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333", markersize=7, label="residual vs TM"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#333", markersize=6, label="residual vs OP"),
        Line2D([0], [0], color="#1c2833", lw=2.2, label=f"roll. median TM ({roll_median_min:g} min)"),
        Line2D([0], [0], color="#5d6d7e", lw=1.8, ls="--", label=f"roll. median OP ({roll_median_min:g} min)"),
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.92)

    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  "
        f"self-nearest {metric} residual progression  |  "
        f"colour=Method A team possession  |  dynamic top-{SELF_NEAREST_K}",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, tab


def plot_self_nearest_residual_phases(
    bundle: dict,
    save_path: Path | None = None,
    metric: str = "speed",
):
    """Early / middle / late residual distributions within each period (box + points).

    Reports late−early median Δ for TM and OP residuals. P1|P2 side by side.
    """
    if metric not in ("speed", "distance"):
        raise ValueError("metric must be 'speed' or 'distance'")
    prefix = "s" if metric == "speed" else "d"
    col_tm, col_op = f"{prefix}_resid_tm", f"{prefix}_resid_op"
    ylab = "speed residual (m/s)" if metric == "speed" else "distance residual (m)"

    tab = enrich_self_nearest_possession(bundle, self_nearest_minute_table(bundle))
    periods = (
        sorted(int(p) for p in tab["period"].dropna().unique()) if not tab.empty else [1, 2]
    )
    phase_order = ["early", "middle", "late"]
    n = max(len(periods), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.6), sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]

    delta_lines = []
    for ax, p in zip(axes, periods):
        g = tab.loc[tab["period"] == int(p)].copy() if not tab.empty else pd.DataFrame()
        if g.empty:
            ax.set_title(f"P{p}: no data")
            continue
        # long form for seaborn-free boxplot
        xpos = []
        data = []
        colors = []
        labels_x = []
        tick_pos = []
        for i, phase in enumerate(phase_order):
            sub = g.loc[g["phase"] == phase]
            for j, (col, color, tag) in enumerate(
                ((col_tm, "#1f4e79", "TM"), (col_op, "#b03a2e", "OP"))
            ):
                vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
                x = i * 2.4 + j
                xpos.append(x)
                data.append(vals)
                colors.append(color)
                labels_x.append(f"{phase}\n{tag}")
                tick_pos.append(x)
        for x, vals, color in zip(xpos, data, colors):
            if len(vals) == 0:
                continue
            bp = ax.boxplot(
                [vals],
                positions=[x],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="#111", lw=1.4),
            )
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.35)
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.25
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=14,
                color=color,
                alpha=0.55,
                edgecolors="none",
                zorder=3,
            )
        ax.axhline(0.0, color="#666", ls=":", lw=1.0)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels_x, fontsize=7)
        ax.set_title(f"P{p}", fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)

        def _med(phase, col):
            v = pd.to_numeric(g.loc[g["phase"] == phase, col], errors="coerce").dropna()
            return float(v.median()) if len(v) else np.nan

        d_tm = _med("late", col_tm) - _med("early", col_tm)
        d_op = _med("late", col_op) - _med("early", col_op)
        delta_lines.append(f"P{p}: Δ_TM={d_tm:+.3f}, Δ_OP={d_op:+.3f}")
        ax.text(
            0.02,
            0.98,
            f"late−early median\nTM {d_tm:+.3f}  OP {d_op:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#ccc"),
        )

    axes[0].set_ylabel(ylab)
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  "
        f"self-nearest {metric} residual · early/middle/late  |  "
        f"{'  ·  '.join(delta_lines)}",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, tab


def plot_self_nearest_residual_cumulative(
    bundle: dict,
    save_path: Path | None = None,
    metric: str = "speed",
):
    """Cumulative signed residual C(t)=Σ r(τ). Slope change is the diagnostic."""
    if metric not in ("speed", "distance"):
        raise ValueError("metric must be 'speed' or 'distance'")
    prefix = "s" if metric == "speed" else "d"
    col_tm, col_op = f"{prefix}_resid_tm", f"{prefix}_resid_op"
    ylab = "cumulative speed residual" if metric == "speed" else "cumulative distance residual"

    tab = enrich_self_nearest_possession(bundle, self_nearest_minute_table(bundle))
    periods = (
        sorted(int(p) for p in tab["period"].dropna().unique()) if not tab.empty else [1, 2]
    )
    n = max(len(periods), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.2), sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, periods):
        g = tab.loc[tab["period"] == int(p)].sort_values("t_centre_min") if not tab.empty else pd.DataFrame()
        if g.empty:
            ax.set_title(f"P{p}: no data")
            continue
        t = g["t_centre_min"].to_numpy(dtype=float)
        for col, color, ls, label in (
            (col_tm, "#1f4e79", "-", "vs teammates"),
            (col_op, "#b03a2e", "--", "vs opponents"),
        ):
            y = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(t) & np.isfinite(y)
            if not ok.any():
                continue
            c = np.cumsum(y[ok])
            ax.plot(t[ok], c, color=color, ls=ls, lw=2.0, label=label)
        ax.axhline(0.0, color="#666", ls=":", lw=1.0)
        ax.set_xlabel("period clock (min)")
        ax.set_title(f"P{p}", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel(ylab)
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  "
        f"cumulative self-nearest {metric} residual  |  "
        f"slope ≈ sustained relative under/over-performance",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, tab


def plot_self_nearest_vs_references(
    bundle: dict,
    save_path: Path | None = None,
    metric: str = "speed",
    *,
    cumulative: bool = True,
    tab: pd.DataFrame | None = None,
    roll_mean_min: float | None = None,
    sub_clock_min: float | None = None,
    sub_period: int | None = None,
    sub_label: str = "sub",
    sub_color: str = "#6c3483",
):
    """Residual vs nearest-2, drawn as a lead: up/green = beating, down/red = trailing.

    Same ``r(t)`` as the residual suite (no possession colour). ``cumulative=True``
    plots ``C(t)=Σ r(τ)``; ``False`` plots the 1-min residual itself. One row per
    reference group (teammates, opponents).
    ``roll_mean_min`` overlays a centered moving average on the non-cumulative
    residual (green/red fill follows the average; the raw minute series stays thin).
    Pass ``tab`` to reuse a precomputed minute table.
    """
    if metric not in ("speed", "distance"):
        raise ValueError("metric must be 'speed' or 'distance'")
    prefix = "s" if metric == "speed" else "d"
    series = (
        (f"{prefix}_resid_tm", "vs nearest-2 teammates", "#1f4e79"),
        (f"{prefix}_resid_op", "vs nearest-2 opponents", "#b03a2e"),
    )
    if cumulative:
        ylab = r"running lead $C(t)=\sum r$  (↑ beats nearest-2)"
        pos_label = "beating nearest-2  (C > 0)"
        neg_label = "trailing nearest-2  (C < 0)"
        kind = "running lead"
        extra = r"same C(t)=Σ r(τ)  ·  curve up = beats nearest-2  ·  curve down = trails"
    else:
        ylab = (
            r"$r=v_{\mathrm{player}}-v_{\mathrm{nearby}}$ (m/s)  (↑ beats nearest-2)"
            if metric == "speed"
            else r"$r=d_{\mathrm{player}}-d_{\mathrm{nearby}}$ (m)  (↑ beats nearest-2)"
        )
        pos_label = "beating nearest-2  (r > 0)"
        neg_label = "trailing nearest-2  (r < 0)"
        kind = "1-min residual"
        extra = "same r(t)  ·  curve up = beats nearest-2  ·  curve down = trails"
        if roll_mean_min is not None and float(roll_mean_min) > 0:
            pos_label = "beating nearest-2  (avg > 0)"
            neg_label = "trailing nearest-2  (avg < 0)"
            extra = (
                f"centered moving average {float(roll_mean_min):g} min  ·  "
                "thin line = raw 1-min r(t)"
            )

    if tab is None:
        tab = enrich_self_nearest_possession(bundle, self_nearest_minute_table(bundle))
    periods = (
        sorted(int(p) for p in tab["period"].dropna().unique()) if not tab.empty else [1, 2]
    )
    n = max(len(periods), 1)
    fig, axes = plt.subplots(
        2, n, figsize=(6.2 * n, 6.6), sharex="col", sharey=True, layout="constrained"
    )
    axes = np.asarray(axes).reshape(2, n)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    y_all = []
    for col_i, p in enumerate(periods):
        g = (
            tab.loc[tab["period"] == int(p)].sort_values("t_centre_min")
            if not tab.empty
            else pd.DataFrame()
        )
        for row_i, (col, row_title, color) in enumerate(series):
            ax = axes[row_i, col_i]
            if g.empty:
                ax.set_title(f"P{p}: no data")
                ax.axhline(0.0, color="#666", ls=":", lw=1.0)
                continue
            t = g["t_centre_min"].to_numpy(dtype=float)
            y = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(t) & np.isfinite(y)
            if not ok.any():
                ax.set_title(f"P{p}: no data")
                ax.axhline(0.0, color="#666", ls=":", lw=1.0)
                continue
            t, y = t[ok], y[ok]
            raw = None
            if cumulative:
                t_plot = np.concatenate([[t[0] - 0.5], t])
                z = np.concatenate([[0.0], np.cumsum(y)])
            elif roll_mean_min is not None and float(roll_mean_min) > 0:
                t_plot, raw = t, y
                z = _rolling_mean_1d(t, y, 0.5 * float(roll_mean_min))
            else:
                t_plot, z = t, y
            y_all.append(z[np.isfinite(z)] if np.isfinite(z).any() else z)
            if raw is not None:
                ax.plot(t_plot, raw, color="#b0b7bf", lw=0.9, zorder=2)
            ax.fill_between(
                t_plot,
                0.0,
                z,
                where=z >= 0,
                color="#1a7f37",
                alpha=0.28,
                interpolate=True,
                zorder=1,
                linewidth=0,
            )
            ax.fill_between(
                t_plot,
                0.0,
                z,
                where=z < 0,
                color="#c0392b",
                alpha=0.28,
                interpolate=True,
                zorder=1,
                linewidth=0,
            )
            ax.plot(t_plot, z, color=color, lw=2.2, zorder=3)
            ax.axhline(0.0, color="#666666", ls=":", lw=1.1, zorder=2)
            ax.set_title(f"P{p}  ·  {row_title}", fontsize=11)
            ax.grid(True, alpha=0.25)
            finite_z = z[np.isfinite(z)]
            end = float(finite_z[-1]) if len(finite_z) else 0.0
            ax.text(
                0.98,
                0.05 if end < 0 else 0.95,
                "trailing" if end < 0 else "beating",
                transform=ax.transAxes,
                va="bottom" if end < 0 else "top",
                ha="right",
                fontsize=8,
                color="#c0392b" if end < 0 else "#1a7f37",
                fontweight="bold",
            )
            if row_i == 1:
                ax.set_xlabel("period clock (min)")

    if y_all:
        vals = np.concatenate(y_all)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = 0.08 * (hi - lo + 1e-9)
        m = max(abs(lo - pad), abs(hi + pad), 0.2)
        axes[0, 0].set_ylim(-m, m)
    for col_i, p in enumerate(periods):
        if sub_period is not None and int(sub_period) != int(p):
            continue
        for row_i in range(2):
            mark_substitution_on_ax(
                axes[row_i, col_i],
                sub_clock_min,
                sub_label,
                vline=True,
                color=sub_color,
            )
    axes[0, 0].set_ylabel(ylab)
    axes[1, 0].set_ylabel(ylab)
    handles = [
        Patch(facecolor="#1a7f37", alpha=0.45, edgecolor="none", label=pos_label),
        Patch(facecolor="#c0392b", alpha=0.45, edgecolor="none", label=neg_label),
    ]
    if roll_mean_min is not None and float(roll_mean_min) > 0 and not cumulative:
        handles.extend(
            [
                Line2D([0], [0], color="#1f4e79", lw=2.2, label=f"moving avg ({float(roll_mean_min):g} min)"),
                Line2D([0], [0], color="#b0b7bf", lw=0.9, label="raw 1-min residual"),
            ]
        )
    axes[0, -1].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92)
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  "
        f"self-nearest {metric} {kind}  |  {extra}",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, tab


def plot_self_nearest_distance_bounded(
    bundle: dict,
    save_path: Path | None = None,
    *,
    cumulative: bool = True,
    min_sum_m: float = 1.0,
    mapping: str = "symmetric",
):
    """Distance score in [0, 1]. Neutral **0.5** = equal distance.

    ``mapping="symmetric"``: ``S=(d_p-d_n)/(d_p+d_n)``, ``N=(S+1)/2``.
    ``mapping="linear_pm100"``: ``q=d_p/d_n-1``, clip to [-1, 1] (−100%…+100%),
    then ``N=(q+1)/2`` so 0% → 0.5. Prints how many points were clipped.
    ``cumulative=True`` uses period-to-date sums; ``False`` uses that minute only.
    Dashed overlay is unbounded ``q`` (twin axis). No possession colour.
    """
    if mapping not in ("symmetric", "linear_pm100"):
        raise ValueError("mapping must be 'symmetric' or 'linear_pm100'")
    series = (
        ("d_player", "d_tm", "vs nearest-2 teammates", "#1f4e79"),
        ("d_player", "d_op", "vs nearest-2 opponents", "#b03a2e"),
    )
    span = "cumulative" if cumulative else "1-min"
    if mapping == "linear_pm100":
        kind = f"{span} N  ·  linear clip ±100%"
        ylab = r"$N=(q_{\mathrm{clip}}+1)/2$  (0.5 = 0%)"
        map_tag = r"q clipped to [-100%, +100%]  →  N=(q+1)/2"
    else:
        kind = f"{span} N  ·  S-map"
        ylab = r"$N$  (0.5 = equal distance)"
        map_tag = r"N=(S+1)/2,  S=(d_p-d_n)/(d_p+d_n)"

    tab = enrich_self_nearest_possession(bundle, self_nearest_minute_table(bundle))
    periods = (
        sorted(int(p) for p in tab["period"].dropna().unique()) if not tab.empty else [1, 2]
    )
    n = max(len(periods), 1)
    fig, axes = plt.subplots(
        2, n, figsize=(6.4 * n, 6.8), sharex="col", sharey=True, layout="constrained"
    )
    axes = np.asarray(axes).reshape(2, n)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    floor = float(min_sum_m)
    q_all = []
    n_valid = 0
    n_clip_hi = 0
    n_clip_lo = 0
    for col_i, p in enumerate(periods):
        g = (
            tab.loc[tab["period"] == int(p)].sort_values("t_centre_min")
            if not tab.empty
            else pd.DataFrame()
        )
        for row_i, (col_p, col_n, row_title, color) in enumerate(series):
            ax = axes[row_i, col_i]
            ax.axhline(0.5, color="#666666", ls=":", lw=1.1, zorder=2)
            ax.set_ylim(0.0, 1.0)
            if g.empty:
                ax.set_title(f"P{p}: no data")
                continue
            t = g["t_centre_min"].to_numpy(dtype=float)
            dp = pd.to_numeric(g[col_p], errors="coerce").to_numpy(dtype=float)
            dn = pd.to_numeric(g[col_n], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(t) & np.isfinite(dp) & np.isfinite(dn) & (dp >= 0) & (dn >= 0)
            if not ok.any():
                ax.set_title(f"P{p}: no data")
                continue
            t, dp, dn = t[ok], dp[ok], dn[ok]
            if cumulative:
                Dp = np.cumsum(dp)
                Dn = np.cumsum(dn)
            else:
                Dp, Dn = dp, dn
            denom = Dp + Dn
            valid = denom >= floor
            if not valid.any():
                ax.set_title(f"P{p}: no data")
                continue
            t_v, Dp_v, Dn_v = t[valid], Dp[valid], Dn[valid]
            q = np.full(Dn_v.shape, np.nan, dtype=float)
            q_ok = Dn_v >= floor
            q[q_ok] = Dp_v[q_ok] / Dn_v[q_ok] - 1.0
            q_all.append(q[np.isfinite(q)])
            n_valid += int(np.isfinite(q).sum())
            hi = np.isfinite(q) & (q > 1.0)
            lo = np.isfinite(q) & (q < -1.0)
            n_clip_hi += int(hi.sum())
            n_clip_lo += int(lo.sum())
            if mapping == "linear_pm100":
                N = np.full(q.shape, np.nan, dtype=float)
                finite = np.isfinite(q)
                N[finite] = np.clip(0.5 * (q[finite] + 1.0), 0.0, 1.0)
            else:
                S = (Dp_v - Dn_v) / (Dp_v + Dn_v)
                N = 0.5 * (S + 1.0)

            ax.fill_between(
                t_v,
                0.5,
                N,
                where=np.isfinite(N) & (N >= 0.5),
                color="#1a7f37",
                alpha=0.28,
                interpolate=True,
                zorder=1,
                linewidth=0,
            )
            ax.fill_between(
                t_v,
                0.5,
                N,
                where=np.isfinite(N) & (N < 0.5),
                color="#c0392b",
                alpha=0.28,
                interpolate=True,
                zorder=1,
                linewidth=0,
            )
            ax.plot(t_v, N, color=color, lw=2.2, zorder=3, label="N [0, 1]")
            clipped = hi | lo
            if clipped.any():
                ax.scatter(
                    t_v[clipped],
                    N[clipped],
                    s=32,
                    marker="x",
                    c="#111111",
                    linewidths=1.15,
                    zorder=4,
                )
            ax2 = ax.twinx()
            ax2.plot(t_v, q, color="#7f8c8d", ls="--", lw=1.15, alpha=0.85, zorder=2, label="q (unbounded %)")
            ax2.axhline(0.0, color="#b0b0b0", ls=":", lw=0.8)
            ax2.axhline(1.0, color="#b0b0b0", ls=":", lw=0.7)
            ax2.axhline(-1.0, color="#b0b0b0", ls=":", lw=0.7)
            ax2.tick_params(axis="y", labelsize=8, colors="#7f8c8d")
            if col_i == n - 1:
                ax2.set_ylabel(r"$q=d_p/d_n-1$", color="#7f8c8d", fontsize=8)
            else:
                ax2.set_ylabel("")
            n_panel = int(np.isfinite(q).sum())
            n_c = int(clipped.sum())
            clip_bit = f"  ·  clipped {n_c}/{n_panel}" if n_c else ""
            ax.set_title(f"P{p}  ·  {row_title}{clip_bit}", fontsize=11)
            ax.grid(True, alpha=0.25)
            finite_N = N[np.isfinite(N)]
            if finite_N.size and abs(float(finite_N[-1]) - 0.5) >= 0.02:
                end = float(finite_N[-1])
                ax.text(
                    0.98,
                    0.08 if end < 0.5 else 0.92,
                    f"N={end:.2f}",
                    transform=ax.transAxes,
                    va="bottom" if end < 0.5 else "top",
                    ha="right",
                    fontsize=8,
                    color="#c0392b" if end < 0.5 else "#1a7f37",
                    fontweight="bold",
                )
            if row_i == 1:
                ax.set_xlabel("period clock (min)")

    axes[0, 0].set_ylabel(ylab)
    axes[1, 0].set_ylabel(ylab)
    n_clip = n_clip_hi + n_clip_lo
    clip_pct = 100.0 * n_clip / n_valid if n_valid else 0.0
    handles = [
        Patch(facecolor="#1a7f37", alpha=0.45, edgecolor="none", label="N > 0.5  (covered more)"),
        Patch(facecolor="#c0392b", alpha=0.45, edgecolor="none", label="N < 0.5  (covered less)"),
        Line2D([0], [0], color="#7f8c8d", ls="--", lw=1.2, label=r"$q=d_p/d_n-1$ (twin axis)"),
        Line2D([0], [0], marker="x", color="#111111", lw=0, label="clipped |q|>100%"),
    ]
    axes[0, -1].legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.92)
    clip_txt = (
        f"clipped {n_clip}/{n_valid} ({clip_pct:.1f}%)  ·  "
        f"{n_clip_hi} above +100%,  {n_clip_lo} below -100%"
        if n_valid
        else "no valid minutes"
    )
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  "
        f"self-nearest distance  ·  {kind}  |  {map_tag}  |  {clip_txt}",
        fontsize=9.5,
    )
    print(f"  {kind}: {clip_txt}")
    if q_all:
        qcat = np.concatenate(q_all) if any(len(a) for a in q_all) else np.array([])
        if qcat.size:
            print(
                f"  q range [{np.nanmin(qcat):+.2f}, {np.nanmax(qcat):+.2f}]  "
                f"({int((qcat > 1).sum())} > +1, {int((qcat < -1).sum())} < -1)"
            )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, tab


def plot_self_nearest_distance_cumulative_pm600(
    bundle: dict,
    save_path: Path | None = None,
    *,
    half_range_m: float = 600.0,
):
    """Cumulative distance residual C(t)=Σ(d_p-d_n), mapped linearly onto [0, 1].

    Clip C to [-R, R] (default R=600 m), then
    ``N = (C_clip + R) / (2R)`` so **0 m → 0.5**. Twin axis is raw C in metres.
    Prints clip rate. No possession colour.
    """
    R = float(half_range_m)
    if R <= 0:
        raise ValueError("half_range_m must be positive")
    series = (
        ("d_resid_tm", "vs nearest-2 teammates", "#1f4e79"),
        ("d_resid_op", "vs nearest-2 opponents", "#b03a2e"),
    )
    ylab = r"$N$  (0.5 = 0 m residual)"

    tab = enrich_self_nearest_possession(bundle, self_nearest_minute_table(bundle))
    periods = (
        sorted(int(p) for p in tab["period"].dropna().unique()) if not tab.empty else [1, 2]
    )
    n = max(len(periods), 1)
    fig, axes = plt.subplots(
        2, n, figsize=(6.4 * n, 6.8), sharex="col", sharey=True, layout="constrained"
    )
    axes = np.asarray(axes).reshape(2, n)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_valid = 0
    n_clip_hi = 0
    n_clip_lo = 0
    c_all = []
    for col_i, p in enumerate(periods):
        g = (
            tab.loc[tab["period"] == int(p)].sort_values("t_centre_min")
            if not tab.empty
            else pd.DataFrame()
        )
        for row_i, (col, row_title, color) in enumerate(series):
            ax = axes[row_i, col_i]
            ax.axhline(0.5, color="#666666", ls=":", lw=1.1, zorder=2)
            ax.set_ylim(0.0, 1.0)
            if g.empty:
                ax.set_title(f"P{p}: no data")
                continue
            t = g["t_centre_min"].to_numpy(dtype=float)
            r = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(t) & np.isfinite(r)
            if not ok.any():
                ax.set_title(f"P{p}: no data")
                continue
            t, r = t[ok], r[ok]
            C = np.cumsum(r)
            c_all.append(C)
            n_valid += int(C.size)
            hi = C > R
            lo = C < -R
            n_clip_hi += int(hi.sum())
            n_clip_lo += int(lo.sum())
            N = np.clip((C + R) / (2.0 * R), 0.0, 1.0)

            ax.fill_between(
                t,
                0.5,
                N,
                where=N >= 0.5,
                color="#1a7f37",
                alpha=0.28,
                interpolate=True,
                zorder=1,
                linewidth=0,
            )
            ax.fill_between(
                t,
                0.5,
                N,
                where=N < 0.5,
                color="#c0392b",
                alpha=0.28,
                interpolate=True,
                zorder=1,
                linewidth=0,
            )
            ax.plot(t, N, color=color, lw=2.2, zorder=3)
            clipped = hi | lo
            if clipped.any():
                ax.scatter(
                    t[clipped],
                    N[clipped],
                    s=32,
                    marker="x",
                    c="#111111",
                    linewidths=1.15,
                    zorder=4,
                )
            ax2 = ax.twinx()
            ax2.plot(t, C, color="#7f8c8d", ls="--", lw=1.15, alpha=0.85, zorder=2)
            ax2.axhline(0.0, color="#b0b0b0", ls=":", lw=0.8)
            ax2.axhline(R, color="#b0b0b0", ls=":", lw=0.7)
            ax2.axhline(-R, color="#b0b0b0", ls=":", lw=0.7)
            ax2.tick_params(axis="y", labelsize=8, colors="#7f8c8d")
            if col_i == n - 1:
                ax2.set_ylabel("C(t) residual (m)", color="#7f8c8d", fontsize=8)
            ax.set_title(
                f"P{p}  ·  {row_title}"
                + (f"  ·  clipped {int(clipped.sum())}/{int(C.size)}" if clipped.any() else ""),
                fontsize=11,
            )
            ax.grid(True, alpha=0.25)
            end = float(N[-1])
            if abs(end - 0.5) >= 0.02:
                ax.text(
                    0.98,
                    0.08 if end < 0.5 else 0.92,
                    f"N={end:.2f}  C={C[-1]:+.0f} m",
                    transform=ax.transAxes,
                    va="bottom" if end < 0.5 else "top",
                    ha="right",
                    fontsize=8,
                    color="#c0392b" if end < 0.5 else "#1a7f37",
                    fontweight="bold",
                )
            if row_i == 1:
                ax.set_xlabel("period clock (min)")

    axes[0, 0].set_ylabel(ylab)
    axes[1, 0].set_ylabel(ylab)
    n_clip = n_clip_hi + n_clip_lo
    clip_pct = 100.0 * n_clip / n_valid if n_valid else 0.0
    handles = [
        Patch(facecolor="#1a7f37", alpha=0.45, edgecolor="none", label="N > 0.5  (ahead, C > 0)"),
        Patch(facecolor="#c0392b", alpha=0.45, edgecolor="none", label="N < 0.5  (behind, C < 0)"),
        Line2D([0], [0], color="#7f8c8d", ls="--", lw=1.2, label="C(t) metres (twin axis)"),
        Line2D([0], [0], marker="x", color="#111111", lw=0, label=f"clipped |C|>{R:g} m"),
    ]
    axes[0, -1].legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.92)
    clip_txt = (
        f"clipped {n_clip}/{n_valid} ({clip_pct:.1f}%)  ·  "
        f"{n_clip_hi} above +{R:g} m,  {n_clip_lo} below -{R:g} m"
        if n_valid
        else "no valid minutes"
    )
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  "
        f"cumulative distance residual  ·  "
        f"C clipped to [{-R:g}, {R:g}] m  ->  N=(C+{R:g})/{2 * R:g}  |  {clip_txt}",
        fontsize=9.5,
    )
    print(f"  cumulative C +/-{R:g} m -> [0,1]: {clip_txt}")
    if c_all:
        ccat = np.concatenate(c_all)
        print(f"  C range [{np.nanmin(ccat):+.1f}, {np.nanmax(ccat):+.1f}] m")
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, tab


def plot_self_nearest_residuals(
    bundle: dict, save_path: Path | None = None, period: int | None = None, metric: str = "speed"
):
    """Back-compat alias → progression plot (possession colour + rolling median)."""
    # ``period`` ignored: progression is always P1|P2 side by side.
    return plot_self_nearest_residual_progression(bundle, save_path=save_path, metric=metric)


# ---------------------------------------------------------------------------
# Polar kinematic fingerprint (one circle per period; match-wide scales)
# ---------------------------------------------------------------------------

POLAR_RING_COLORS = {
    "jerk": ("#6c3483", 0.50),
    "acc": ("#b03a2e", 0.50),
    "speed": ("#1f4e79", 0.50),
    "distance": ("#1a7f37", 0.50),
}
POLAR_RAW_MAX_POINTS = 4000  # standard frame-level downsample for drawing
POLAR_RAW_MAX_POINTS_FINE = 16000  # lighter downsample (full-profile comparison)
POLAR_METRIC_COLS = ("jerk", "acc", "speed", "distance")
POLAR_BAND_LABELS = {
    "jerk": "mean |jerk|",
    "acc": r"mean max($a_{\tan}$,0)",
    "speed": "mean speed",
    "distance": "window dist",
}
POLAR_YTICK_LABELS = ["mean |jerk|", r"mean max($a$,0)", "mean speed", "window dist"]
POLAR_COVERAGE_FRAC = 0.80  # keep windows with ≥ this fraction of expected frames


def _estimate_sample_hz(t: np.ndarray, default: float = 30.0) -> float:
    tt = np.asarray(t, dtype=float)
    tt = tt[np.isfinite(tt)]
    if len(tt) < 3:
        return float(default)
    dt = np.diff(np.sort(tt))
    dt = dt[dt > 0]
    if len(dt) == 0:
        return float(default)
    med = float(np.median(dt))
    if med <= 0:
        return float(default)
    return float(1.0 / med)


def kinematic_polar_window_table(
    df: pd.DataFrame,
    win_s: float | None,
    step_s: float | None = None,
    coverage_frac: float = POLAR_COVERAGE_FRAC,
) -> pd.DataFrame:
    """Aggregate mean |jerk| / mean max(a,0) / mean speed / windowed distance.

    ``win_s=None`` → one row per frame (distance = step_dist; acc = max(a_tan, 0)).
    Otherwise non-overlapping windows of length ``win_s`` (step defaults to ``win_s``).
    Window coverage uses the period's estimated frame rate and ``coverage_frac``.
    """
    d = df.copy().sort_values(["periodGameClockTime", "frameNum"]).reset_index(drop=True)
    for col in ("periodGameClockTime", "speed", "a_tan_use", "jerk_use", "step_dist", "acc_use"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    if "step_dist" not in d.columns:
        d = add_step_distance(d)

    t = d["periodGameClockTime"].to_numpy(dtype=float)
    speed = d["speed"].to_numpy(dtype=float) if "speed" in d.columns else np.full(len(d), np.nan)
    a_tan = d["a_tan_use"].to_numpy(dtype=float) if "a_tan_use" in d.columns else np.full(len(d), np.nan)
    jerk = d["jerk_use"].to_numpy(dtype=float) if "jerk_use" in d.columns else np.full(len(d), np.nan)
    step = d["step_dist"].to_numpy(dtype=float)

    if win_s is None or float(win_s) <= 0:
        # frame-level: zero-clipped positive acceleration contribution
        a_clip = np.where(np.isfinite(a_tan), np.maximum(a_tan, 0.0), np.nan)
        return pd.DataFrame(
            {
                "t_s": t,
                "jerk": np.abs(jerk),
                "acc": a_clip,
                "speed": speed,
                "distance": step,
                "n_frames": 1,
            }
        )

    win = float(win_s)
    step_w = float(step_s) if step_s is not None else win
    t_ok = t[np.isfinite(t)]
    if len(t_ok) == 0:
        return pd.DataFrame()
    t0 = float(np.nanmin(t_ok))
    t1 = float(np.nanmax(t_ok))
    fs = _estimate_sample_hz(t)
    min_frames = max(3, int(float(coverage_frac) * win * fs))
    starts = np.arange(t0, t1 - win + 1e-9, step_w)
    rows = []
    for s0 in starts:
        s1 = s0 + win
        m = np.isfinite(t) & (t >= s0) & (t < s1)
        if int(m.sum()) < min_frames:
            continue
        a = a_tan[m]
        a = a[np.isfinite(a)]
        a_clip = np.maximum(a, 0.0) if len(a) else a
        j = jerk[m]
        j = j[np.isfinite(j)]
        v = speed[m]
        v = v[np.isfinite(v)]
        dist = step[m]
        dist = dist[np.isfinite(dist)]
        rows.append(
            {
                "t_s": 0.5 * (s0 + s1),
                "jerk": float(np.mean(np.abs(j))) if len(j) else np.nan,
                "acc": float(np.mean(a_clip)) if len(a_clip) else np.nan,
                "speed": float(np.mean(v)) if len(v) else np.nan,
                "distance": float(np.sum(dist)) if len(dist) else np.nan,
                "n_frames": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def kinematic_polar_match_scales(bundle: dict, win_s: float | None) -> dict[str, float]:
    """Max of each polar metric over the **whole match** (all periods, same win_s)."""
    scales = {c: np.nan for c in POLAR_METRIC_COLS}
    for p in periods_in(bundle["tracking"]):
        tab = kinematic_polar_window_table(slice_period(bundle["tracking"], int(p)), win_s=win_s)
        if tab.empty:
            continue
        for c in POLAR_METRIC_COLS:
            if c not in tab.columns:
                continue
            mx = float(np.nanmax(tab[c].to_numpy(dtype=float))) if len(tab) else np.nan
            if not np.isfinite(mx):
                continue
            cur = scales[c]
            scales[c] = mx if (not np.isfinite(cur) or mx > cur) else cur
    return scales


def _norm01(series: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Normalize to [0,1] by ``scale`` (match-wide max) or by the series max if scale is None."""
    s = np.asarray(series, dtype=float)
    out = np.full(len(s), np.nan)
    ok = np.isfinite(s)
    if not ok.any():
        return out
    if scale is not None and np.isfinite(scale) and float(scale) > 0:
        mx = float(scale)
    else:
        mx = float(np.nanmax(s[ok]))
    if mx <= 0:
        out[ok] = 0.0
        return out
    out[ok] = np.clip(s[ok] / mx, 0.0, 1.0)
    return out


def plot_kinematic_polar_fingerprint(
    bundle: dict,
    win_s: float | None = 30.0,
    save_path: Path | None = None,
    period: int | None = None,
    max_raw_points: int = POLAR_RAW_MAX_POINTS,
):
    """Polar kinematic fingerprint — **one circle per period** (not whole match).

    Rings after **whole-match** max-normalization to [0,1] (same scale on P1 and P2):
      mean |jerk| 0→1, mean max(a_tan,0) 1→2, mean speed 2→3, windowed distance 3→4.
    Angle = progress through that period's clock span (stoppage included).
    ``win_s=None`` uses frame-level samples (downsampled for drawing via ``max_raw_points``).
    One figure with a separate polar circle per period when ``period=None``.
    """
    scales = kinematic_polar_match_scales(bundle, win_s)
    if period is None:
        pers = list(periods_in(bundle["tracking"]))
        n = max(len(pers), 1)
        fig = plt.figure(figsize=(6.2 * n, 6.4), layout="constrained")
        for i, p in enumerate(pers):
            ax = fig.add_subplot(1, n, i + 1, projection="polar")
            _draw_kinematic_polar_on_ax(
                ax,
                bundle,
                int(p),
                win_s=win_s,
                max_raw_points=max_raw_points,
                scales=scales,
                show_legend=(i == n - 1),
            )
        if win_s is None:
            win_lab = f"frame-level (≤{int(max_raw_points)} pts/period)"
        else:
            win_lab = f"{win_s:g}s windows"
        fig.suptitle(
            f"{bundle['nickname']}  |  {bundle['label']}  |  "
            f"polar kinematic fingerprint  |  {win_lab}  |  "
            f"one circle per period  |  match-wide max-norm",
            fontsize=11,
        )
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
        return fig

    fig = plt.figure(figsize=(7.2, 7.2), layout="constrained")
    ax = fig.add_subplot(111, projection="polar")
    _draw_kinematic_polar_on_ax(
        ax,
        bundle,
        int(period),
        win_s=win_s,
        max_raw_points=max_raw_points,
        scales=scales,
        show_legend=True,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def _draw_kinematic_polar_on_ax(
    ax,
    bundle: dict,
    period: int,
    *,
    win_s: float | None,
    max_raw_points: int,
    scales: dict[str, float],
    show_legend: bool,
):
    df = slice_period(bundle["tracking"], int(period))
    tab = kinematic_polar_window_table(df, win_s=win_s)
    if tab.empty:
        ax.set_title(f"P{period}: no data")
        return

    t = tab["t_s"].to_numpy(dtype=float)
    ok_t = np.isfinite(t)
    if not ok_t.any():
        ax.set_title(f"P{period}: no times")
        return
    t_lo, t_hi = float(np.nanmin(t[ok_t])), float(np.nanmax(t[ok_t]))
    span = max(t_hi - t_lo, 1e-6)
    theta = 2.0 * np.pi * (t - t_lo) / span

    idx = np.where(ok_t)[0]
    if win_s is None and len(idx) > max_raw_points:
        step = int(np.ceil(len(idx) / max_raw_points))
        tab = tab.iloc[idx[::step]].reset_index(drop=True)
        t = tab["t_s"].to_numpy(dtype=float)
        theta = 2.0 * np.pi * (t - t_lo) / span

    bands = [
        ("jerk", 0.0, POLAR_RING_COLORS["jerk"]),
        ("acc", 1.0, POLAR_RING_COLORS["acc"]),
        ("speed", 2.0, POLAR_RING_COLORS["speed"]),
        ("distance", 3.0, POLAR_RING_COLORS["distance"]),
    ]
    # Close the polygon for fill aesthetics; radial marker marks temporal discontinuity.
    theta_c = np.concatenate([theta, theta[:1]])
    for name, r0, (color, alpha) in bands:
        y = _norm01(tab[name].to_numpy(dtype=float), scale=scales.get(name))
        r = r0 + y
        r_c = np.concatenate([r, r[:1]])
        ax.fill_between(
            theta_c,
            np.full_like(r_c, r0),
            np.where(np.isfinite(r_c), r_c, r0),
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            theta_c,
            np.where(np.isfinite(r_c), r_c, np.nan),
            color=color,
            lw=1.3,
            zorder=3,
            label=POLAR_BAND_LABELS.get(name, name),
        )

    for r_grid in (0, 1, 2, 3, 4):
        ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, r_grid), color="#bbbbbb", lw=0.7, zorder=1)
    # Period start/end discontinuity (θ wrap is not temporal continuity).
    ax.plot([0.0, 0.0], [0.0, 4.05], color="#222222", lw=1.4, zorder=4)
    ax.set_ylim(0, 4.05)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(POLAR_YTICK_LABELS, fontsize=7)
    ax.set_xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    ax.set_xticklabels(
        [
            f"{t_lo/60:.0f}'",
            f"{(t_lo + 0.25 * span)/60:.0f}'",
            f"{(t_lo + 0.5 * span)/60:.0f}'",
            f"{(t_lo + 0.75 * span)/60:.0f}'",
        ],
        fontsize=8,
    )
    if win_s is None:
        win_lab = f"frame ≤{int(max_raw_points)}pts"
    else:
        win_lab = f"{win_s:g}s"
    ax.set_title(f"P{period}  |  {win_lab}  |  n={len(tab)}  |  match-norm", fontsize=10)
    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.05), fontsize=7, framealpha=0.9)


# ---------------------------------------------------------------------------
# Relative high-intensity / sprint events (player-specific ±1 SD hysteresis)
# Notebook-first; not wired into CLI until approved.
# ---------------------------------------------------------------------------

SPRINT_Y_METRICS = {
    "sum_speed": ("sum_speed", "sum of speed samples"),
    "mean_speed": ("mean_speed", "mean speed (m/s)"),
    "peak_speed": ("peak_speed", "sprint peak speed (m/s)"),
    "duration_s": ("duration_s", "duration (s)"),
    "distance_m": ("distance_m", "sprint distance (m)"),
    "max_acc": ("max_acc", r"max $a_{\tan}$ (m/s$^2$)"),
    "mean_acc": ("mean_acc", r"mean positive $a_{\tan}$ (m/s$^2$)"),
    "mean_abs_acc": ("mean_abs_acc", r"mean $|a_{\tan}|$ (m/s$^2$)"),
    "max_jerk": ("max_jerk", r"max $j_{\tan}$"),
    "mean_abs_jerk": ("mean_abs_jerk", r"mean $|j_{\tan}|$"),
}

SPRINT_TIMELINE_METRICS = {
    "mean_speed": ("mean_speed", "event mean speed (m/s)"),
    "peak_speed": ("peak_speed", "event peak speed (m/s)"),
    "duration_s": ("duration_s", "event duration (s)"),
}


def extract_vendor_speed(
    tracking_path: Path, side: str, shirt_number: int | str
) -> pd.DataFrame:
    """Pull PFF-provided ``speed`` from raw (unsmoothed) player points in tracking JSONL."""
    players_key = "homePlayers" if side == "home" else "awayPlayers"
    jersey = str(shirt_number)
    rows = []
    for rec in ps.iter_jsonl(tracking_path):
        hit = None
        for p in rec.get(players_key) or []:
            if str(p.get("jerseyNum")) == jersey:
                hit = p
                break
        if hit is None:
            continue
        rows.append(
            {
                "frameNum": rec.get("frameNum"),
                "period": rec.get("period"),
                "speed_pff": hit.get("speed"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["frameNum"] = pd.to_numeric(out["frameNum"], errors="coerce")
    out["period"] = pd.to_numeric(out["period"], errors="coerce")
    out["speed_pff"] = pd.to_numeric(out["speed_pff"], errors="coerce")
    return out


def _bundle_tracking_path(bundle: dict) -> Path:
    mid = str(bundle["match_id"])
    if bundle.get("match_dir"):
        return Path(bundle["match_dir"]) / f"{mid}_tracking.jsonl"
    when = bundle.get("when")
    if when in MATCHES:
        return MATCHES[when]["dir"] / f"{mid}_tracking.jsonl"
    if mid in VENDOR_SPEED_MATCHES:
        return VENDOR_SPEED_MATCHES[mid]["dir"] / f"{mid}_tracking.jsonl"
    raise KeyError(f"cannot resolve tracking path for match_id={mid!r}")


def attach_vendor_speed(bundle: dict, force: bool = False) -> dict:
    """Attach ``speed_pff`` (vendor) onto ``bundle['tracking']``; keep stride-12 ``speed``."""
    df = bundle["tracking"]
    if (
        not force
        and "speed_pff" in df.columns
        and pd.to_numeric(df["speed_pff"], errors="coerce").notna().any()
    ):
        return bundle
    path = _bundle_tracking_path(bundle)
    slim = extract_vendor_speed(path, bundle["side"], bundle["shirt"])
    tracking = df.drop(columns=["speed_pff"], errors="ignore").copy()
    if slim.empty:
        tracking["speed_pff"] = np.nan
    else:
        tracking = tracking.merge(slim, on=["frameNum", "period"], how="left")
    bundle = dict(bundle)
    bundle["tracking"] = tracking
    return bundle


def load_player_kinematics(
    match_id: str,
    player_id: str | int,
    *,
    match_dir: Path | None = None,
    label: str | None = None,
    attach_vendor: bool = True,
    prefer_smoothed: bool = True,
) -> dict[str, Any]:
    """Load focal tracking + stride-12 kinematics (no possession). For vendor-speed compares."""
    mid = str(match_id)
    if match_dir is None:
        if mid in VENDOR_SPEED_MATCHES:
            match_dir = VENDOR_SPEED_MATCHES[mid]["dir"]
            label = label or VENDOR_SPEED_MATCHES[mid]["label"]
        elif mid in {s["match_id"] for s in MATCHES.values()}:
            spec = next(s for s in MATCHES.values() if s["match_id"] == mid)
            match_dir = spec["dir"]
            label = label or spec["label"]
        else:
            raise ValueError(f"unknown match_id {mid}; pass match_dir=")
    match_dir = Path(match_dir)
    label = label or mid

    roster = ps.load_roster(match_dir / f"{mid}_roster.json", mid)
    meta = ps.load_json(match_dir / f"{mid}_metadata.json")
    row = roster.loc[roster["player_id"].astype(str) == str(player_id)]
    if row.empty:
        raise ValueError(f"player_id {player_id} not in roster for match {mid}")
    row = row.iloc[0]
    team_id = str(row["team_id"])
    side = ps.side_for_team(meta, team_id)
    shirt = int(row["shirt_number"])
    nickname = str(row["nickname"])

    path = match_dir / f"{mid}_tracking.jsonl"
    tracking = ps.extract_focal_tracking(path, side, shirt, prefer_smoothed_player=prefer_smoothed)
    if tracking.empty:
        raise ValueError(
            f"no tracking frames for {nickname} (id={player_id}, {side} #{shirt}) in match {mid}"
        )
    for col in ("frameNum", "period", "video_time_s", "periodGameClockTime", "x", "y"):
        if col in tracking.columns:
            tracking[col] = pd.to_numeric(tracking[col], errors="coerce")
    tracking = add_stride_kinematics(tracking)
    tracking = add_step_distance(tracking)

    bundle = {
        "when": None,
        "match_id": mid,
        "match_dir": match_dir,
        "label": label,
        "player_id": str(player_id),
        "nickname": nickname,
        "team_id": team_id,
        "team_name": str(row["team"]),
        "opp_team_name": "",
        "side": side,
        "shirt": shirt,
        "tracking": tracking,
        "meta": meta,
        "roster": roster,
    }
    if attach_vendor:
        bundle = attach_vendor_speed(bundle, force=True)
    return bundle


def plot_pff_vs_stride_speed(
    bundle: dict,
    save_path: Path | None = None,
    period: int | None = None,
    max_points: int = 6000,
):
    """Overlay vendor ``speed_pff`` and stride-12 ``speed``. ``period=None`` → P1|P2 side by side."""

    def _draw(ax, p):
        df = slice_period(bundle["tracking"], int(p)).sort_values(["periodGameClockTime", "frameNum"])
        t = period_clock_minutes(df).to_numpy(dtype=float)
        for col, lab, color in (
            ("speed_pff", "PFF vendor", "#737373"),
            ("speed", "stride-12 derivative", "#1f4e79"),
        ):
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(t) & np.isfinite(y)
            if not ok.any():
                continue
            idx = np.where(ok)[0]
            if len(idx) > max_points:
                idx = idx[:: int(np.ceil(len(idx) / max_points))]
            ax.plot(t[idx], y[idx], color=color, lw=0.7, alpha=0.9, label=lab)
        ax.set_xlabel("period clock (min)")
        ax.set_ylabel("speed (m/s)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"P{p}", fontsize=10)
        ax.grid(True, alpha=0.25)

    if period is not None:
        fig, ax = plt.subplots(figsize=(12.5, 4.0), layout="constrained")
        _draw(ax, int(period))
        ax.set_title(
            f"{bundle['nickname']}  |  {bundle['label']}  |  P{period}  |  PFF vs stride-12 speed",
            fontsize=10,
        )
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
        return fig

    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.0), sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, pers):
        _draw(ax, int(p))
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  PFF vs stride-12 speed",
        fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def _time_diff_series(y: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Forward difference dy/dt aligned to sample i (value at i from i-1→i)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t_s, dtype=float)
    out = np.full(len(y), np.nan)
    if len(y) < 2:
        return out
    dt = np.diff(t)
    dy = np.diff(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = np.where(dt > 0, dy / dt, np.nan)
    out[1:] = mid
    if len(out) >= 3:
        out[0] = out[1]
    return out


def _mean_positive(arr: np.ndarray) -> float:
    """Mean of strictly positive finite samples (acceleration only, not deceleration)."""
    a = np.asarray(arr, dtype=float)
    pos = a[np.isfinite(a) & (a > 0)]
    return float(np.mean(pos)) if len(pos) else np.nan


def _sprint_thr_from_events(events: pd.DataFrame) -> dict:
    thr = {
        "mu_v": float(events["mu_v"].iloc[0]) if len(events) and "mu_v" in events.columns else np.nan,
        "sigma_v": float(events["sigma_v"].iloc[0]) if len(events) and "sigma_v" in events.columns else np.nan,
        "t_high": float(events["t_high"].iloc[0]) if len(events) and "t_high" in events.columns else np.nan,
        "t_low": float(events["t_low"].iloc[0]) if len(events) and "t_low" in events.columns else np.nan,
    }
    if len(events) and "k_low" in events.columns and pd.notna(events["k_low"].iloc[0]):
        thr["k_low"] = float(events["k_low"].iloc[0])
        thr["k_high"] = float(events["k_high"].iloc[0]) if "k_high" in events.columns else 1.0
    if len(events) and "threshold_mps" in events.columns and pd.notna(events["threshold_mps"].iloc[0]):
        thr["method"] = "absolute_threshold"
        thr["threshold_mps"] = float(events["threshold_mps"].iloc[0])
    return thr


def compute_speed_thresholds(
    speed: np.ndarray,
    *,
    k_high: float = 1.0,
    k_low: float = 1.0,
) -> dict[str, float]:
    """``T_high = μ + k_high·σ``, ``T_low = max(0, μ − k_low·σ)``.

    ``k_low`` is subtracted from the mean, so a *smaller* (or negative) value
    *raises* the start/end floor: ``k_low=0.5`` → ``μ−0.5σ``; ``k_low=-1.5``
    → ``μ+1.5σ``. Need ``T_high > T_low`` (e.g. ``k_high=2`` with ``k_low=-1.5``).
    """
    s = np.asarray(speed, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return {
            "mu_v": np.nan,
            "sigma_v": np.nan,
            "t_high": np.nan,
            "t_low": np.nan,
            "k_high": float(k_high),
            "k_low": float(k_low),
            "n": int(len(s)),
        }
    mu = float(np.mean(s))
    sig = float(np.std(s, ddof=0))
    return {
        "mu_v": mu,
        "sigma_v": sig,
        "t_high": mu + float(k_high) * sig,
        "t_low": max(0.0, mu - float(k_low) * sig),
        "k_high": float(k_high),
        "k_low": float(k_low),
        "n": int(len(s)),
    }


def detect_relative_sprint_events(
    bundle: dict,
    speed_col: str = "speed",
    source_label: str | None = None,
    *,
    k_high: float = 1.0,
    k_low: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Player-specific hysteresis HI events on a chosen speed column.

    Default ±1 SD: ``T_high=μ+σ``, ``T_low=μ−σ``. Raising the start/end floor
    splits neighbouring peaks: ``k_low=0.5`` → ``T_low=μ−0.5σ``; ``k_low=-1.5``
    → ``T_low=μ+1.5σ`` (above the mean). Requires ``T_high > T_low``.

    ``speed_col``: ``speed`` (stride-12 from position) or ``speed_pff`` (vendor).
    Acceleration/jerk for characterization: if ``speed`` use ``a_tan_use`` and
    d(a_tan)/dt; if ``speed_pff`` use d(speed_pff)/dt and d²(speed_pff)/dt².
    """
    if source_label is None:
        source_label = "stride12" if speed_col == "speed" else speed_col

    df = bundle["tracking"].copy().sort_values(["period", "video_time_s", "frameNum"]).reset_index(drop=True)
    if speed_col not in df.columns:
        raise KeyError(f"{speed_col!r} missing on tracking — call attach_vendor_speed() for speed_pff")
    for col in (speed_col, "a_tan_use", "step_dist", "video_time_s", "periodGameClockTime", "period"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "step_dist" not in df.columns:
        df = add_step_distance(df)

    thr = compute_speed_thresholds(
        df[speed_col].to_numpy(dtype=float), k_high=k_high, k_low=k_low
    )
    thr["speed_col"] = speed_col
    thr["source_label"] = source_label
    t_low, t_high = thr["t_low"], thr["t_high"]
    if not (np.isfinite(t_low) and np.isfinite(t_high) and t_high > t_low):
        return pd.DataFrame(), thr

    rows = []
    event_id = 0
    for period, g in df.groupby("period", sort=True):
        g = g.reset_index(drop=True)
        speed = g[speed_col].to_numpy(dtype=float)
        t_s = g["video_time_s"].to_numpy(dtype=float)
        clock_s = g["periodGameClockTime"].to_numpy(dtype=float)
        step = g["step_dist"].to_numpy(dtype=float)
        if speed_col == "speed" and "a_tan_use" in g.columns:
            a_tan = g["a_tan_use"].to_numpy(dtype=float)
        else:
            a_tan = _time_diff_series(speed, t_s)
        j_tan = _time_diff_series(a_tan, t_s)

        n = len(g)
        i = 0
        while i < n - 1:
            if not (np.isfinite(speed[i]) and np.isfinite(speed[i + 1])):
                i += 1
                continue
            if not (speed[i] < t_low <= speed[i + 1]):
                i += 1
                continue
            start_i = i + 1
            reached_high = False
            peak_i = start_i
            peak_v = speed[start_i] if np.isfinite(speed[start_i]) else -np.inf
            j = start_i
            end_i = None
            while j < n - 1:
                v0, v1 = speed[j], speed[j + 1]
                if np.isfinite(v1) and v1 > peak_v:
                    peak_v = float(v1)
                    peak_i = j + 1
                if np.isfinite(v1) and v1 >= t_high:
                    reached_high = True
                if np.isfinite(v0) and np.isfinite(v1) and v0 >= t_low > v1:
                    end_i = j
                    break
                j += 1
            if end_i is None:
                if reached_high and np.isfinite(speed[-1]) and speed[-1] >= t_low:
                    end_i = n - 1
                else:
                    i = start_i
                    continue
            if not reached_high or end_i < start_i:
                i = max(end_i, start_i) + 1
                continue

            sl = slice(start_i, end_i + 1)
            vs = speed[sl]
            ats = a_tan[sl]
            js = j_tan[sl]
            steps = step[sl]
            ok_v = np.isfinite(vs)
            if int(ok_v.sum()) < 3:
                i = end_i + 1
                continue
            if not (np.isfinite(t_s[start_i]) and np.isfinite(t_s[end_i])):
                i = end_i + 1
                continue
            if peak_i < start_i or peak_i > end_i:
                peak_i = start_i + int(np.nanargmax(vs))

            event_id += 1
            rows.append(
                {
                    "event_id": event_id,
                    "period": int(period),
                    "speed_source": source_label,
                    "start_time_s": float(t_s[start_i]),
                    "peak_time_s": float(t_s[peak_i]),
                    "end_time_s": float(t_s[end_i]),
                    "start_clock_min": float(clock_s[start_i]) / 60.0,
                    "peak_clock_min": float(clock_s[peak_i]) / 60.0,
                    "end_clock_min": float(clock_s[end_i]) / 60.0,
                    "duration_s": float(t_s[end_i] - t_s[start_i]),
                    "sum_speed": float(np.nansum(vs)),
                    "mean_speed": float(np.nanmean(vs)),
                    "peak_speed": float(np.nanmax(vs)),
                    "distance_m": float(np.nansum(steps[np.isfinite(steps)])),
                    "max_acc": float(np.nanmax(ats)) if np.isfinite(ats).any() else np.nan,
                    "min_acc": float(np.nanmin(ats)) if np.isfinite(ats).any() else np.nan,
                    "mean_acc": _mean_positive(ats),
                    "mean_abs_acc": float(np.nanmean(np.abs(ats))) if np.isfinite(ats).any() else np.nan,
                    "max_jerk": float(np.nanmax(js)) if np.isfinite(js).any() else np.nan,
                    "min_jerk": float(np.nanmin(js)) if np.isfinite(js).any() else np.nan,
                    "mean_jerk": float(np.nanmean(js)) if np.isfinite(js).any() else np.nan,
                    "mean_abs_jerk": float(np.nanmean(np.abs(js))) if np.isfinite(js).any() else np.nan,
                    "n_samples": int(ok_v.sum()),
                    "mu_v": thr["mu_v"],
                    "sigma_v": thr["sigma_v"],
                    "t_high": t_high,
                    "t_low": t_low,
                    "k_high": thr["k_high"],
                    "k_low": thr["k_low"],
                }
            )
            i = end_i + 1

    events = pd.DataFrame(rows)
    if events.empty:
        return events, thr
    events = events.sort_values(["start_time_s", "event_id"]).reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    prev_end = events["end_time_s"].shift(1)
    events["recovery_before_s"] = events["start_time_s"] - prev_end
    events.loc[events["recovery_before_s"] < 0, "recovery_before_s"] = np.nan
    return events, thr


def _sprint_event_metrics_from_slice(
    *,
    period: int,
    speed: np.ndarray,
    t_s: np.ndarray,
    clock_s: np.ndarray,
    step: np.ndarray,
    a_tan: np.ndarray,
    j_tan: np.ndarray,
    start_i: int,
    end_i: int,
    source_label: str,
    thr: dict,
) -> dict | None:
    """HI-event metrics on ``start_i..end_i`` (inclusive), same columns as the detector."""
    if end_i < start_i:
        return None
    sl = slice(start_i, end_i + 1)
    vs = speed[sl]
    ats = a_tan[sl]
    js = j_tan[sl]
    steps = step[sl]
    ok_v = np.isfinite(vs)
    if int(ok_v.sum()) < 3:
        return None
    if not (np.isfinite(t_s[start_i]) and np.isfinite(t_s[end_i])):
        return None
    peak_i = start_i + int(np.nanargmax(vs))
    return {
        "period": int(period),
        "speed_source": source_label,
        "start_time_s": float(t_s[start_i]),
        "peak_time_s": float(t_s[peak_i]),
        "end_time_s": float(t_s[end_i]),
        "start_clock_min": float(clock_s[start_i]) / 60.0,
        "peak_clock_min": float(clock_s[peak_i]) / 60.0,
        "end_clock_min": float(clock_s[end_i]) / 60.0,
        "duration_s": float(t_s[end_i] - t_s[start_i]),
        "sum_speed": float(np.nansum(vs)),
        "mean_speed": float(np.nanmean(vs)),
        "peak_speed": float(np.nanmax(vs)),
        "distance_m": float(np.nansum(steps[np.isfinite(steps)])),
        "max_acc": float(np.nanmax(ats)) if np.isfinite(ats).any() else np.nan,
        "min_acc": float(np.nanmin(ats)) if np.isfinite(ats).any() else np.nan,
        "mean_acc": _mean_positive(ats),
        "mean_abs_acc": float(np.nanmean(np.abs(ats))) if np.isfinite(ats).any() else np.nan,
        "max_jerk": float(np.nanmax(js)) if np.isfinite(js).any() else np.nan,
        "min_jerk": float(np.nanmin(js)) if np.isfinite(js).any() else np.nan,
        "mean_jerk": float(np.nanmean(js)) if np.isfinite(js).any() else np.nan,
        "mean_abs_jerk": float(np.nanmean(np.abs(js))) if np.isfinite(js).any() else np.nan,
        "n_samples": int(ok_v.sum()),
        "mu_v": thr.get("mu_v", np.nan),
        "sigma_v": thr.get("sigma_v", np.nan),
        "t_high": thr.get("t_high", np.nan),
        "t_low": thr.get("t_low", np.nan),
        "k_high": thr.get("k_high", np.nan),
        "k_low": thr.get("k_low", np.nan),
    }


def merge_close_sprint_events(
    bundle: dict,
    events: pd.DataFrame,
    min_gap_s: float = 1.0,
    speed_col: str = "speed",
    source_label: str | None = None,
) -> pd.DataFrame:
    """Merge consecutive same-period events whose gap is ``< min_gap_s``.

    A cluster's metrics are recomputed over the union window (first start
    through last end), so a sub-1 s dip is inside the effort rather than a
    recovery. Periods are not merged. ``recovery_before_s`` is rebuilt after.
    """
    if events is None or events.empty:
        out = events.copy() if events is not None else pd.DataFrame()
        if len(out):
            out["n_atoms"] = 1
            out["merge_gap_s"] = float(min_gap_s)
        return out

    if source_label is None:
        if "speed_source" in events.columns and pd.notna(events["speed_source"].iloc[0]):
            source_label = str(events["speed_source"].iloc[0])
        else:
            source_label = "stride12" if speed_col == "speed" else speed_col

    ev = events.sort_values(["start_time_s", "event_id"]).reset_index(drop=True)
    clusters: list[list[int]] = [[0]]
    for i in range(1, len(ev)):
        gap = float(ev.loc[i, "start_time_s"] - ev.loc[i - 1, "end_time_s"])
        same_period = int(ev.loc[i, "period"]) == int(ev.loc[i - 1, "period"])
        if same_period and np.isfinite(gap) and 0 <= gap < float(min_gap_s):
            clusters[-1].append(i)
        else:
            clusters.append([i])

    df = bundle["tracking"].copy().sort_values(["period", "video_time_s", "frameNum"]).reset_index(drop=True)
    if speed_col not in df.columns:
        raise KeyError(f"{speed_col!r} missing on tracking")
    for col in (speed_col, "a_tan_use", "step_dist", "video_time_s", "periodGameClockTime", "period"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "step_dist" not in df.columns:
        df = add_step_distance(df)

    period_cache: dict[int, dict] = {}

    def _period_arrays(period: int) -> dict:
        if period in period_cache:
            return period_cache[period]
        g = df.loc[df["period"] == int(period)].reset_index(drop=True)
        speed = g[speed_col].to_numpy(dtype=float)
        t_s = g["video_time_s"].to_numpy(dtype=float)
        clock_s = g["periodGameClockTime"].to_numpy(dtype=float)
        step = g["step_dist"].to_numpy(dtype=float)
        if speed_col == "speed" and "a_tan_use" in g.columns:
            a_tan = g["a_tan_use"].to_numpy(dtype=float)
        else:
            a_tan = _time_diff_series(speed, t_s)
        period_cache[period] = {
            "speed": speed,
            "t_s": t_s,
            "clock_s": clock_s,
            "step": step,
            "a_tan": a_tan,
            "j_tan": _time_diff_series(a_tan, t_s),
        }
        return period_cache[period]

    rows = []
    thr0 = _sprint_thr_from_events(ev)
    for idxs in clusters:
        sub = ev.loc[idxs]
        n_atoms = len(idxs)
        period = int(sub["period"].iloc[0])
        start_t = float(sub["start_time_s"].iloc[0])
        end_t = float(sub["end_time_s"].iloc[-1])
        row = None
        if n_atoms > 1:
            arr = _period_arrays(period)
            t_s = arr["t_s"]
            ok_t = np.isfinite(t_s)
            if ok_t.any():
                start_i = int(np.argmin(np.abs(np.where(ok_t, t_s, np.inf) - start_t)))
                end_i = int(np.argmin(np.abs(np.where(ok_t, t_s, np.inf) - end_t)))
                row = _sprint_event_metrics_from_slice(
                    period=period,
                    speed=arr["speed"],
                    t_s=arr["t_s"],
                    clock_s=arr["clock_s"],
                    step=arr["step"],
                    a_tan=arr["a_tan"],
                    j_tan=arr["j_tan"],
                    start_i=start_i,
                    end_i=end_i,
                    source_label=source_label,
                    thr=thr0,
                )
        if row is None:
            row = sub.iloc[0].to_dict()
            if n_atoms > 1:
                row["end_time_s"] = end_t
                row["end_clock_min"] = float(sub["end_clock_min"].iloc[-1])
                row["duration_s"] = float(end_t - start_t)
                row["peak_speed"] = float(sub["peak_speed"].max())
                peak_i = int(sub["peak_speed"].values.argmax())
                row["peak_time_s"] = float(sub["peak_time_s"].iloc[peak_i])
                row["peak_clock_min"] = float(sub["peak_clock_min"].iloc[peak_i])
                if "distance_m" in sub.columns:
                    row["distance_m"] = float(sub["distance_m"].sum())
                if "sum_speed" in sub.columns:
                    row["sum_speed"] = float(sub["sum_speed"].sum())
                if "n_samples" in sub.columns:
                    row["n_samples"] = int(sub["n_samples"].sum())
                if "max_acc" in sub.columns:
                    row["max_acc"] = float(sub["max_acc"].max())
                if "max_jerk" in sub.columns:
                    row["max_jerk"] = float(sub["max_jerk"].max())
        row["n_atoms"] = int(n_atoms)
        row["merge_gap_s"] = float(min_gap_s)
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["start_time_s"]).reset_index(drop=True)
    out["event_id"] = np.arange(1, len(out) + 1)
    prev_end = out["end_time_s"].shift(1)
    out["recovery_before_s"] = out["start_time_s"] - prev_end
    out.loc[out["recovery_before_s"] < 0, "recovery_before_s"] = np.nan
    return out


def detect_threshold_sprint_events(
    bundle: dict,
    threshold_mps: float = 7.0,
    speed_col: str = "speed",
    source_label: str | None = None,
    min_samples: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Sprint = contiguous run where ``speed_col > threshold_mps`` (default 7 m/s).

    Same event metrics / columns as ``detect_relative_sprint_events`` so timeline
    and recovery scatter plotters work unchanged. ``t_high`` / ``t_low`` are both
    set to the absolute threshold for plot reference lines.
    """
    if source_label is None:
        source_label = "stride12" if speed_col == "speed" else speed_col

    df = bundle["tracking"].copy().sort_values(["period", "video_time_s", "frameNum"]).reset_index(drop=True)
    if speed_col not in df.columns:
        raise KeyError(f"{speed_col!r} missing on tracking — call attach_vendor_speed() for speed_pff")
    for col in (speed_col, "a_tan_use", "step_dist", "video_time_s", "periodGameClockTime", "period"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "step_dist" not in df.columns:
        df = add_step_distance(df)

    thr = {
        "mu_v": np.nan,
        "sigma_v": np.nan,
        "t_high": float(threshold_mps),
        "t_low": float(threshold_mps),
        "threshold_mps": float(threshold_mps),
        "method": "absolute_threshold",
        "speed_col": speed_col,
        "source_label": source_label,
        "n": int(pd.to_numeric(df[speed_col], errors="coerce").notna().sum()),
    }

    rows = []
    event_id = 0
    thr_v = float(threshold_mps)
    for period, g in df.groupby("period", sort=True):
        g = g.reset_index(drop=True)
        speed = g[speed_col].to_numpy(dtype=float)
        t_s = g["video_time_s"].to_numpy(dtype=float)
        clock_s = g["periodGameClockTime"].to_numpy(dtype=float)
        step = g["step_dist"].to_numpy(dtype=float)
        if speed_col == "speed" and "a_tan_use" in g.columns:
            a_tan = g["a_tan_use"].to_numpy(dtype=float)
        else:
            a_tan = _time_diff_series(speed, t_s)
        j_tan = _time_diff_series(a_tan, t_s)

        above = np.isfinite(speed) & (speed > thr_v)
        n = len(g)
        i = 0
        while i < n:
            if not above[i]:
                i += 1
                continue
            start_i = i
            while i + 1 < n and above[i + 1]:
                i += 1
            end_i = i
            sl = slice(start_i, end_i + 1)
            vs = speed[sl]
            ats = a_tan[sl]
            js = j_tan[sl]
            steps = step[sl]
            ok_v = np.isfinite(vs)
            if int(ok_v.sum()) < min_samples:
                i = end_i + 1
                continue
            if not (np.isfinite(t_s[start_i]) and np.isfinite(t_s[end_i])):
                i = end_i + 1
                continue
            peak_i = start_i + int(np.nanargmax(vs))

            event_id += 1
            rows.append(
                {
                    "event_id": event_id,
                    "period": int(period),
                    "speed_source": source_label,
                    "start_time_s": float(t_s[start_i]),
                    "peak_time_s": float(t_s[peak_i]),
                    "end_time_s": float(t_s[end_i]),
                    "start_clock_min": float(clock_s[start_i]) / 60.0,
                    "peak_clock_min": float(clock_s[peak_i]) / 60.0,
                    "end_clock_min": float(clock_s[end_i]) / 60.0,
                    "duration_s": float(t_s[end_i] - t_s[start_i]),
                    "sum_speed": float(np.nansum(vs)),
                    "mean_speed": float(np.nanmean(vs)),
                    "peak_speed": float(np.nanmax(vs)),
                    "distance_m": float(np.nansum(steps[np.isfinite(steps)])),
                    "max_acc": float(np.nanmax(ats)) if np.isfinite(ats).any() else np.nan,
                    "min_acc": float(np.nanmin(ats)) if np.isfinite(ats).any() else np.nan,
                    "mean_acc": _mean_positive(ats),
                    "mean_abs_acc": float(np.nanmean(np.abs(ats))) if np.isfinite(ats).any() else np.nan,
                    "max_jerk": float(np.nanmax(js)) if np.isfinite(js).any() else np.nan,
                    "min_jerk": float(np.nanmin(js)) if np.isfinite(js).any() else np.nan,
                    "mean_jerk": float(np.nanmean(js)) if np.isfinite(js).any() else np.nan,
                    "mean_abs_jerk": float(np.nanmean(np.abs(js))) if np.isfinite(js).any() else np.nan,
                    "n_samples": int(ok_v.sum()),
                    "mu_v": thr["mu_v"],
                    "sigma_v": thr["sigma_v"],
                    "t_high": thr_v,
                    "t_low": thr_v,
                    "threshold_mps": thr_v,
                }
            )
            i = end_i + 1

    events = pd.DataFrame(rows)
    if events.empty:
        return events, thr
    events = events.sort_values(["start_time_s", "event_id"]).reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    prev_end = events["end_time_s"].shift(1)
    events["recovery_before_s"] = events["start_time_s"] - prev_end
    events.loc[events["recovery_before_s"] < 0, "recovery_before_s"] = np.nan
    return events, thr


def _draw_speed_profile_on_ax(
    ax,
    bundle: dict,
    events: pd.DataFrame,
    thr: dict,
    period: int,
    speed_col: str,
    max_points: int,
):
    src = thr.get("source_label", speed_col)
    df = slice_period(bundle["tracking"], int(period)).copy()
    df = df.sort_values(["periodGameClockTime", "frameNum"])
    t_min = period_clock_minutes(df).to_numpy(dtype=float)
    spd = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t_min) & np.isfinite(spd)
    if ok.any():
        idx = np.where(ok)[0]
        if len(idx) > max_points:
            idx = idx[:: int(np.ceil(len(idx) / max_points))]
        ax.plot(t_min[idx], spd[idx], color="#34495e", lw=0.55, alpha=0.9, label=f"speed ({src})")

    ev = events.loc[events["period"] == int(period)] if events is not None and len(events) else pd.DataFrame()
    labeled = False
    for _, r in ev.iterrows():
        ax.axvspan(
            float(r["start_clock_min"]),
            float(r["end_clock_min"]),
            color="#e74c3c",
            alpha=0.18,
            label="HI event" if not labeled else None,
            lw=0,
        )
        labeled = True
        ax.plot(float(r["peak_clock_min"]), float(r["peak_speed"]), "o", color="#c0392b", ms=3.5, zorder=3)

    if thr:
        th = thr.get("t_high", np.nan)
        mu = thr.get("mu_v", np.nan)
        tl = thr.get("t_low", np.nan)
        if np.isfinite(th):
            ax.axhline(th, color="#c0392b", ls="--", lw=1.2, label=f"T_high = {th:.2f}")
        if np.isfinite(mu):
            ax.axhline(mu, color="#7f8c8d", ls=":", lw=1.1, label=f"μ = {mu:.2f}")
        if np.isfinite(tl) and thr.get("method") != "absolute_threshold":
            ax.axhline(tl, color="#2980b9", ls="--", lw=1.2, label=f"T_low = {tl:.2f}")

    ax.set_xlabel("period clock (min)")
    ax.set_ylabel("speed (m/s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_title(f"P{period}  |  n_events={len(ev)}", fontsize=10)
    ax.grid(True, alpha=0.25)


def plot_speed_profile_with_events(
    bundle: dict,
    events: pd.DataFrame | None = None,
    thr: dict | None = None,
    save_path: Path | None = None,
    period: int | None = None,
    max_points: int = 8000,
    speed_col: str = "speed",
):
    """Overlay thresholds + HI intervals on speed. ``period=None`` → P1|P2 side by side."""
    if events is None or thr is None:
        events, thr = detect_relative_sprint_events(bundle, speed_col=speed_col)
    src = thr.get("source_label", speed_col)

    if period is not None:
        fig, ax = plt.subplots(figsize=(12.5, 4.2), layout="constrained")
        _draw_speed_profile_on_ax(ax, bundle, events, thr, int(period), speed_col, max_points)
        ax.set_title(
            f"{bundle['nickname']}  |  {bundle['label']}  |  P{period}  |  "
            f"{src} speed + HI events  |  n_events="
            f"{int((events['period'] == int(period)).sum()) if events is not None and len(events) else 0}",
            fontsize=10,
        )
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
        return fig

    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.0), sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, pers):
        _draw_speed_profile_on_ax(ax, bundle, events, thr, int(p), speed_col, max_points)
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  {src} speed + HI events",
        fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def _draw_sprint_timeline_on_ax(ax, ev: pd.DataFrame, thr: dict, height_metric: str, period: int):
    hcol, ylab = SPRINT_TIMELINE_METRICS[height_metric]
    if ev.empty:
        ax.set_title(f"P{period}: no HI events")
    else:
        for _, r in ev.iterrows():
            ax.bar(
                r["start_clock_min"],
                r[hcol],
                width=max(float(r["duration_s"]) / 60.0, 1e-3),
                align="edge",
                color="#2c3e50",
                alpha=0.75,
                edgecolor="none",
            )
        if height_metric in ("mean_speed", "peak_speed"):
            th = thr.get("t_high", np.nan)
            if np.isfinite(th):
                thr_lab = (
                    f"threshold={th:.2f}"
                    if thr.get("method") == "absolute_threshold"
                    else f"T_high={th:.2f}"
                )
                ax.axhline(th, color="#c0392b", ls="--", lw=1.0, label=thr_lab)
            mu = thr.get("mu_v", np.nan)
            if np.isfinite(mu):
                ax.axhline(mu, color="#7f8c8d", ls=":", lw=1.0, label=f"μ={mu:.2f}")
            ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("period clock (min)")
    ax.set_ylabel(ylab)
    ax.set_title(f"P{period}  |  n={len(ev)}", fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)


def plot_sprint_event_timeline(
    bundle: dict,
    events: pd.DataFrame | None = None,
    save_path: Path | None = None,
    period: int | None = None,
    height_metric: str = "mean_speed",
):
    """Bar timeline. ``period=None`` → P1|P2 side by side (shared y)."""
    if height_metric not in SPRINT_TIMELINE_METRICS:
        raise ValueError(f"height_metric must be one of {list(SPRINT_TIMELINE_METRICS)}")

    if events is None:
        events, thr = detect_relative_sprint_events(bundle)
    else:
        thr = _sprint_thr_from_events(events)

    src = ""
    if events is not None and len(events) and "speed_source" in events.columns:
        src = f"  ·  {events['speed_source'].iloc[0]}"
    thr_tag = ""
    if thr.get("k_low") is not None and thr.get("k_high") is not None:
        thr_tag = f"  ·  k_low={thr['k_low']:g}, k_high={thr['k_high']:g}"
    elif events is not None and len(events) and "k_low" in events.columns:
        thr_tag = f"  ·  k_low={events['k_low'].iloc[0]:g}, k_high={events['k_high'].iloc[0]:g}"
    if events is not None and len(events) and "merge_gap_s" in events.columns and pd.notna(events["merge_gap_s"].iloc[0]):
        thr_tag += f"  ·  merge<{float(events['merge_gap_s'].iloc[0]):g}s"

    if period is not None:
        ev = events.loc[events["period"] == int(period)].copy() if events is not None and len(events) else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(12.0, 3.8), layout="constrained")
        _draw_sprint_timeline_on_ax(ax, ev, thr, height_metric, int(period))
        ax.set_title(
            f"{bundle['nickname']}  |  {bundle['label']}  |  P{period}  |  "
            f"HI timeline · {height_metric}{src}{thr_tag}  |  n={len(ev)}",
            fontsize=10,
        )
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
        return fig

    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 3.8), sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, pers):
        ev = events.loc[events["period"] == int(p)].copy() if events is not None and len(events) else pd.DataFrame()
        _draw_sprint_timeline_on_ax(ax, ev, thr, height_metric, int(p))
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  HI timeline · {height_metric}{src}{thr_tag}",
        fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def _draw_recovery_scatter_on_ax(ax, ev: pd.DataFrame, y_metric: str, period: int, fig=None):
    ycol, ylab = SPRINT_Y_METRICS[y_metric]
    if ev.empty or ycol not in ev.columns:
        ax.set_title(f"P{period}: no events")
        return None
    m = np.isfinite(ev["recovery_before_s"]) & np.isfinite(ev[ycol])
    sub = ev.loc[m]
    sc = None
    if len(sub):
        sc = ax.scatter(
            sub["recovery_before_s"],
            sub[ycol],
            c=sub["start_clock_min"],
            cmap=SELF_NEAREST_CMAP,
            s=42,
            alpha=0.85,
            edgecolors="none",
        )
    ax.set_xlabel("recovery before event (s)")
    ax.set_ylabel(ylab)
    ax.set_title(
        f"P{period}  |  n={int(np.isfinite(ev['recovery_before_s']).sum()) if len(ev) else 0}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    return sc


def plot_sprint_recovery_scatter(
    bundle: dict,
    events: pd.DataFrame | None = None,
    y_metric: str = "sum_speed",
    save_path: Path | None = None,
    period: int | None = None,
    sub_clock_min: float | None = None,
    sub_period: int | None = None,
    sub_label: str = "sub",
):
    """Recovery (x) vs metric (y). ``period=None`` → P1|P2 side by side (shared axes)."""
    if y_metric not in SPRINT_Y_METRICS:
        raise ValueError(f"y_metric must be one of {list(SPRINT_Y_METRICS)}")

    if events is None:
        events, _ = detect_relative_sprint_events(bundle)

    src = ""
    if events is not None and len(events) and "speed_source" in events.columns:
        src = f"  ·  {events['speed_source'].iloc[0]}"
    thr_tag = ""
    if events is not None and len(events) and "k_low" in events.columns:
        thr_tag = f"  ·  k_low={events['k_low'].iloc[0]:g}, k_high={events['k_high'].iloc[0]:g}"
    if events is not None and len(events) and "merge_gap_s" in events.columns and pd.notna(events["merge_gap_s"].iloc[0]):
        thr_tag += f"  ·  merge<{float(events['merge_gap_s'].iloc[0]):g}s"

    if period is not None:
        ev = events.loc[events["period"] == int(period)].copy() if events is not None and len(events) else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(6.5, 5.0), layout="constrained")
        sc = _draw_recovery_scatter_on_ax(ax, ev, y_metric, int(period), fig)
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("event start · period clock (min)")
        if sub_period is None or int(sub_period) == int(period):
            mark_substitution_on_ax(ax, sub_clock_min, sub_label, vline=False)
        ax.set_title(
            f"{bundle['nickname']}  |  {bundle['label']}  |  P{period}  |  "
            f"recovery vs {y_metric}{src}{thr_tag}",
            fontsize=10,
        )
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
        return fig

    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.8), sharex=True, sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]
    sc_last = None
    for ax, p in zip(axes, pers):
        ev = events.loc[events["period"] == int(p)].copy() if events is not None and len(events) else pd.DataFrame()
        sc = _draw_recovery_scatter_on_ax(ax, ev, y_metric, int(p), fig)
        if sc is not None:
            sc_last = sc
        if sub_period is None or int(sub_period) == int(p):
            mark_substitution_on_ax(ax, sub_clock_min, sub_label, vline=False)
    if sc_last is not None:
        cbar = fig.colorbar(sc_last, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("event start · period clock (min)")
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  recovery vs {y_metric}{src}{thr_tag}",
        fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def add_maxnorm_recovery_scores(
    events: pd.DataFrame,
    y_col: str,
    x_col: str = "recovery_before_s",
) -> tuple[pd.DataFrame, dict]:
    """Divide-by-max on recovery ``X`` and output ``Y``, then bounded scores.

    ``x = X/X_max``, ``y = Y/Y_max`` (player-match maxima over finite ``X>0``
    and finite ``Y>=0``). Then ``q = y/x`` and

    - ``s_angle = (2/π) arctan(q)``
    - ``s_ratio = q/(1+q) = y/(x+y)``

    Both equal 0.5 when ``x=y``. Divide-by-max (not min-max) so the shortest
    rest is not ``x=0``. First-event ``X`` (NaN) stays NaN.
    """
    out = events.copy()
    x = pd.to_numeric(out[x_col], errors="coerce")
    y = pd.to_numeric(out[y_col], errors="coerce")
    ok = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y >= 0)
    x_max = float(x.loc[ok].max()) if bool(ok.any()) else np.nan
    y_max = float(y.loc[ok].max()) if bool(ok.any()) else np.nan
    xn = x / x_max
    yn = y / y_max
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yn / xn
        s_angle = (2.0 / np.pi) * np.arctan(q)
        s_ratio = yn / (xn + yn)
    out["x_norm"] = xn
    out["y_norm"] = yn
    out["q_yx"] = q
    out["s_angle"] = s_angle
    out["s_ratio"] = s_ratio
    for col in ("x_norm", "y_norm", "q_yx", "s_angle", "s_ratio"):
        out.loc[~ok, col] = np.nan
    scales = {
        "x_col": x_col,
        "y_col": y_col,
        "x_max": x_max,
        "y_max": y_max,
        "n": int(ok.sum()),
    }
    return out, scales


def plot_maxnorm_score_vs_time(
    bundle: dict,
    events: pd.DataFrame,
    y_metric: str,
    save_path: Path | None = None,
    short_recovery_s: float = 10.0,
):
    """``s_angle`` and ``s_ratio`` vs period clock. Open markers: recovery < ``short_recovery_s``."""
    if y_metric not in SPRINT_Y_METRICS:
        raise ValueError(f"y_metric must be one of {list(SPRINT_Y_METRICS)}")
    scored, scales = add_maxnorm_recovery_scores(events, y_metric)
    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 6.2), sharex="col", sharey="row", layout="constrained")
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    ylab = SPRINT_Y_METRICS[y_metric][1]
    rows = (
        ("s_angle", r"$S_{\mathrm{angle}}=(2/\pi)\arctan(y/x)$"),
        ("s_ratio", r"$S_{\mathrm{ratio}}=y/(x+y)$"),
    )
    for col_i, p in enumerate(pers):
        ev = scored.loc[scored["period"] == int(p)].copy()
        t = pd.to_numeric(ev["start_clock_min"], errors="coerce")
        rec = pd.to_numeric(ev["recovery_before_s"], errors="coerce")
        short = rec < float(short_recovery_s)
        for row_i, (col, ylab_s) in enumerate(rows):
            ax = axes[row_i, col_i]
            z = pd.to_numeric(ev[col], errors="coerce")
            m = np.isfinite(t) & np.isfinite(z)
            if m.any():
                ax.plot(t[m], z[m], color="#1f4e79", lw=1.1, marker="o", ms=4.5, zorder=2)
                if short[m].any():
                    ax.plot(
                        t[m & short],
                        z[m & short],
                        ls="none",
                        marker="o",
                        ms=8.0,
                        markerfacecolor="none",
                        markeredgecolor="#c0392b",
                        markeredgewidth=1.3,
                        label=f"recovery < {short_recovery_s:g}s",
                        zorder=3,
                    )
            ax.axhline(0.5, color="#b03a2e", ls="--", lw=1.0, label=r"$S=0.5$  ($x=y$)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"P{int(p)}  |  n={int(m.sum())}", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(ylab_s, fontsize=9)
            if row_i == 1:
                ax.set_xlabel("period clock (min)")
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  max-norm {ylab}  |  "
        f"X_max={scales['x_max']:.1f}s  Y_max={scales['y_max']:.2f}  |  "
        f"open red = recovery < {short_recovery_s:g}s",
        fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, scored, scales


def plot_maxnorm_unit_scatter(
    bundle: dict,
    events: pd.DataFrame,
    y_metric: str,
    save_path: Path | None = None,
):
    """Unit-square scatter of ``(x_norm, y_norm)`` with the ``y=x`` line (S=0.5)."""
    if y_metric not in SPRINT_Y_METRICS:
        raise ValueError(f"y_metric must be one of {list(SPRINT_Y_METRICS)}")
    scored, scales = add_maxnorm_recovery_scores(events, y_metric)
    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.6), sharex=True, sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]
    ylab = SPRINT_Y_METRICS[y_metric][1]
    sc_last = None
    for ax, p in zip(axes, pers):
        ev = scored.loc[scored["period"] == int(p)]
        xn = pd.to_numeric(ev["x_norm"], errors="coerce")
        yn = pd.to_numeric(ev["y_norm"], errors="coerce")
        t = pd.to_numeric(ev["start_clock_min"], errors="coerce")
        m = np.isfinite(xn) & np.isfinite(yn)
        ax.plot([0, 1], [0, 1], color="#b03a2e", ls="--", lw=1.1, label=r"$y=x$  ($S=0.5$)")
        if m.any():
            sc_last = ax.scatter(
                xn[m],
                yn[m],
                c=t[m],
                cmap=SELF_NEAREST_CMAP,
                s=42,
                alpha=0.85,
                edgecolors="none",
            )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x=X/X_{\max}$  (recovery)")
        ax.set_ylabel(r"$y=Y/Y_{\max}$")
        ax.set_title(f"P{int(p)}  |  n={int(m.sum())}", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=7)
    if sc_last is not None:
        cbar = fig.colorbar(sc_last, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("event start · period clock (min)")
    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  max-norm unit square  |  {ylab}  |  "
        f"X_max={scales['x_max']:.1f}s  Y_max={scales['y_max']:.2f}",
        fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, scored, scales


def _fit_line_theilsen_or_ols(
    x: np.ndarray, y: np.ndarray, method: str = "theilsen"
) -> tuple[float, float, str]:
    """Intercept ``a`` and slope ``b`` for ``y = a + b x``.

    ``method='theilsen'`` (default) tries Theil–Sen then OLS.
    ``method='ols'`` is ordinary least squares only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan, np.nan, "too_few"
    want = str(method).lower().strip()
    if want not in ("theilsen", "ols"):
        raise ValueError("method must be 'theilsen' or 'ols'")

    if want == "theilsen":
        try:
            from scipy.stats import theilslopes

            slope, intercept, _, _ = theilslopes(y, x)
            if np.isfinite(slope) and np.isfinite(intercept):
                return float(intercept), float(slope), "theilsen"
        except Exception:
            pass

    A = np.column_stack([np.ones(len(x)), x])
    try:
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        intercept, slope = float(coef[0]), float(coef[1])
        if np.isfinite(slope) and np.isfinite(intercept):
            return intercept, slope, "ols"
    except Exception:
        pass
    return np.nan, np.nan, "fail"


def add_rolling_line_fit(
    events: pd.DataFrame,
    y_col: str,
    *,
    n_window: int = 8,
    min_n: int = 5,
    min_x_range_s: float = 20.0,
    x_col: str = "recovery_before_s",
    x_ref: float | None = None,
    method: str = "theilsen",
    s_x: float | None = None,
    s_y: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Causal rolling line ``y = a + b x`` on the last ``n_window`` usable events.

    Same period only. Scales ``s_x``, ``s_y`` default to that table's 90th
    percentiles (frozen), or use the passed values. ``b_star = b * s_x / s_y``
    so the angle is not unit-dependent.

    - ``a``: intercept at ``x=0`` (extrapolation)
    - ``h_ref = a + b * x_ref``: height at a fixed recovery (default: median ``X``)
    - ``theta_rad = arctan(b_star)``, ``A = (theta + π/2)/π`` in ``[0,1]``
    """
    out = events.copy().sort_values(["period", "start_time_s", "event_id"]).reset_index(drop=True)
    x = pd.to_numeric(out[x_col], errors="coerce")
    y = pd.to_numeric(out[y_col], errors="coerce")
    period = pd.to_numeric(out["period"], errors="coerce")
    usable = np.isfinite(x) & (x > 0) & np.isfinite(y) & np.isfinite(period)
    xu = x.loc[usable].to_numpy(dtype=float)
    yu = y.loc[usable].to_numpy(dtype=float)
    if s_x is None or not (np.isfinite(s_x) and float(s_x) > 0):
        s_x = float(np.percentile(xu, 90.0)) if len(xu) else np.nan
        if not (np.isfinite(s_x) and s_x > 0):
            s_x = float(np.nanmedian(xu)) if len(xu) else np.nan
    else:
        s_x = float(s_x)
    if s_y is None or not (np.isfinite(s_y) and float(s_y) > 0):
        s_y = float(np.percentile(yu, 90.0)) if len(yu) else np.nan
        if not (np.isfinite(s_y) and s_y > 0):
            s_y = float(np.nanmedian(yu)) if len(yu) else np.nan
    else:
        s_y = float(s_y)
    if x_ref is None:
        x_ref = float(np.nanmedian(xu)) if len(xu) else np.nan

    n = len(out)
    a_arr = np.full(n, np.nan)
    b_arr = np.full(n, np.nan)
    n_arr = np.full(n, np.nan)
    meth = np.full(n, "", dtype=object)
    idx = np.flatnonzero(usable.to_numpy())
    periods = period.to_numpy()
    x_np = x.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)

    for p in sorted({int(v) for v in periods[idx] if np.isfinite(v)}):
        loc = [i for i in idx if int(periods[i]) == p]
        for k, i in enumerate(loc):
            w0 = max(0, k - (int(n_window) - 1))
            win = loc[w0 : k + 1]
            n_arr[i] = float(len(win))
            if len(win) < int(min_n):
                meth[i] = "too_few"
                continue
            xw, yw = x_np[win], y_np[win]
            if float(np.nanmax(xw) - np.nanmin(xw)) < float(min_x_range_s):
                meth[i] = "x_range"
                continue
            a_i, b_i, m_i = _fit_line_theilsen_or_ols(xw, yw, method=method)
            a_arr[i], b_arr[i], meth[i] = a_i, b_i, m_i

    out["roll_n"] = n_arr
    out["roll_a"] = a_arr
    out["roll_b"] = b_arr
    out["roll_method"] = meth
    with np.errstate(divide="ignore", invalid="ignore"):
        b_star = b_arr * (s_x / s_y)
        theta = np.arctan(b_star)
        a_star = (theta + 0.5 * np.pi) / np.pi
        h_ref = a_arr + b_arr * float(x_ref)
    out["roll_b_star"] = b_star
    out["roll_theta_rad"] = theta
    out["roll_theta_deg"] = np.degrees(theta)
    out["roll_A"] = a_star
    out["roll_H"] = h_ref
    meta = {
        "y_col": y_col,
        "x_col": x_col,
        "n_window": int(n_window),
        "min_n": int(min_n),
        "min_x_range_s": float(min_x_range_s),
        "s_x": s_x,
        "s_y": s_y,
        "x_ref": float(x_ref) if x_ref is not None else np.nan,
        "n_usable": int(usable.sum()),
        "n_fit": int(np.isfinite(a_arr).sum()),
        "method": str(method),
    }
    return out, meta


def plot_rolling_line_vs_time(
    bundle: dict,
    events: pd.DataFrame,
    y_metric: str,
    *,
    n_window: int = 8,
    min_n: int = 5,
    min_x_range_s: float = 20.0,
    x_ref: float | None = None,
    method: str = "theilsen",
    save_path: Path | None = None,
):
    """Rolling line angle ``A(t)``, intercept ``a(t)`` at x=0, and height ``H`` at ``x_ref``."""
    if y_metric not in SPRINT_Y_METRICS:
        raise ValueError(f"y_metric must be one of {list(SPRINT_Y_METRICS)}")
    fitted, meta = add_rolling_line_fit(
        events,
        y_metric,
        n_window=n_window,
        min_n=min_n,
        min_x_range_s=min_x_range_s,
        x_ref=x_ref,
        method=method,
    )
    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(3, n, figsize=(6.2 * n, 8.4), sharex="col", layout="constrained")
    if n == 1:
        axes = np.asarray(axes).reshape(3, 1)

    ylab = SPRINT_Y_METRICS[y_metric][1]
    xref = meta["x_ref"]
    rows = (
        ("roll_A", r"$A=(\theta+\pi/2)/\pi$", 0.5, (-0.05, 1.05)),
        ("roll_a", r"intercept $a$ at $x=0$  (" + ylab + ")", 0.0, None),
        ("roll_H", rf"$H=a+b\,x_{{\mathrm{{ref}}}}$  ($x_{{\mathrm{{ref}}}}={xref:.0f}$ s)", None, None),
    )
    for col_i, p in enumerate(pers):
        ev = fitted.loc[fitted["period"] == int(p)].copy()
        t = pd.to_numeric(ev["start_clock_min"], errors="coerce")
        for row_i, (col, ylab_s, href, ylim) in enumerate(rows):
            ax = axes[row_i, col_i]
            z = pd.to_numeric(ev[col], errors="coerce")
            m = np.isfinite(t) & np.isfinite(z)
            if m.any():
                ax.plot(t[m], z[m], color="#1f4e79", lw=1.15, marker="o", ms=4.2)
            if href is not None:
                ax.axhline(href, color="#b03a2e", ls="--", lw=1.0)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"P{int(p)}  |  n_fit={int(m.sum())}", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(ylab_s, fontsize=9)
            if row_i == 2:
                ax.set_xlabel("period clock (min)")

    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  rolling line  {ylab}  |  "
        f"N={meta['n_window']} (min {meta['min_n']}, Δx≥{meta['min_x_range_s']:g}s)  |  "
        f"s_x={meta['s_x']:.1f}s  s_y={meta['s_y']:.2f}  x_ref={meta['x_ref']:.1f}s",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, fitted, meta


def plot_rolling_line_component_vs_time(
    bundle: dict,
    events: pd.DataFrame,
    y_metric: str,
    component: str,
    *,
    n_window: int = 8,
    min_n: int = 5,
    min_x_range_s: float = 20.0,
    method: str = "ols",
    s_x: float | None = None,
    s_y: float | None = None,
    save_path: Path | None = None,
    sub_clock_min: float | None = None,
    sub_period: int | None = None,
    sub_label: str = "sub",
    sub_color: str = "#6c3483",
):
    """One-row P1|P2 plot of scaled line angle ``A(t)`` or intercept ``a(t)`` at x=0."""
    if y_metric not in SPRINT_Y_METRICS:
        raise ValueError(f"y_metric must be one of {list(SPRINT_Y_METRICS)}")
    key = str(component).strip()
    if key in ("A", "roll_A", "angle"):
        col, href, ylim = "roll_A", 0.5, (-0.05, 1.05)
        ylab_s = r"$A=(\theta+\pi/2)/\pi$  (scaled line angle)"
        kind = "line angle $A(t)$"
    elif key in ("a", "roll_a", "intercept"):
        col, href, ylim = "roll_a", 0.0, None
        ylab_s = r"intercept $a$ at $x=0$  (" + SPRINT_Y_METRICS[y_metric][1] + ")"
        kind = "intercept $a(t)$ at $x=0$"
    else:
        raise ValueError("component must be 'A' (line angle) or 'a' (intercept at 0)")

    fitted, meta = add_rolling_line_fit(
        events,
        y_metric,
        n_window=n_window,
        min_n=min_n,
        min_x_range_s=min_x_range_s,
        method=method,
        s_x=s_x,
        s_y=s_y,
    )
    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 3.35), sharex="col", layout="constrained")
    if n == 1:
        axes = np.asarray([axes])

    ylab = SPRINT_Y_METRICS[y_metric][1]
    meth = str(meta.get("method", method)).upper()
    if meth == "THEILSEN":
        meth = "Theil–Sen"
    for col_i, p in enumerate(pers):
        ax = axes[col_i]
        ev = fitted.loc[fitted["period"] == int(p)]
        t = pd.to_numeric(ev["start_clock_min"], errors="coerce")
        z = pd.to_numeric(ev[col], errors="coerce")
        m = np.isfinite(t) & np.isfinite(z)
        if m.any():
            ax.plot(t[m], z[m], color="#1f4e79", lw=1.25, marker="o", ms=4.2)
        if href is not None:
            ax.axhline(href, color="#b03a2e", ls="--", lw=1.0)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"P{int(p)}  |  n_fit={int(m.sum())}", fontsize=10)
        if col_i == 0:
            ax.set_ylabel(ylab_s, fontsize=9)
        ax.set_xlabel("period clock (min)")
        if sub_period is None or int(sub_period) == int(p):
            mark_substitution_on_ax(
                ax, sub_clock_min, sub_label, vline=True, color=sub_color
            )

    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  {meth}  |  {kind}  |  {ylab}  |  "
        f"N={meta['n_window']} (min {meta['min_n']}, Δx≥{meta['min_x_range_s']:g}s)  |  "
        f"s_x={meta['s_x']:.1f}s  s_y={meta['s_y']:.2f}",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, fitted, meta


def plot_rolling_line_ols_vs_theilsen(
    bundle: dict,
    events: pd.DataFrame,
    y_metric: str,
    *,
    n_window: int = 8,
    min_n: int = 5,
    min_x_range_s: float = 20.0,
    save_path: Path | None = None,
):
    """Overlay OLS vs Theil–Sen for scaled line angle ``A(t)`` and intercept ``a(t)`` at x=0."""
    if y_metric not in SPRINT_Y_METRICS:
        raise ValueError(f"y_metric must be one of {list(SPRINT_Y_METRICS)}")
    ols, m_ols = add_rolling_line_fit(
        events,
        y_metric,
        n_window=n_window,
        min_n=min_n,
        min_x_range_s=min_x_range_s,
        method="ols",
    )
    sen, m_sen = add_rolling_line_fit(
        events,
        y_metric,
        n_window=n_window,
        min_n=min_n,
        min_x_range_s=min_x_range_s,
        method="theilsen",
    )
    pers = list(periods_in(bundle["tracking"]))
    n = max(len(pers), 1)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 6.4), sharex="col", layout="constrained")
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    ylab = SPRINT_Y_METRICS[y_metric][1]
    styles = (
        (sen, "#c0392b", "Theil–Sen", 1.15, "s"),
        (ols, "#1f4e79", "OLS", 1.45, "o"),
    )
    rows = (
        ("roll_A", r"$A=(\theta+\pi/2)/\pi$  (scaled line angle)", 0.5, (-0.05, 1.05)),
        ("roll_a", r"intercept $a$ at $x=0$  (" + ylab + ")", 0.0, None),
    )
    for col_i, p in enumerate(pers):
        for row_i, (col, ylab_s, href, ylim) in enumerate(rows):
            ax = axes[row_i, col_i]
            n_fit = 0
            for fitted, color, lab, lw, mk in styles:
                ev = fitted.loc[fitted["period"] == int(p)]
                t = pd.to_numeric(ev["start_clock_min"], errors="coerce")
                z = pd.to_numeric(ev[col], errors="coerce")
                msk = np.isfinite(t) & np.isfinite(z)
                n_fit = max(n_fit, int(msk.sum()))
                if msk.any():
                    ax.plot(
                        t[msk],
                        z[msk],
                        color=color,
                        lw=lw,
                        marker=mk,
                        ms=4.0,
                        label=lab,
                    )
            if href is not None:
                ax.axhline(href, color="#7f8c8d", ls="--", lw=1.0)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"P{int(p)}  |  n_fit={n_fit}", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(ylab_s, fontsize=9)
            if row_i == 1:
                ax.set_xlabel("period clock (min)")
            if row_i == 0 and col_i == n - 1:
                ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"{bundle['nickname']}  |  {bundle['label']}  |  OLS vs Theil–Sen  |  {ylab}  |  "
        f"N={m_ols['n_window']} (min {m_ols['min_n']}, Δx≥{m_ols['min_x_range_s']:g}s)  |  "
        f"s_x={m_ols['s_x']:.1f}s  s_y={m_ols['s_y']:.2f}",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, {"ols": ols, "theilsen": sen}, {"ols": m_ols, "theilsen": m_sen}


def _flatten_figs(obj):
    """Yield matplotlib figures from nested dict/list/tuple returns."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_figs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten_figs(v)
    else:
        # duck-type: matplotlib Figure
        if hasattr(obj, "savefig"):
            yield obj


def run_player_when(when: str, player_id: str | int, out_dir: Path, show: bool = False, force: bool = False):
    bundle = load_player_match(when, player_id, force=force)
    folder = out_dir / f"{bundle['match_id']}_{safe_name(bundle['nickname'])}"
    folder.mkdir(parents=True, exist_ok=True)

    figs = []
    # Possession timelines first (context before projected window graphs)
    figs4, figs3 = plot_possession_timelines(
        bundle,
        save_path_4=folder / "00_possession_timeline_4class.png",
        save_path_3=folder / "00b_possession_timeline_3class.png",
    )
    figs.append(figs4)
    figs.append(figs3)

    figs.append(plot_as_time_colored(bundle, folder / "01_as_time_colored.png"))
    figs.append(plot_as_by_possession(bundle, "A", folder / "02_as_class_method_A.png"))
    figs.append(plot_as_by_possession(bundle, "B", folder / "03_as_class_method_B.png"))

    peer_figs, peer_tabs, peer_diags = plot_peer_percentiles(
        bundle, folder / "04_peer_percentiles.png"
    )
    figs.append(peer_figs)
    for p, diag in peer_diags.items():
        pd.DataFrame([diag]).to_csv(folder / f"04_peer_diagnostics_P{p}.csv", index=False)
        tab = peer_tabs[p]
        if not tab.empty:
            tab.to_csv(folder / f"04_peer_percentiles_P{p}.csv", index=False)

    figs.append(plot_rolling_max(bundle, folder / "05_rolling_max.png"))
    figs.append(plot_kinematic_heatmaps(bundle, folder / "06_kinematic_heatmaps.png", zoom="full"))
    figs.append(plot_kinematic_heatmaps(bundle, folder / "06b_kinematic_heatmaps_half.png", zoom="half"))
    figs.append(plot_jerk_heatmap_tight(bundle, folder / "06c_jerk_heatmap_tight.png"))
    figs.append(plot_distance_progression(bundle, folder / "07_distance_progression.png"))
    fig_ph, segs_ph = plot_phase_duration_histograms(
        bundle, folder / "08_phase_duration_hist_3class.png"
    )
    figs.append(fig_ph)
    if segs_ph is not None and not segs_ph.empty:
        segs_ph.to_csv(folder / "08_phase_durations_3class.csv", index=False)
        ps.segment_duration_stats(segs_ph).to_csv(folder / "08_phase_duration_stats.csv", index=False)

    sn_figs, sn_tabs = plot_self_nearest_scatter(
        bundle, folder / "09_self_nearest_scatter.png"
    )
    figs.append(sn_figs)
    for p, tab in sn_tabs.items():
        if tab is not None and not tab.empty:
            tab.to_csv(folder / f"09_self_nearest_scatter_P{p}.csv", index=False)

    for metric, tag in (("speed", "speed"), ("distance", "distance")):
        fig_prog, tab_res = plot_self_nearest_residual_progression(
            bundle, folder / f"09b_residual_progression_{tag}.png", metric=metric
        )
        figs.append(fig_prog)
        fig_ph, _ = plot_self_nearest_residual_phases(
            bundle, folder / f"09c_residual_phases_{tag}.png", metric=metric
        )
        figs.append(fig_ph)
        fig_cu, _ = plot_self_nearest_residual_cumulative(
            bundle, folder / f"09d_residual_cumulative_{tag}.png", metric=metric
        )
        figs.append(fig_cu)
        fig_vs, _ = plot_self_nearest_vs_references(
            bundle, folder / f"09e_residual_lead_{tag}.png", metric=metric
        )
        figs.append(fig_vs)
        fig_inst, _ = plot_self_nearest_vs_references(
            bundle, folder / f"09f_residual_instant_{tag}.png", metric=metric, cumulative=False
        )
        figs.append(fig_inst)
        if tab_res is not None and not tab_res.empty:
            tab_res.to_csv(folder / f"09_self_nearest_residuals_{tag}.csv", index=False)

    meta = {
        "nickname": bundle["nickname"],
        "player_id": bundle["player_id"],
        "team": bundle["team_name"],
        "opp": bundle["opp_team_name"],
        "match": bundle["label"],
        "side": bundle["side"],
        "r_pz": bundle["r_pz"],
        "out_dir": str(folder),
        "periods": ",".join(str(p) for p in periods_in(bundle["tracking"])),
        "roll_win_min": ROLL_WIN_MIN,
        "roll_step_min": ROLL_STEP_MIN,
        "roll_overlap_s": rolling_overlap_min() * 60.0,
    }
    pd.Series(meta).to_csv(folder / "meta.csv")
    print("Wrote figures to", folder)
    if not show:
        for fig in _flatten_figs(figs):
            plt.close(fig)
    # return first period diag for backwards callers
    diag_p = next(iter(peer_diags.values())) if peer_diags else {}
    return bundle, folder, diag_p
