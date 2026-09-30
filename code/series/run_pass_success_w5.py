"""5-minute pass success and pass+xG composite for 5 players x 2 matches, plus 3 extras.

P(t) is the 5-minute sliding complete-pass rate (hop 1 min, min n = 2), per half.
Q(t) is nearby shot xG. w(t) peaks at 0.15 and is a Gaussian over ±2.5 min.
C(t) = (1-w)P + wQ, in [0, 1].
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES_DIR = Path(__file__).resolve().parent
PKG_DIR = SERIES_DIR.parents[1]
sys.path.insert(0, str(PKG_DIR.parent))
from beyond_physical_fatigue.code.paths import ensure_lib_on_path, football_root  # noqa: E402

ensure_lib_on_path()
ROOT = football_root()
CM = ROOT / "cognitive_metrics"
# Prefer the vendored xg package inside this repo when present.
_xg_parent = SERIES_DIR.parent  # .../code
if (_xg_parent / "xg" / "pff.py").exists() and str(_xg_parent) not in sys.path:
    sys.path.insert(0, str(_xg_parent))
elif str(CM) not in sys.path:
    sys.path.insert(0, str(CM))

from xg.pff import score_pff_shots  # noqa: E402

OUT = SERIES_DIR
WINDOW_MIN = 5.0
HOP_MIN = 1.0
MIN_N = 2
W_MAX = 0.15
SHOT_HALF_WIDTH = 2.5
SHOT_SIGMA = SHOT_HALF_WIDTH / 2.0

BOTH = [
    {"game_id": "3820", "dir": ROOT / "3820_MAR_CRO", "label": "3820 MAR-CRO"},
    {"game_id": "10516", "dir": ROOT / "10516_CRO_MAR", "label": "10516 CRO-MAR"},
]
FIVE = ["Amrabat", "Modric", "Gvardiol", "Hakimi", "Perisic"]
EXTRAS = [
    {
        "needle": "Morita",
        "game_id": "3854",
        "dir": ROOT / "3854_JPN_ESP",
        "label": "3854 JPN-ESP",
        "mark": {"clock_min": 94.0, "period": 2, "label": "cramp 90+4"},
    },
    {
        "needle": "Acuna",
        "game_id": "3835",
        "dir": ROOT / "3835_ARG_MEX",
        "label": "3835 ARG-MEX",
        "mark": {"clock_min": 92.0, "period": 2, "label": "cramp 90+2"},
    },
    {
        "needle": "Musah",
        "game_id": "3815",
        "dir": ROOT / "3815_USA_WAL",
        "label": "3815 USA-WAL",
        "mark": {"clock_min": 75.0, "period": 2, "label": "cramp 75'"},
    },
]


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()


def as_pid(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def find_player(match_dir: Path, game_id: str, needle: str):
    raw = json.loads((match_dir / f"{game_id}_roster.json").read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("players") or raw.get("roster") or []
    key = fold(needle)
    for rec in raw:
        if str(rec.get("positionGroupType") or "").upper() == "GK":
            continue
        pl = rec.get("player") or {}
        nick = str(pl.get("nickname") or pl.get("name") or "")
        if key in fold(nick):
            return int(pl["id"]), nick
    raise RuntimeError(f"{needle} not on roster for {game_id}")


def load_passes(match_dir: Path, game_id: str, focal_pid: int) -> pd.DataFrame:
    raw_events = json.loads((match_dir / f"{game_id}_events.json").read_text(encoding="utf-8"))
    rows = []
    for e in raw_events:
        pe = e.get("possessionEvents") or {}
        ge = e.get("gameEvents") or {}
        if pe.get("possessionEventType") != "PA":
            continue
        passer_id = as_pid(pe.get("passerPlayerId"))
        recv_id = as_pid(pe.get("receiverPlayerId"))
        target_id = as_pid(pe.get("targetPlayerId"))
        other_id = recv_id if recv_id is not None else target_id
        if focal_pid not in {passer_id, other_id}:
            continue
        rows.append(
            {
                "period": ge.get("period"),
                "game_clock_min": (float(ge["startGameClock"]) / 60.0) if ge.get("startGameClock") is not None else np.nan,
                "success": pe.get("passOutcomeType") == "C",
            }
        )
    return pd.DataFrame(rows)


def score_shots(match_dir: Path, game_id: str, display_name: str) -> pd.DataFrame:
    raw_events = json.loads((match_dir / f"{game_id}_events.json").read_text(encoding="utf-8"))
    key = fold(display_name.split()[-1])
    rows = []
    n_skip = 0
    for event in raw_events:
        pe = event.get("possessionEvents") or {}
        if pe.get("possessionEventType") != "SH":
            continue
        if key not in fold(pe.get("shooterPlayerName") or ""):
            continue
        try:
            rows.extend(score_pff_shots([event], shooter_name=None))
        except Exception:
            n_skip += 1
    shots = pd.DataFrame(rows)
    if n_skip:
        print(f"    skipped {n_skip} shots with no freeze-frame", flush=True)
    if shots.empty:
        return shots
    shots = shots.drop(columns=["features"], errors="ignore")
    shots["game_clock_min"] = pd.to_numeric(shots["game_clock_s"], errors="coerce") / 60.0
    return shots


def pass_success_curve(pass_inv: pd.DataFrame, clock_start: float) -> pd.DataFrame:
    if pass_inv.empty:
        return pd.DataFrame(columns=["t_min", "window_start_min", "window_end_min", "n", "P"])
    t = pd.to_numeric(pass_inv["game_clock_min"], errors="coerce").to_numpy(dtype=float)
    success = pass_inv["success"].astype(bool).to_numpy()
    t_max = float(np.nanmax(t))
    starts = np.arange(clock_start, t_max + 1e-9, HOP_MIN) if t_max >= clock_start else np.array([])
    rows = []
    for start in starts:
        end = float(start + WINDOW_MIN)
        mask = (t >= start) & (t < end)
        n = int(mask.sum())
        n_ok = int(success[mask].sum()) if n else 0
        rows.append(
            {
                "t_min": float(start + 0.5 * WINDOW_MIN),
                "window_start_min": float(start),
                "window_end_min": end,
                "n": n,
                "P": (n_ok / n) if n >= MIN_N else np.nan,
            }
        )
    return pd.DataFrame(rows)


def shot_modifier(t, shots):
    t = np.asarray(t, dtype=float)
    q = np.full_like(t, np.nan)
    if not shots or t.size == 0:
        return np.zeros_like(t), q
    tj = np.asarray([float(s[0]) for s in shots], dtype=float)
    xg = np.asarray([float(s[1]) for s in shots], dtype=float)
    dt = t[:, None] - tj[None, :]
    g = np.exp(-0.5 * (dt / SHOT_SIGMA) ** 2)
    g = np.where(np.abs(dt) <= SHOT_HALF_WIDTH, g, 0.0)
    gsum = g.sum(axis=1)
    w = W_MAX * np.clip(gsum, 0.0, 1.0)
    np.divide((g * xg[None, :]).sum(axis=1), gsum, out=q, where=gsum > 0)
    return w, q


def half_bundle(inv: pd.DataFrame, shot_df: pd.DataFrame):
    halves = {}
    for period, clock_start, key in ((1, 0.0, "p1"), (2, 45.0, "p2")):
        sub = inv.loc[pd.to_numeric(inv["period"], errors="coerce").eq(period)].copy() if len(inv) else inv
        if shot_df is None or shot_df.empty:
            shots = []
        else:
            shots = list(
                map(
                    tuple,
                    shot_df.loc[
                        pd.to_numeric(shot_df["period"], errors="coerce").eq(period),
                        ["game_clock_min", "xG"],
                    ].to_numpy(),
                )
            )
        curve = pass_success_curve(sub, clock_start)
        if len(curve):
            w, q = shot_modifier(curve["t_min"].to_numpy(), shots)
            curve["w"] = w
            curve["Q"] = q
            p = curve["P"].to_numpy(dtype=float)
            curve["C"] = np.where(w > 0, (1.0 - w) * p + w * q, p)
        n = len(sub)
        n_ok = int(sub["success"].sum()) if n else 0
        halves[key] = {
            "period": period,
            "curve": curve,
            "shots": shots,
            "t_max": float(np.nanmax(sub["game_clock_min"])) if n else clock_start,
            "overall_pct": (100.0 * n_ok / n) if n else np.nan,
            "n": n,
            "n_ok": n_ok,
        }
    return halves


def mark(ax, period: int, mark_info):
    if not mark_info or int(mark_info.get("period", -1)) != int(period):
        return
    x = float(mark_info["clock_min"])
    ax.axvline(x, color="#922b21", ls=":", lw=1.1)
    ax.text(x, 0.98, mark_info.get("label", "mark"), transform=ax.get_xaxis_transform(),
            color="#922b21", fontsize=8, ha="left", va="top", rotation=90)


def plot_success(halves, name, match_label, dest, mark_info=None):
    net_c, fail_c = "#1f4e79", "#c0392b"
    fig, axes = plt.subplots(
        2, 2, figsize=(13.2, 6.4), sharey="row",
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08, "wspace": 0.08},
    )
    for col, key, xlim0 in ((0, "p1", 0.0), (1, "p2", 45.0)):
        half = halves[key]
        curve = half["curve"]
        ax, axn = axes[0, col], axes[1, col]
        xlim1 = max(xlim0 + 48.0, half["t_max"] + 1.0)
        if len(curve):
            t = curve["t_min"].to_numpy()
            ax.plot(t, 100.0 * curve["P"], color=net_c, lw=2.0, label="5-min pass success")
            axn.bar(t, curve["n"], width=HOP_MIN * 0.85, color=net_c, alpha=0.75, edgecolor="none")
        if np.isfinite(half["overall_pct"]):
            ax.axhline(half["overall_pct"], color="0.45", ls=":", lw=1.1, label=f"half overall  {half['overall_pct']:.1f}%")
        axn.axhline(MIN_N, color=fail_c, ls="--", lw=1.0, label=f"min n = {MIN_N}")
        for a in (ax, axn):
            a.axvline(45.0 if key == "p1" else 90.0, color="0.65", ls="--", lw=0.9)
            a.set_xlim(xlim0, xlim1)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            mark(a, half["period"], mark_info)
        ax.set_ylim(-2, 108)
        ax.set_title("P1  first half (+ extra time)" if key == "p1" else "P2  second half (+ extra time)")
        axn.set_xlabel("Game clock (min)")
        if col == 0:
            ax.set_ylabel("% success in window")
            axn.set_ylabel("Passes in window")
            ax.legend(frameon=False, loc="lower left")
            axn.legend(frameon=False, loc="upper right")
    fig.suptitle(f"{name} — 5-min pass success, hop 1 min, {match_label}", y=0.98, fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.09)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_composite(halves, name, match_label, dest, mark_info=None):
    ok_c, net_c, xg_c = "#1b7f4e", "#1f4e79", "#b45309"
    fig, axes = plt.subplots(
        3, 2, figsize=(13.6, 8.4), sharey="row",
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.25], "hspace": 0.10, "wspace": 0.08},
    )
    for col, key, xlim0 in ((0, "p1", 0.0), (1, "p2", 45.0)):
        half = halves[key]
        curve = half["curve"]
        ax_p, ax_q, ax_c = axes[:, col]
        xlim1 = max(xlim0 + 48.0, half["t_max"] + 1.0)
        if len(curve):
            t = curve["t_min"].to_numpy()
            ax_p.plot(t, curve["P"], color=net_c, lw=2.0, label="P(t)  5-min pass success")
            q_line = curve["Q"].where(curve["w"] > 0)
            ax_q.fill_between(t, 0.0, curve["w"], color="0.75", alpha=0.45, label=f"w(t)  max={W_MAX:.2f}")
            ax_q.plot(t, q_line, color=xg_c, lw=2.0, label="Q(t)  nearby shot xG")
            ax_c.plot(t, curve["P"], color=net_c, lw=1.3, ls="--", alpha=0.55, label="P(t)")
            ax_c.plot(t, curve["C"], color=ok_c, lw=2.2, label=r"$C(t)=(1-w)P+wQ$")
        for tj, xg in half["shots"]:
            ax_q.vlines(tj, 0.0, xg, color=xg_c, lw=1.4, alpha=0.9)
            ax_q.scatter([tj], [xg], color=xg_c, s=28, zorder=3)
        if not half["shots"]:
            ax_q.text(0.04, 0.72, "no shots", transform=ax_q.transAxes, color="0.45", fontsize=10)
        for a in (ax_p, ax_q, ax_c):
            a.axvline(45.0 if key == "p1" else 90.0, color="0.65", ls="--", lw=0.9)
            a.set_xlim(xlim0, xlim1)
            a.set_ylim(-0.02, 1.08)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            mark(a, half["period"], mark_info)
        ax_c.set_xlabel("Game clock (min)")
        if col == 0:
            ax_p.set_ylabel("P(t)")
            ax_q.set_ylabel("Q(t), w(t)")
            ax_c.set_ylabel("C(t)")
            ax_p.legend(frameon=False, loc="lower left")
            ax_q.legend(frameon=False, loc="upper right")
            ax_c.legend(frameon=False, loc="lower left")
    axes[0, 0].set_title("P1  first half (+ extra time)")
    axes[0, 1].set_title("P2  second half (+ extra time)")
    fig.suptitle(f"{name} — 5-min pass success with shot xG, {match_label}", y=0.98, fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)


def slug(name: str) -> str:
    return fold(name).replace(" ", "_")


def run_one(match_dir, game_id, label, needle, mark_info=None):
    pid, nick = find_player(match_dir, game_id, needle)
    inv = load_passes(match_dir, game_id, pid)
    shots = score_shots(match_dir, game_id, nick)
    halves = half_bundle(inv, shots)
    tag = f"{slug(nick)}_{game_id}"
    plot_success(halves, nick, label, OUT / f"pass_success_w5_{tag}.png", mark_info)
    plot_composite(halves, nick, label, OUT / f"pass_xg_w5_{tag}.png", mark_info)
    frames = []
    for key, half in halves.items():
        if half["curve"].empty:
            continue
        part = half["curve"].copy()
        part.insert(0, "period", half["period"])
        part.insert(0, "half", key)
        frames.append(part)
        print(
            f"  {nick} {label} P{half['period']}: {half['n_ok']}/{half['n']} "
            f"({half['overall_pct']:.1f}%)  shots={len(half['shots'])}",
            flush=True,
        )
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT / f"pass_xg_w5_{tag}.csv", index=False)
    print(f"  saved {tag}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for needle in FIVE:
        for spec in BOTH:
            jobs.append((spec["dir"], spec["game_id"], spec["label"], needle, None))
    for extra in EXTRAS:
        jobs.append((extra["dir"], extra["game_id"], extra["label"], extra["needle"], extra["mark"]))
    for i, job in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {job[3]} {job[2]}", flush=True)
        run_one(*job)


if __name__ == "__main__":
    main()
