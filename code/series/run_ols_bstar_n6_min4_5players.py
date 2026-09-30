"""N6_min4_dx20s b* A(t) for the 5 players on 3820 and 10516.

Same figures as the Modric 10506 folder in ols_bstar_angle_settings/N6_min4_dx20s/:
A(t) with the A↔θ mapping strip, example scatters (full match + origin, equal
aspect in b* units, angle diagram), fitted CSVs. Frozen global scales.

Usage (from the football workspace that holds match folders):
    python beyond_physical_fatigue/code/series/run_ols_bstar_n6_min4_5players.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
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

import match_player_analysis as mpa  # noqa: E402

OUT_ROOT = SERIES_DIR
BSTAR_PY = SERIES_DIR / "run_ols_bstar_angle_settings.py"
_spec = importlib.util.spec_from_file_location("ols_bstar_helpers", BSTAR_PY)
bstar = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(bstar)

Y_METRICS = ("distance_m", "peak_speed")
PLAYERS = ["Amrabat", "Modric", "Gvardiol", "Hakimi", "Perisic"]
MATCHES = [
    {"game_id": "3820", "when": "early", "label": "3820 MAR-CRO"},
    {"game_id": "10516", "when": "late", "label": "10516 CRO-MAR"},
]
SETTING = {"n_window": 6, "min_n": 4, "min_x_s": 20.0}
K_HIGH = 2.25
K_LOW = -1.5
MERGE_GAP_S = 1.25
S_X = bstar.GLOBAL_S_X
S_Y = bstar.GLOBAL_S_Y
N_EXAMPLES = bstar.N_EXAMPLES


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def find_pid(needle: str, when: str) -> tuple[str, str]:
    hits = mpa.find_player(needle, when)
    if hits is None or hits.empty:
        raise RuntimeError(f"No roster hit for {needle!r} in {when}")
    row = hits.iloc[0]
    return str(row["player_id"]), str(row.get("nickname") or needle)


def period_recovery(ev: pd.DataFrame) -> pd.DataFrame:
    out = ev.sort_values(["period", "start_time_s", "event_id"]).reset_index(drop=True)
    rec = np.full(len(out), np.nan)
    period = pd.to_numeric(out["period"], errors="coerce")
    start = pd.to_numeric(out["start_time_s"], errors="coerce")
    end = pd.to_numeric(out["end_time_s"], errors="coerce")
    for p in period.dropna().unique():
        idx = np.flatnonzero(period.to_numpy() == p)
        if len(idx) < 2:
            continue
        rec[idx[1:]] = start.to_numpy()[idx[1:]] - end.to_numpy()[idx[:-1]]
    rec[rec < 0] = np.nan
    out["recovery_before_s"] = rec
    return out


def load_sprints(needle: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    pid, nick = find_pid(needle, spec["when"])
    bundle = mpa.load_player_kinematics(spec["game_id"], pid, attach_vendor=False)
    atoms, thr = mpa.detect_relative_sprint_events(bundle, k_high=K_HIGH, k_low=K_LOW)
    merged = mpa.merge_close_sprint_events(bundle, atoms, min_gap_s=MERGE_GAP_S)
    merged = period_recovery(merged)
    merged["player"] = nick
    merged["player_needle"] = needle
    merged["player_id"] = pid
    merged["match_id"] = spec["game_id"]
    merged["match_label"] = spec["label"]
    meta = {
        "player": nick,
        "player_needle": needle,
        "player_id": pid,
        "match_id": spec["game_id"],
        "match_label": spec["label"],
        "n_sprints": int(len(merged)),
        "n_usable": int(
            np.isfinite(pd.to_numeric(merged["recovery_before_s"], errors="coerce")).sum()
        ),
        "k_high": K_HIGH,
        "k_low": K_LOW,
        "merge_gap_s": MERGE_GAP_S,
        "setting": SETTING,
        "s_x": S_X,
        "s_y": S_Y,
        "scale_source": "global_p90_5players_3820_10516",
    }
    return merged, meta


def plot_player_grid(all_fits: list[dict], ycol: str, dest: Path) -> None:
    players = PLAYERS
    matches = [m["game_id"] for m in MATCHES]
    fig, axes = plt.subplots(
        len(players),
        len(matches),
        figsize=(8.0 * len(matches), 2.55 * len(players)),
        sharex=False,
        sharey=True,
        layout="constrained",
    )
    lookup = {(r["player_needle"], r["match_id"], r["ycol"]): r for r in all_fits}
    for i, name in enumerate(players):
        for j, gid in enumerate(matches):
            ax = axes[i, j]
            rec = lookup.get((name, gid, ycol))
            if rec is None:
                ax.set_axis_off()
                continue
            fitted = rec["fitted"]
            for p in mpa.periods_in(fitted) or []:
                sub = fitted.loc[pd.to_numeric(fitted["period"], errors="coerce") == int(p)]
                tt = pd.to_numeric(sub["start_clock_min"], errors="coerce")
                zz = pd.to_numeric(sub["roll_A"], errors="coerce")
                m = np.isfinite(tt) & np.isfinite(zz)
                if m.any():
                    ax.plot(tt[m], zz[m], lw=1.1, marker="o", ms=3.2, label=f"P{int(p)}")
            ax.axhline(0.5, color="#b03a2e", ls="--", lw=0.9)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.22)
            if i == 0:
                ax.set_title(rec["match_label"], fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
            if i == len(players) - 1:
                ax.set_xlabel("period clock (min)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left", ncol=2)
    ylab = mpa.SPRINT_Y_METRICS[ycol][1]
    fig.suptitle(
        f"{ylab}  |  A(t) from b*  |  N6 min4 dx>=20s  |  s_x={S_X:g}  s_y={S_Y[ycol]:g}  |  "
        f"5 players, both matches",
        fontsize=11,
    )
    bstar.add_A_mapping_cbar(fig, axes)
    bstar.savefig_retry(fig, dest)
    plt.close(fig)
    print("saved", dest)


def run_one(merged: pd.DataFrame, meta: dict, dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "settings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    merged.to_csv(dest / f"{slug(meta['player_needle'])}_{meta['match_id']}_sprints.csv", index=False)
    n_window, min_n, min_x = SETTING["n_window"], SETTING["min_n"], SETTING["min_x_s"]
    ev0 = merged.sort_values(["period", "start_time_s", "event_id"]).reset_index(drop=True)
    player = meta["player"]
    label = meta["match_label"]
    prefix = f"{slug(meta['player_needle'])}_{meta['match_id']}"
    rows = []
    for ycol in Y_METRICS:
        ylab = mpa.SPRINT_Y_METRICS[ycol][1]
        fitted, fit_meta = mpa.add_rolling_line_fit(
            ev0,
            ycol,
            n_window=n_window,
            min_n=min_n,
            min_x_range_s=min_x,
            method="ols",
            s_x=S_X,
            s_y=S_Y[ycol],
        )
        rec = bstar.summarize(fitted, ycol, SETTING, fit_meta)
        rec.update(
            {
                "player": player,
                "player_needle": meta["player_needle"],
                "match_id": meta["match_id"],
                "match_label": label,
            }
        )
        rows.append(rec)
        n_fit = int(np.isfinite(pd.to_numeric(fitted["roll_A"], errors="coerce")).sum())
        print(f"  {ycol}: n_fit={n_fit}  s_x={fit_meta['s_x']:.1f}  s_y={fit_meta['s_y']:.3f}")
        if n_fit == 0:
            continue
        fitted.to_csv(dest / f"{prefix}_ols_bstar_A_{ycol}.csv", index=False)
        bstar.plot_A_vs_time(
            fitted,
            ylab,
            dest / f"{prefix}_ols_bstar_A_{ycol}.png",
            n_window,
            min_n,
            min_x,
            float(fit_meta["s_x"]),
            float(fit_meta["s_y"]),
            player=player,
            match_label=label,
            sub_clock=None,
        )
        for i in bstar.pick_examples(fitted, n=N_EXAMPLES):
            a = float(fitted.loc[i, "roll_a"])
            b = float(fitted.loc[i, "roll_b"])
            b_star = float(fitted.loc[i, "roll_b_star"])
            theta_deg = float(fitted.loc[i, "roll_theta_deg"])
            a_val = float(fitted.loc[i, "roll_A"])
            if not all(np.isfinite(v) for v in (a, b, b_star, theta_deg, a_val)):
                continue
            safe = bstar.clock_label(fitted.loc[i]).replace(":", "")
            bstar.plot_example(
                fitted,
                ycol,
                ylab,
                int(i),
                a,
                b,
                b_star,
                theta_deg,
                a_val,
                n_window,
                f"{player}  |  {label}  |  {ylab}  |  N6_min4_dx20s  |  b*",
                dest / f"{prefix}_ols_bstar_example_{ycol}_{safe}.png",
                s_x=float(fit_meta["s_x"]),
                s_y=float(fit_meta["s_y"]),
            )
        rec["_fitted"] = fitted
        rec["_ycol"] = ycol
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("out", OUT_ROOT)
    print(f"setting N6_min4_dx20s  global s_x={S_X:g}  s_y={S_Y}")
    (OUT_ROOT / "global_scales.json").write_text(
        json.dumps(
            {
                "s_x": S_X,
                "s_y": S_Y,
                "setting": SETTING,
                "source": "90th percentile over 585 usable sprints, 5 players x 3820 + 10516",
                "players": PLAYERS,
                "matches": [m["game_id"] for m in MATCHES],
                "chosen": {"s_x": 200.0, "s_y_distance_m": 42.0, "s_y_peak_speed": 7.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_rows = []
    all_fits: list[dict] = []
    for spec in MATCHES:
        for name in PLAYERS:
            print(f"\n=== {name}  {spec['label']}", flush=True)
            try:
                merged, meta = load_sprints(name, spec)
            except Exception as exc:
                print(f"  SKIP: {exc}")
                continue
            dest = OUT_ROOT / f"{slug(name)}_{spec['game_id']}"
            print(f"  n_sprints={meta['n_sprints']}  n_with_recovery={meta['n_usable']}")
            for rec in run_one(merged, meta, dest):
                fitted = rec.pop("_fitted", None)
                ycol = rec.pop("_ycol", rec.get("y_metric"))
                summary_rows.append(rec)
                if fitted is not None:
                    all_fits.append(
                        {
                            "player_needle": name,
                            "match_id": spec["game_id"],
                            "match_label": spec["label"],
                            "ycol": ycol,
                            "fitted": fitted,
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_ROOT / "settings_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n", summary.to_string(index=False))
    print("saved", summary_path)

    for ycol in Y_METRICS:
        plot_player_grid(all_fits, ycol, OUT_ROOT / f"comparison_A_{ycol}.png")

    lines = [
        "# OLS b* A(t) — N6_min4_dx20s, 5 players × 2 matches",
        "",
        "Chosen window: **N=6**, **min_n=4**, **Δx ≥ 20 s**, OLS.",
        "",
        "    b* = b * s_x / s_y",
        "    theta = arctan(b*)",
        "    A = (theta + pi/2) / pi     # (0, 1), A=0.5 is flat",
        "",
        "Frozen global scales (same as `ols_bstar_angle_settings/`):",
        "`s_x = 200 s`, `s_y(distance) = 42 m`, `s_y(peak speed) = 7.0 m/s`.",
        "See `global_scales.json`.",
        "",
        "Players: Amrabat, Modric, Gvardiol, Hakimi, Perisic.",
        "Matches: 3820 MAR-CRO and 10516 CRO-MAR.",
        "",
        "Each player/match folder matches the Modric 10506 `N6_min4_dx20s` set:",
        "- `*_ols_bstar_A_{distance_m,peak_speed}.png` — A(t) by period, light A↔θ strip on the right",
        "- `*_ols_bstar_example_*` — window scatter in (x/s_x, y/s_y), full match + origin, equal aspect, angle diagram",
        "- fitted CSVs and the sprint table",
        "",
        "See `settings_summary.csv` and `comparison_A_{distance_m,peak_speed}.png`.",
        "",
        "| player | match | metric | n_fit | median A | IQR | min | max | median θ* |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.player} | {r.match_id} | {r.y_metric} | {r.n_fit} | "
            f"{r.median_A:.3f} | {r.iqr_A:.3f} | {r.min_A:.3f} | {r.max_A:.3f} | "
            f"{r.median_theta_star_deg:.2f} |"
        )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", OUT_ROOT / "README.md")


if __name__ == "__main__":
    main()
