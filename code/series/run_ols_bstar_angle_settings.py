"""Sweep OLS b*-scaled line-angle A(t) and save each run in its own folder.

Same windows as the raw-angle sweep, but using the original scaled slope:

    b* = b * s_x / s_y
    theta = arctan(b*)
    A = (theta + pi/2) / pi     in (0, 1),  A=0.5 is flat

s_x and s_y are frozen **global** 90th-percentile scales from the 5-player
x 2-match sprint cloud (Amrabat, Modric, Gvardiol, Hakimi, Perisic on
3820 and 10516), not the single-match 90ths.

Usage (from the football workspace that holds match folders):
    python -m beyond_physical_fatigue.code.series.run_ols_bstar_angle_settings
    # or: python beyond_physical_fatigue/code/series/run_ols_bstar_angle_settings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Arc, FancyArrowPatch

# Light A ↔ θ strip: A=0 → −90°, A=0.5 → 0° (flat), A=1 → +90°.
A_MAP_CMAP = LinearSegmentedColormap.from_list(
    "A_map_light",
    [(0.0, "#9ecae1"), (0.5, "#f7f7f7"), (1.0, "#fcbba1")],
)
A_MAP_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
A_MAP_LABELS = ("0  (−90°)", "0.25 (−45°)", "0.50  (0°)", "0.75 (+45°)", "1  (+90°)")


def add_A_mapping_cbar(fig, axes) -> None:
    """Light color strip on the right of the time figure: A maps linearly to θ."""
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=A_MAP_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", pad=0.018, fraction=0.028, aspect=28)
    cbar.set_ticks(list(A_MAP_TICKS))
    cbar.set_ticklabels(list(A_MAP_LABELS))
    cbar.ax.tick_params(labelsize=7.5, colors="0.38", length=2.5)
    cbar.set_label("A  ↔  θ", color="0.38", fontsize=8)
    cbar.outline.set_edgecolor("0.62")
    cbar.outline.set_linewidth(0.6)

SERIES_DIR = Path(__file__).resolve().parent
PKG_DIR = SERIES_DIR.parents[1]
sys.path.insert(0, str(PKG_DIR.parent))
from beyond_physical_fatigue.code.paths import ensure_lib_on_path, football_root  # noqa: E402

ensure_lib_on_path()
ROOT = football_root()

import match_player_analysis as mpa  # noqa: E402

OUT_ROOT = SERIES_DIR
SPRINT_CSV = ROOT / "cognitive_metrics" / "main_figures_nb_outputs" / "luka_modric_10506_sprints.csv"
Y_METRICS = ("distance_m", "peak_speed")
N_EXAMPLES = 3

# Global 90th percentiles over 585 usable sprints (5 players x 3820 + 10516).
# Raw p90: recovery 195.3 s, distance 41.7 m, peak speed 7.00 m/s.
GLOBAL_S_X = 200.0
GLOBAL_S_Y = {"distance_m": 42.0, "peak_speed": 7.0}

SETTINGS = [
    {"n_window": 5, "min_n": 3, "min_x_s": 20.0},
    {"n_window": 5, "min_n": 4, "min_x_s": 20.0},
    {"n_window": 6, "min_n": 3, "min_x_s": 20.0},
    {"n_window": 6, "min_n": 4, "min_x_s": 20.0},
    {"n_window": 6, "min_n": 5, "min_x_s": 20.0},
    {"n_window": 8, "min_n": 3, "min_x_s": 20.0},
    {"n_window": 8, "min_n": 5, "min_x_s": 20.0},
    {"n_window": 10, "min_n": 5, "min_x_s": 20.0},
    {"n_window": 5, "min_n": 3, "min_x_s": 15.0},
    {"n_window": 5, "min_n": 3, "min_x_s": 30.0},
]


def savefig_retry(fig, dest: Path, tries: int = 8) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"_{dest.stem}_tmp.png")
    last = None
    for i in range(tries):
        try:
            fig.savefig(tmp, dpi=140, format="png")
            tmp.replace(dest)
            return
        except OSError as exc:
            last = exc
            plt.pause(0.25 * (i + 1))
    if tmp.exists():
        try:
            tmp.replace(dest)
            return
        except OSError as exc:
            last = exc
    raise last


def setting_tag(s: dict) -> str:
    dx = int(s["min_x_s"]) if float(s["min_x_s"]).is_integer() else s["min_x_s"]
    return f"N{s['n_window']}_min{s['min_n']}_dx{dx}s"


def usable_mask(ev: pd.DataFrame, ycol: str) -> pd.Series:
    x = pd.to_numeric(ev["recovery_before_s"], errors="coerce")
    y = pd.to_numeric(ev[ycol], errors="coerce")
    p = pd.to_numeric(ev["period"], errors="coerce")
    return np.isfinite(x) & (x > 0) & np.isfinite(y) & np.isfinite(p)


def window_iloc(ev: pd.DataFrame, end_i: int, ycol: str, n_window: int) -> list[int]:
    usable = usable_mask(ev, ycol).to_numpy()
    period = pd.to_numeric(ev["period"], errors="coerce").to_numpy()
    if end_i < 0 or end_i >= len(ev) or not np.isfinite(period[end_i]):
        return []
    p = int(period[end_i])
    loc = [j for j in np.flatnonzero(usable) if int(period[j]) == p]
    if end_i not in loc:
        return []
    k = loc.index(end_i)
    w0 = max(0, k - (int(n_window) - 1))
    return loc[w0 : k + 1]


def pick_examples(fitted: pd.DataFrame, n: int = 3) -> list[int]:
    ok = fitted.loc[np.isfinite(fitted["roll_A"])]
    if ok.empty:
        return []
    if len(ok) <= n:
        return [int(i) for i in ok.index]
    s = ok.sort_values("roll_A")
    picks = [s.index[0], s.index[len(s) // 2], s.index[-1]]
    return list(dict.fromkeys(int(i) for i in picks))


def clock_label(row) -> str:
    m = float(pd.to_numeric(row.get("start_clock_min"), errors="coerce"))
    if not np.isfinite(m):
        return "na"
    mm = int(np.floor(m))
    ss = int(round((m - mm) * 60.0))
    if ss == 60:
        mm += 1
        ss = 0
    return f"{mm}:{ss:02d}"


def draw_bstar_diagram(ax, b_star: float, theta_deg: float, a_val: float) -> None:
    """Equal-aspect view of the unit-free slope b* vs +x."""
    hyp = float(np.hypot(1.0, b_star))
    if not np.isfinite(hyp) or hyp == 0:
        ax.set_title("no finite b*")
        return
    ux, uy = 1.0 / hyp, float(b_star) / hyp
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.axvline(0.0, color="0.85", lw=0.6)
    ax.add_patch(
        FancyArrowPatch((0.0, 0.0), (1.15, 0.0), arrowstyle="-|>", mutation_scale=12, color="0.25", lw=1.4)
    )
    ax.plot([0.0, 1.2 * ux], [0.0, 1.2 * uy], color="#c0392b", lw=2.2, zorder=3)
    ax.add_patch(
        FancyArrowPatch(
            (0.0, 0.0),
            (1.15 * ux, 1.15 * uy),
            arrowstyle="-|>",
            mutation_scale=12,
            color="#c0392b",
            lw=1.6,
        )
    )
    t1, t2 = (float(theta_deg), 0.0) if theta_deg < 0 else (0.0, float(theta_deg))
    if np.isfinite(theta_deg) and abs(theta_deg) > 0.15:
        ax.add_patch(Arc((0.0, 0.0), 0.72, 0.72, angle=0.0, theta1=t1, theta2=t2, color="#1f4e79", lw=1.6))
    ax.text(
        0.38,
        0.14 * np.sign(b_star if b_star != 0 else 1.0),
        f"theta*={theta_deg:.1f} deg\nA={a_val:.3f}",
        color="#1f4e79",
        fontsize=10,
    )
    ax.text(1.18, -0.08, "+x", fontsize=10, ha="left", va="top")
    ax.set_xlim(-0.15, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel("run (scaled by s_x)")
    ax.set_ylabel("rise (scaled by s_y)")
    ax.set_title(f"b* angle with +x   A=(theta+pi/2)/pi={a_val:.3f}")
    ax.grid(True, alpha=0.2)


def apply_true_slope_aspect(ax, aspect: float = 1.0) -> None:
    if not np.isfinite(aspect) or aspect <= 0:
        aspect = 1.0
    ax.set_aspect(float(aspect), adjustable="box", anchor="SW")


def match_origin_limits(x_all, y_all, *, pad_frac=0.04):
    """Axis box: origin plus the full match range of x and of y."""
    xf = np.asarray(x_all, dtype=float)
    yf = np.asarray(y_all, dtype=float)
    xf = xf[np.isfinite(xf)]
    yf = yf[np.isfinite(yf)]
    x0 = min(0.0, float(xf.min()) if xf.size else 0.0)
    x1 = max(0.0, float(xf.max()) if xf.size else 1.0)
    y0 = min(0.0, float(yf.min()) if yf.size else 0.0)
    y1 = max(0.0, float(yf.max()) if yf.size else 1.0)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    xpad = max(pad_frac * (x1 - x0), 1e-6)
    ypad = max(pad_frac * (y1 - y0), 1e-6)
    return x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad


def scatter_figure_size(x0, x1, y0, y1, *, angle_in=3.3, min_h=1.8, max_w=24.0):
    xs = max(float(x1 - x0), 1e-6)
    ys = max(float(y1 - y0), 1e-6)
    h = float(min_h)
    w_left = h * xs / ys
    if w_left > max_w:
        w_left = float(max_w)
        h = max(1.15, w_left * ys / xs)
    else:
        h_nice = min(float(angle_in), float(max_w) * ys / xs)
        h = max(h, h_nice)
        w_left = h * xs / ys
    fig_w = w_left + float(angle_in) + 1.5
    fig_h = max(h, float(angle_in)) + 1.15
    return fig_w, fig_h, [w_left, float(angle_in)]


def apply_match_origin_equal_aspect(ax, x0, x1, y0, y1) -> None:
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    xs = max(x1 - x0, 1e-6)
    ys = max(y1 - y0, 1e-6)
    if ys / xs >= 0.16:
        apply_true_slope_aspect(ax, 1.0)
    ax.set_autoscale_on(False)
    yticks = [t for t in ax.get_yticks() if y0 <= t <= y1]
    if 0.0 not in yticks:
        yticks = sorted(yticks + [0.0])
    if yticks:
        ax.set_yticks(yticks)


def plot_example(
    ev,
    ycol,
    ylab,
    end_i,
    a,
    b,
    b_star,
    theta_deg,
    a_val,
    n_window,
    title,
    dest: Path,
    s_x: float,
    s_y: float,
) -> None:
    win = window_iloc(ev, end_i, ycol, n_window)
    x = pd.to_numeric(ev["recovery_before_s"], errors="coerce")
    y = pd.to_numeric(ev[ycol], errors="coerce")
    t = pd.to_numeric(ev["start_clock_min"], errors="coerce")
    p = int(pd.to_numeric(ev.loc[end_i, "period"], errors="coerce"))
    clock = clock_label(ev.loc[end_i])
    sx = float(s_x) if np.isfinite(s_x) and s_x > 0 else 1.0
    sy = float(s_y) if np.isfinite(s_y) and s_y > 0 else 1.0
    xs = x / sx
    ys = y / sy
    x0, x1, y0, y1 = match_origin_limits(xs, ys)
    if (y1 - y0) / max(x1 - x0, 1e-6) >= 0.16:
        fig_w, fig_h, ratios = scatter_figure_size(x0, x1, y0, y1)
    else:
        fig_w, fig_h, ratios = 13.4, 5.4, [2.15, 1.0]
    fig, (ax, ax_ang) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), layout="constrained", width_ratios=ratios
    )
    ok = np.isfinite(xs.to_numpy(dtype=float)) & np.isfinite(ys.to_numpy(dtype=float))
    ax.scatter(xs[ok], ys[ok], c="0.78", s=24, zorder=1, label="all match", clip_on=True)
    if win:
        ax.scatter(
            xs.iloc[win],
            ys.iloc[win],
            c=t.iloc[win],
            cmap="plasma",
            s=58,
            zorder=3,
            edgecolors="0.15",
            linewidths=0.4,
            label=f"window n={len(win)}",
        )
        ax.scatter(
            [xs.iloc[end_i]],
            [ys.iloc[end_i]],
            s=110,
            facecolors="none",
            edgecolors="#111",
            linewidths=1.4,
            zorder=4,
            label=f"current {clock}",
        )
    xline = np.linspace(x0, x1, 80)
    ax.plot(xline, (a / sy) + b_star * xline, color="#111", lw=1.7, zorder=2, label=f"b*={b_star:.3f}", clip_on=True)
    ax.axhline(0.0, color="0.45", lw=0.8, zorder=0)
    ax.axvline(0.0, color="0.45", lw=0.8, zorder=0)
    ax.plot(0.0, 0.0, marker="+", color="#111", ms=9, mew=1.1, zorder=5)
    apply_match_origin_equal_aspect(ax, x0, x1, y0, y1)
    ax.set_xlabel("recovery / s_x")
    ax.set_ylabel(f"{ylab} / s_y")
    ax.set_title(f"P{p}  window ending {clock}   (equal aspect in b* units, full match + origin)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    draw_bstar_diagram(ax_ang, b_star, theta_deg, a_val)
    fig.suptitle(title, fontsize=11)
    savefig_retry(fig, dest)
    plt.close(fig)


def plot_A_vs_time(
    fitted,
    ylab,
    dest: Path,
    n_window: int,
    min_n: int,
    min_x: float,
    s_x: float,
    s_y: float,
    *,
    player: str = "Luka Modric",
    match_label: str = "10506 JPN-CRO",
    sub_clock: float | None = 99.0,
    sub_label: str = "sub 99:00",
) -> None:
    pers = list(mpa.periods_in(fitted)) or [1, 2]
    n = max(len(pers), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 1.1, 3.55), sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, pers):
        sub = fitted.loc[pd.to_numeric(fitted["period"], errors="coerce") == int(p)]
        tt = pd.to_numeric(sub["start_clock_min"], errors="coerce")
        zz = pd.to_numeric(sub["roll_A"], errors="coerce")
        m = np.isfinite(tt) & np.isfinite(zz)
        if m.any():
            ax.plot(tt[m], zz[m], color="#1f4e79", lw=1.25, marker="o", ms=4.2)
        ax.axhline(0.5, color="#b03a2e", ls="--", lw=1.0, label="A=0.5  (flat)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"P{int(p)}  |  n_fit={int(m.sum())}", fontsize=10)
        ax.set_xlabel("period clock (min)")
        if ax is axes[0]:
            ax.set_ylabel("A = (theta + pi/2) / pi")
            ax.legend(fontsize=8, loc="lower left")
        if sub_clock is not None and int(p) == 3:
            mpa.mark_substitution_on_ax(ax, float(sub_clock), sub_label, vline=True)
    add_A_mapping_cbar(fig, axes)
    fig.suptitle(
        f"{player}  |  {match_label}  |  OLS  |  A(t) from b*  |  {ylab}  |  "
        f"N={n_window} (min {min_n}, dx>={min_x:g}s)  |  s_x={s_x:.1f}s  s_y={s_y:.2f}",
        fontsize=10,
    )
    savefig_retry(fig, dest)
    plt.close(fig)


def smoothness(val: pd.Series, t: pd.Series, period: pd.Series) -> float:
    diffs = []
    for _, g in pd.DataFrame({"z": val, "t": t, "period": period}).groupby("period"):
        g = g.sort_values("t")
        z = pd.to_numeric(g["z"], errors="coerce").to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        if len(z) >= 2:
            diffs.append(np.abs(np.diff(z)))
    if not diffs:
        return float("nan")
    return float(np.mean(np.concatenate(diffs)))


def summarize(fitted: pd.DataFrame, ycol: str, setting: dict, meta: dict) -> dict:
    a = pd.to_numeric(fitted["roll_A"], errors="coerce")
    ok = np.isfinite(a)
    t = pd.to_numeric(fitted["start_clock_min"], errors="coerce")
    period = pd.to_numeric(fitted["period"], errors="coerce")
    z = a.loc[ok]
    th = pd.to_numeric(fitted["roll_theta_deg"], errors="coerce").loc[ok]
    return {
        "folder": setting_tag(setting),
        "n_window": setting["n_window"],
        "min_n": setting["min_n"],
        "min_x_s": setting["min_x_s"],
        "y_metric": ycol,
        "n_usable": int(usable_mask(fitted, ycol).sum()),
        "n_fit": int(ok.sum()),
        "s_x": meta.get("s_x", float("nan")),
        "s_y": meta.get("s_y", float("nan")),
        "median_A": float(z.median()) if len(z) else float("nan"),
        "iqr_A": float(z.quantile(0.75) - z.quantile(0.25)) if len(z) else float("nan"),
        "min_A": float(z.min()) if len(z) else float("nan"),
        "max_A": float(z.max()) if len(z) else float("nan"),
        "median_theta_star_deg": float(th.median()) if len(th) else float("nan"),
        "mean_abs_step_A": smoothness(a, t, period),
        "n_cross_half": int(np.sum(np.diff(np.sign(z.to_numpy() - 0.5)) != 0)) if len(z) >= 2 else 0,
    }


def run_one(merged: pd.DataFrame, setting: dict, dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        **setting,
        "s_x": GLOBAL_S_X,
        "s_y": GLOBAL_S_Y,
        "scale_source": "global_p90_5players_3820_10516",
    }
    (dest / "settings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    n_window, min_n, min_x = setting["n_window"], setting["min_n"], setting["min_x_s"]
    ev0 = merged.sort_values(["period", "start_time_s", "event_id"]).reset_index(drop=True)
    for ycol in Y_METRICS:
        ylab = mpa.SPRINT_Y_METRICS[ycol][1]
        fitted, meta = mpa.add_rolling_line_fit(
            ev0,
            ycol,
            n_window=n_window,
            min_n=min_n,
            min_x_range_s=min_x,
            method="ols",
            s_x=GLOBAL_S_X,
            s_y=GLOBAL_S_Y[ycol],
        )
        rows.append(summarize(fitted, ycol, setting, meta))
        n_fit = int(np.isfinite(pd.to_numeric(fitted["roll_A"], errors="coerce")).sum())
        print(f"  {ycol}: n_fit={n_fit}  s_x={meta['s_x']:.1f}  s_y={meta['s_y']:.3f}")
        if n_fit == 0:
            continue
        fitted.to_csv(dest / f"luka_modric_10506_ols_bstar_A_{ycol}.csv", index=False)
        plot_A_vs_time(
            fitted,
            ylab,
            dest / f"luka_modric_10506_ols_bstar_A_{ycol}.png",
            n_window,
            min_n,
            min_x,
            float(meta["s_x"]),
            float(meta["s_y"]),
        )
        for i in pick_examples(fitted, n=N_EXAMPLES):
            a = float(fitted.loc[i, "roll_a"])
            b = float(fitted.loc[i, "roll_b"])
            b_star = float(fitted.loc[i, "roll_b_star"])
            theta_deg = float(fitted.loc[i, "roll_theta_deg"])
            a_val = float(fitted.loc[i, "roll_A"])
            if not all(np.isfinite(v) for v in (a, b, b_star, theta_deg, a_val)):
                continue
            safe = clock_label(fitted.loc[i]).replace(":", "")
            plot_example(
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
                f"Luka Modric  |  10506 JPN-CRO  |  {ylab}  |  {setting_tag(setting)}  |  b*",
                dest / f"luka_modric_10506_ols_bstar_example_{ycol}_{safe}.png",
                s_x=float(meta["s_x"]),
                s_y=float(meta["s_y"]),
            )
    return rows


def plot_comparison(all_fits: dict[str, dict[str, pd.DataFrame]]) -> None:
    tags = list(all_fits)
    cmap = plt.cm.tab10
    for ycol in Y_METRICS:
        ylab = mpa.SPRINT_Y_METRICS[ycol][1]
        pers = [1, 2, 3]
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), sharey=True, layout="constrained")
        for ax, p in zip(axes, pers):
            for i, tag in enumerate(tags):
                df = all_fits[tag].get(ycol)
                if df is None:
                    continue
                sub = df.loc[pd.to_numeric(df["period"], errors="coerce") == int(p)]
                tt = pd.to_numeric(sub["start_clock_min"], errors="coerce")
                zz = pd.to_numeric(sub["roll_A"], errors="coerce")
                m = np.isfinite(tt) & np.isfinite(zz)
                if not m.any():
                    continue
                ax.plot(tt[m], zz[m], lw=1.15, marker="o", ms=3.2, color=cmap(i % 10), label=tag)
            ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"P{p}")
            ax.set_xlabel("period clock (min)")
            if p == 1:
                ax.set_ylabel("A = (theta + pi/2) / pi")
                ax.legend(fontsize=7, loc="best", ncol=2)
        add_A_mapping_cbar(fig, axes)
        fig.suptitle(f"b* scaled A(t) overlay  |  10506  |  {ylab}")
        savefig_retry(fig, OUT_ROOT / f"comparison_A_{ycol}.png")
        plt.close(fig)
        print("saved", OUT_ROOT / f"comparison_A_{ycol}.png")


def main() -> None:
    if not SPRINT_CSV.exists():
        raise SystemExit(f"missing sprint table {SPRINT_CSV} — run the sprint cell first")
    merged = pd.read_csv(SPRINT_CSV)
    print("sprints", SPRINT_CSV, "n=", len(merged))
    print("out", OUT_ROOT)
    print(f"global scales  s_x={GLOBAL_S_X:g} s   s_y={GLOBAL_S_Y}")
    (OUT_ROOT / "global_scales.json").write_text(
        json.dumps(
            {
                "s_x": GLOBAL_S_X,
                "s_y": GLOBAL_S_Y,
                "source": "90th percentile over 585 usable sprints, 5 players x 3820 + 10516",
                "players": ["Amrabat", "Modric", "Gvardiol", "Hakimi", "Perisic"],
                "matches": ["3820", "10516"],
                "raw_p90": {"recovery_before_s": 195.3, "distance_m": 41.7, "peak_speed": 7.00},
                "chosen": {"s_x": 200.0, "s_y_distance_m": 42.0, "s_y_peak_speed": 7.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_rows = []
    all_fits: dict[str, dict[str, pd.DataFrame]] = {}
    for setting in SETTINGS:
        tag = setting_tag(setting)
        dest = OUT_ROOT / tag
        print("\n===", tag, flush=True)
        summary_rows.extend(run_one(merged, setting, dest))
        all_fits[tag] = {}
        for ycol in Y_METRICS:
            csv = dest / f"luka_modric_10506_ols_bstar_A_{ycol}.csv"
            if csv.exists():
                all_fits[tag][ycol] = pd.read_csv(csv)

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_ROOT / "settings_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n", summary.to_string(index=False))
    print("saved", summary_path)
    plot_comparison(all_fits)

    lines = [
        "# OLS b*-scaled line-angle A(t) settings",
        "",
        "Same window grid as `ols_raw_angle_settings/`, but the curve is the original scaled angle:",
        "",
        "    b* = b * s_x / s_y",
        "    theta = arctan(b*)",
        "    A = (theta + pi/2) / pi",
        "",
        "A=0.5 is a flat line. Range is (0, 1).",
        "",
        "## Frozen global scales",
        "",
        "Pooled **585** usable sprints (recovery $x>0$) from Amrabat, Modric, Gvardiol,",
        "Hakimi, Perisic on **3820** and **10516** (the pairing where all five appear).",
        "",
        "| quantity | raw 90th | per-match 90ths | frozen scale |",
        "|---|---:|---|---:|",
        "| recovery $s_x$ | 195.3 s | 177–262 s | **200 s** |",
        "| distance $s_y$ | 41.7 m | 31–46 m | **42 m** |",
        "| peak speed $s_y$ | 7.00 m/s | 6.1–7.8 m/s | **7.0 m/s** |",
        "",
        "There are **two $s_y$** values because distance and peak speed have different units.",
        "One pair $(s_x,s_y)$ cannot serve both clouds. $s_x=200$ is shared.",
        "These replace the old per-player–match 90ths. See `global_scales.json`.",
        "",
        "The official notebook $A(t)$ cell still uses that match table's 90ths",
        "(10506 Modric: $s_x\\approx208$ s, $s_y\\approx47$ m / $7.1$ m/s).",
        "",
        "Left example scatter is `(x/s_x, y/s_y)` with equal aspect over the **full match** cloud and the **origin**, so the visual slope is arctan(b*).",
        "",
        "Each `A(t)` figure (and the comparison overlays) has a **light color strip** on the",
        "right: A = 0 → −90°, A = 0.5 → 0° (flat), A = 1 → +90°.",
        "",
        "The chosen window **N6_min4_dx20s** applied to the 5 players on 3820 + 10516",
        "is in `ols_bstar_N6_min4_dx20s/` (same A(t), mapping strip, and example scatters).",
        "",
        "See `settings_summary.csv` and `comparison_A_{distance_m,peak_speed}.png`.",
        "",
        "| folder | N | min_n | dx_s | metric | n_fit | median A | IQR | min | max | mean |dA| |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| `{r.folder}` | {r.n_window} | {r.min_n} | {r.min_x_s:g} | {r.y_metric} | "
            f"{r.n_fit} | {r.median_A:.3f} | {r.iqr_A:.3f} | {r.min_A:.3f} | "
            f"{r.max_A:.3f} | {r.mean_abs_step_A:.3f} |"
        )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", OUT_ROOT / "README.md")


if __name__ == "__main__":
    main()
