"""
RSP roll-10 Φ from RAW (unsmoothed) homePlayers / awayPlayers.

Smoke:  python run_rsp_raw_roll10.py --smoke
Full:   python run_rsp_raw_roll10.py
One:    python run_rsp_raw_roll10.py --job gvardiol_10516
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

SERIES_DIR = Path(__file__).resolve().parent
CODE_DIR = SERIES_DIR.parent
PKG_DIR = CODE_DIR.parent
sys.path.insert(0, str(PKG_DIR.parent))
from beyond_physical_fatigue.code.paths import (  # noqa: E402
    ensure_lib_on_path,
    football_root,
    package_profiles_csv,
)

ensure_lib_on_path()
ROOT = football_root()

import match_player_analysis as mpa  # noqa: E402
import run_batch_rsp as rsp  # noqa: E402
import wc_position_profiles as wcp  # noqa: E402

OUT_ROOT = SERIES_DIR
_pkg_profiles = package_profiles_csv()
_ws_profiles = ROOT / "kinematic_profiles" / "wc_position_profiles.csv"
CSV_PROFILES = _ws_profiles if _ws_profiles.exists() else _pkg_profiles
ROLL_W = rsp.ROLL_W

# Clock marks for non-MAR-CRO jobs (drawn on match-time axes).
JOBS = [
    {
        "key": "amrabat_3820",
        "needles": ("Amrabat",),
        "match_id": "3820",
        "dir": ROOT / "3820_MAR_CRO",
        "label": "early 3820 MAR-CRO",
        "when": "early",
    },
    {
        "key": "amrabat_10516",
        "needles": ("Amrabat",),
        "match_id": "10516",
        "dir": ROOT / "10516_CRO_MAR",
        "label": "late 10516 CRO-MAR",
        "when": "late",
    },
    {
        "key": "hakimi_3820",
        "needles": ("Hakimi",),
        "match_id": "3820",
        "dir": ROOT / "3820_MAR_CRO",
        "label": "early 3820 MAR-CRO",
        "when": "early",
    },
    {
        "key": "hakimi_10516",
        "needles": ("Hakimi",),
        "match_id": "10516",
        "dir": ROOT / "10516_CRO_MAR",
        "label": "late 10516 CRO-MAR",
        "when": "late",
    },
    {
        "key": "modric_3820",
        "needles": ("Modric", "Modrić"),
        "match_id": "3820",
        "dir": ROOT / "3820_MAR_CRO",
        "label": "early 3820 MAR-CRO",
        "when": "early",
    },
    {
        "key": "modric_10516",
        "needles": ("Modric", "Modrić"),
        "match_id": "10516",
        "dir": ROOT / "10516_CRO_MAR",
        "label": "late 10516 CRO-MAR",
        "when": "late",
    },
    {
        "key": "modric_10506",
        "needles": ("Modric", "Modrić"),
        "match_id": "10506",
        "dir": ROOT / "10506_JPN_CRO",
        "label": "mid 10506 JPN-CRO",
        "when": "mid",
        "clock_marks": [{"clock_min": 99.0, "period": 3, "label": "sub 99:00"}],
    },
    {
        "key": "perisic_3820",
        "needles": ("Perisic", "Perišić"),
        "match_id": "3820",
        "dir": ROOT / "3820_MAR_CRO",
        "label": "early 3820 MAR-CRO",
        "when": "early",
    },
    {
        "key": "perisic_10516",
        "needles": ("Perisic", "Perišić"),
        "match_id": "10516",
        "dir": ROOT / "10516_CRO_MAR",
        "label": "late 10516 CRO-MAR",
        "when": "late",
    },
    {
        "key": "gvardiol_3820",
        "needles": ("Gvardiol",),
        "match_id": "3820",
        "dir": ROOT / "3820_MAR_CRO",
        "label": "early 3820 MAR-CRO",
        "when": "early",
    },
    {
        "key": "gvardiol_10516",
        "needles": ("Gvardiol",),
        "match_id": "10516",
        "dir": ROOT / "10516_CRO_MAR",
        "label": "late 10516 CRO-MAR",
        "when": "late",
    },
    {
        "key": "morita_3854",
        "needles": ("Morita", "Hidemasa"),
        "match_id": "3854",
        "dir": ROOT / "3854_JPN_ESP",
        "label": "3854 JPN-ESP",
        "when": "3854",
        "clock_marks": [{"clock_min": 94.0, "period": 2, "label": "cramp 90+4"}],
    },
    {
        "key": "acuna_3835",
        "needles": ("Acuña", "Acuna", "Acu"),
        "match_id": "3835",
        "dir": ROOT / "3835_ARG_MEX",
        "label": "3835 ARG-MEX",
        "when": "3835",
        "clock_marks": [{"clock_min": 92.0, "period": 2, "label": "cramp 90+2"}],
    },
    {
        "key": "musah_3815",
        "needles": ("Musah", "Yunus"),
        "match_id": "3815",
        "dir": ROOT / "3815_USA_WAL",
        "label": "3815 USA-WAL",
        "when": "3815",
        "clock_marks": [{"clock_min": 75.0, "period": 2, "label": "cramp 75'"}],
    },
]

SMOKE_KEY = "gvardiol_10516"

NOTES_TEXT = """\
RSP raw roll-10 Φ — method notes
================================

Purpose
-------
Recompute the position-dependent speed residual (RSP) for selected players using
RAW tracking (`homePlayers` / `awayPlayers`), so player speeds match the same
track source as the tournament position profiles.

Why this folder exists
----------------------
An earlier timeline-suite / new_players path pulled positions from
`possession_cache/..._kin.parquet`, which comes from `extract_focal_tracking`
with `prefer_smoothed_player=True` (`homePlayersSmoothed`). Those smoothed
tracks yield lower stride-12 speeds. The global profiles in
`kinematic_profiles/wc_position_profiles.csv` were built with
`wc_position_profiles._stream_match`, which reads RAW `homePlayers` /
`awayPlayers` only. Mixing smoothed players against a raw profile pulled φ
below 0.5 even when the old raw-vs-raw RSP (e.g. WhatsApp Gvardiol late) was
above 0.5 in P1.

This batch is raw player vs raw profile — same pairing as the original
`run_batch_rsp.py` / `RSP/` figures.

Players / matches
-----------------
- Core five × 2: Amrabat, Hakimi, Modrić, Perišić, Gvardiol on 3820 (early
  MAR–CRO) and 10516 (late CRO–MAR).
- Modrić mid 10506 JPN–CRO (sub at 99:00).
- Acuña 3835 ARG–MEX, Musah 3815 USA–WAL, Morita 3854 JPN–ESP.
Smoke test: Gvardiol late 10516 only (`--smoke`).

Track source (player)
---------------------
- Stream match JSONL via `wc_position_profiles._stream_match`.
- Positions: RAW `homePlayers` / `awayPlayers` (never `*PlayersSmoothed`).
- Kinematics: `match_player_analysis.add_stride_kinematics`
  (central difference, STRIDE = 12 frames) then `add_step_distance`.
- Minute table: `wc_position_profiles._player_minute_values`
  (mean speed / |a| / distance / |jerk| per regulation minute 1–90, with the
  usual coverage gate).

Reference (profile)
-------------------
- File: `kinematic_profiles/wc_position_profiles.csv`
- Level: detailed; group = roster `positionGroupType` for that match
  (e.g. LCB, CM, RB).
- That CSV was accumulated with the same raw `_stream_match` path.

Score formula (speed; same as original RSP / timeline §4)
---------------------------------------------------------
For regulation minutes in each half separately:

1. LOWESS smooth the player's minute speed (`frac=0.5`, `it=0`) → player_fit
2. LOWESS smooth the profile mean series the same way → ref_fit
3. Instantaneous residual (m/s):
     resid = player_fit − ref_fit
   (We do NOT divide the LOWESS means by sd.)
4. Standardize with the profile per-minute sd:
     z = resid / sd
5. Trailing roll-10 mean of z inside the half (window restarts at half-time):
     roll10_z
6. Normalize for plotting:
     phi_roll10 = Φ(roll10_z)   # standard normal CDF
   Horizontal 0.5 = on trend; dotted lines ≈ Φ(±1).

Also stored: half-long Φ(mean-so-far of z) as phi_5a (companion diagnostic).

Outputs per job
---------------
- `{slug}_{match}_rsp_minutes.csv` — raw minute metrics
- `{slug}_{match}_speed_residual_roll10.csv` — minute, residual, z, roll10_z,
  phi_roll10, phi_5a, position, match_id
- `{slug}_{match}_speed_roll10.png` — full-match Φ(rolling z), w=10
  (timeline-suite style)
- `{slug}_{match}_speed_residual_roll10.png` — P1|P2 panels: roll-10 residual
  (top) and Φ (bottom)
- `NOTES.txt` — this file
- `smoke_summary.txt` — written by `--smoke`

Reproduce
---------
  python cognitive_metrics/main_figures_nb_outputs/rsp_raw_roll10/run_rsp_raw_roll10.py --smoke
  python cognitive_metrics/main_figures_nb_outputs/rsp_raw_roll10/run_rsp_raw_roll10.py
  python .../run_rsp_raw_roll10.py --job gvardiol_10516
"""


def slug(name: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return rsp.slug(text).lower()


def write_notes() -> None:
    path = OUT_ROOT / "NOTES.txt"
    path.write_text(NOTES_TEXT, encoding="utf-8")
    print(f"wrote {path.name}")


def resolve_job(job: dict) -> dict:
    """Attach player_id, nickname, position from the match roster."""
    mid = str(job["match_id"])
    roster_path = Path(job["dir"]) / f"{mid}_roster.json"
    raw = json.loads(roster_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("players") or raw.get("roster") or []
    needles = tuple(str(n).lower() for n in job["needles"])
    hit = None
    for rec in raw:
        nick = str((rec.get("player") or {}).get("nickname") or "")
        nick_l = nick.lower()
        if any(n in nick_l for n in needles):
            hit = rec
            break
    if hit is None:
        raise SystemExit(f"No roster hit for {job['needles']} in {mid}")
    pid = str((hit.get("player") or {}).get("id"))
    pos = str(hit.get("positionGroupType") or "").upper()
    if pos == "GK":
        raise SystemExit(f"GK excluded: {nick} {mid}")
    out = dict(job)
    out["player_id"] = pid
    out["nickname"] = nick
    out["position"] = pos
    out["slug"] = slug(nick)
    return out


def discover_local() -> dict[str, wcp.MatchFiles]:
    return {m.match_id: m for m in wcp.discover_matches(ROOT)}


def extract_raw_minutes(match: wcp.MatchFiles, player_id: str) -> pd.DataFrame:
    """Raw homePlayers/awayPlayers → stride-12 → regulation-minute table."""
    return rsp.extract_match_minutes(match, {str(player_id)})


def score_speed(minutes_tab: pd.DataFrame, position: str, profiles: pd.DataFrame):
    sl = minutes_tab.loc[minutes_tab.metric == "speed"].sort_values("minute")
    ref = profiles.loc[
        (profiles.level == "detailed")
        & (profiles.group == position)
        & (profiles.metric == "speed")
    ].sort_values("minute")
    if sl.empty or ref.empty:
        return None
    mins, phi_5a, phi_r10, _acc = rsp.scores_for_slice(sl, ref)
    # Rebuild resid / z / roll10_z for the CSV (scores_for_slice returns Phi + sum).
    merged = sl[["minute", "value"]].merge(ref[["minute", "mean", "sd"]], on="minute", how="inner")
    player_fit = rsp.lowess_half(merged.minute.to_numpy(), merged.value.to_numpy())
    ref_fit = rsp.lowess_half(merged.minute.to_numpy(), merged["mean"].to_numpy())
    sd = merged["sd"].to_numpy(dtype=float)
    resid = player_fit - ref_fit
    z = np.divide(resid, sd, out=np.full_like(resid, np.nan), where=np.isfinite(sd) & (sd > 0))
    roll10_z = rsp.rolling_mean_half(merged.minute.to_numpy(), z, ROLL_W)
    out = pd.DataFrame(
        {
            "minute": mins,
            "residual": resid,
            "z": z,
            "roll10_z": roll10_z,
            "phi_roll10": phi_r10,
            "phi_5a": phi_5a,
            "position": position,
            "match_id": str(minutes_tab["match_id"].iloc[0]),
            "player_id": str(minutes_tab["player_id"].iloc[0]),
            "nickname": str(minutes_tab["nickname"].iloc[0]),
            "track_source": "raw_homePlayers_awayPlayers",
        }
    )
    return out


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"wrote {path.relative_to(OUT_ROOT)}")
    except OSError as exc:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        fig.savefig(alt, dpi=140, bbox_inches="tight")
        print(f"wrote {alt.name} (could not overwrite {path.name}: {exc})")
    plt.close(fig)


def plot_full_match_phi(job: dict, scored: pd.DataFrame, dest: Path) -> None:
    mins = scored["minute"].to_numpy(dtype=float)
    phi = scored["phi_roll10"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 3.7), layout="constrained")
    rsp.style_phi(ax, shade_w=ROLL_W)
    color = rsp.WHEN_COLOR.get(job.get("when"), "#1a5276")
    rsp.plot_halves(ax, mins, phi, color=color, lw=1.9, label=job["label"])
    x_hi = 90.0
    if np.isfinite(mins).any():
        x_hi = max(x_hi, float(np.nanmax(mins)) + 1.0)
    for info in job.get("clock_marks") or []:
        x_hi = max(x_hi, float(info["clock_min"]) + 2.0)
        ax.axvline(float(info["clock_min"]), color="#922b21", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlim(1, x_hi)
    ax.set_title("speed")
    ax.set_xlabel("regulation minute")
    ax.set_ylabel(rf"$\Phi$(rolling $z$), $w={ROLL_W}$")
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"RSP  |  {job['nickname']}  {job['label']}  ({job['position']})  |  "
        f"rolling $w={ROLL_W}$  |  speed only  |  RAW tracks  |  "
        r"0.5 = on trend  |  dotted = $\Phi(\pm 1)$  |  shaded = warm-up",
        fontsize=10,
    )
    savefig(fig, dest)


def plot_period_panels(job: dict, scored: pd.DataFrame, dest: Path) -> None:
    mins = scored["minute"].to_numpy(dtype=float)
    r10 = scored["roll10_z"].to_numpy(dtype=float)
    phi = scored["phi_roll10"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4), sharey="row", layout="constrained")
    for col, period in enumerate((1, 2)):
        if period == 1:
            mask = (mins >= 1) & (mins < 46)
            xlim, vline, warm = (1, 48), 45.0, (0.5, ROLL_W - 0.5)
        else:
            mask = mins >= 46
            xlim, vline, warm = (45, 96), 90.0, (45.5, 45.0 + ROLL_W - 0.5)
        ax = axes[0, col]
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.axvline(vline, color="#888888", lw=0.8, ls="--")
        ax.axvspan(*warm, color="#9aa0a6", alpha=0.16, zorder=0, lw=0)
        if np.any(mask & np.isfinite(r10)):
            ax.plot(mins[mask], r10[mask], color="#1a5276", lw=1.9)
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"P{period}")
        if col == 0:
            ax.set_ylabel(rf"rolling $z$  ($w={ROLL_W}$)")
        ax = axes[1, col]
        ax.axhline(0.5, color="#333333", lw=0.8)
        ax.axhline(norm.cdf(-1.0), color="#888888", lw=0.7, ls=":")
        ax.axhline(norm.cdf(1.0), color="#888888", lw=0.7, ls=":")
        ax.axvline(vline, color="#888888", lw=0.8, ls="--")
        ax.axvspan(*warm, color="#9aa0a6", alpha=0.16, zorder=0, lw=0)
        if np.any(mask & np.isfinite(phi)):
            ax.plot(mins[mask], phi[mask], color="#1a5276", lw=1.9)
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("regulation minute")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel(rf"$\Phi$(rolling $z$), $w={ROLL_W}$")
    fig.suptitle(
        f"position-dependent speed residual  |  {job['nickname']} ({job['position']})  |  "
        f"{job['label']}  |  RAW tracks",
        fontsize=10,
    )
    savefig(fig, dest)


def run_job(job: dict, local: dict, profiles: pd.DataFrame) -> dict:
    job = resolve_job(job)
    mid = str(job["match_id"])
    if mid not in local:
        raise FileNotFoundError(f"local tracking not found for {mid}")
    print(f"== {job['key']}  {job['nickname']}  {mid}  {job['position']} ==", flush=True)
    dest = OUT_ROOT / job["key"]
    dest.mkdir(parents=True, exist_ok=True)

    minutes = extract_raw_minutes(local[mid], job["player_id"])
    minutes = minutes.loc[minutes.player_id.astype(str) == str(job["player_id"])].copy()
    minutes_path = dest / f"{job['slug']}_{mid}_rsp_minutes.csv"
    minutes.to_csv(minutes_path, index=False)
    print(f"wrote {minutes_path.relative_to(OUT_ROOT)}")

    scored = score_speed(minutes, job["position"], profiles)
    if scored is None:
        raise RuntimeError(f"no speed score for {job['key']} position={job['position']}")
    csv_path = dest / f"{job['slug']}_{mid}_speed_residual_roll{ROLL_W}.csv"
    scored.to_csv(csv_path, index=False)
    print(f"wrote {csv_path.relative_to(OUT_ROOT)}")

    plot_full_match_phi(job, scored, dest / f"{job['slug']}_{mid}_speed_roll{ROLL_W}.png")
    plot_period_panels(job, scored, dest / f"{job['slug']}_{mid}_speed_residual_roll{ROLL_W}.png")

    p1 = scored.loc[scored["minute"] <= 45, "phi_roll10"]
    p2 = scored.loc[scored["minute"] > 45, "phi_roll10"]
    summary = {
        "key": job["key"],
        "nickname": job["nickname"],
        "match_id": mid,
        "position": job["position"],
        "phi_p1_mean": float(p1.mean()) if len(p1) else float("nan"),
        "phi_p1_end": float(scored.loc[scored["minute"] == 45, "phi_roll10"].iloc[0])
        if (scored["minute"] == 45).any()
        else float("nan"),
        "phi_start": float(scored["phi_roll10"].iloc[0]),
        "phi_p2_mean": float(p2.mean()) if len(p2) else float("nan"),
        "n_above_0_5_p1": int((p1 > 0.5).sum()),
        "dest": str(dest),
    }
    print(
        f"  phi start={summary['phi_start']:.3f}  "
        f"P1 mean={summary['phi_p1_mean']:.3f}  P1 end={summary['phi_p1_end']:.3f}  "
        f"P1 n>0.5={summary['n_above_0_5_p1']}/45",
        flush=True,
    )
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help=f"Run only {SMOKE_KEY}")
    p.add_argument("--job", type=str, default=None, help="Single job key, e.g. gvardiol_10516")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_notes()
    profiles = pd.read_csv(CSV_PROFILES)
    local = discover_local()

    if args.smoke:
        keys = [SMOKE_KEY]
    elif args.job:
        keys = [args.job]
    else:
        keys = [j["key"] for j in JOBS]

    by_key = {j["key"]: j for j in JOBS}
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"Unknown job key(s): {missing}. Known: {sorted(by_key)}")

    summaries = []
    for key in keys:
        summaries.append(run_job(by_key[key], local, profiles))

    if args.smoke:
        s = summaries[0]
        lines = [
            f"smoke job: {s['key']}",
            f"player: {s['nickname']}  match: {s['match_id']}  position: {s['position']}",
            f"phi start: {s['phi_start']:.6f}",
            f"phi P1 mean: {s['phi_p1_mean']:.6f}",
            f"phi P1 end (min 45): {s['phi_p1_end']:.6f}",
            f"P1 minutes with phi>0.5: {s['n_above_0_5_p1']} / 45",
            "",
            "Expect (raw-vs-raw, matches old RSP/WhatsApp): P1 mostly above 0.5,",
            "P1 end ~ 0.55, start ~ 0.60.",
            "",
        ]
        ok = s["phi_p1_mean"] > 0.5 and s["n_above_0_5_p1"] >= 40
        lines.append(f"smoke_ok: {ok}")
        summary_text = "\n".join(lines) + "\n"
        (OUT_ROOT / "smoke_summary.txt").write_text(summary_text, encoding="utf-8")
        try:
            print(summary_text)
        except UnicodeEncodeError:
            print(summary_text.encode("ascii", "replace").decode("ascii"))
        if not ok:
            raise SystemExit("Smoke check failed: expected Gvardiol late P1 phi mostly > 0.5")
    else:
        pd.DataFrame(summaries).to_csv(OUT_ROOT / "batch_summary.csv", index=False)
        print(f"wrote batch_summary.csv  ({len(summaries)} jobs)")


if __name__ == "__main__":
    main()
