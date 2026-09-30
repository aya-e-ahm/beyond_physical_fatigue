"""
Relative-performance score (RSP) for the 5 focal players, early and late.

Scores from minute-wise z (player LOWESS − detailed-position LOWESS) / global SD:
  5a     Phi of the half-long mean — how has he done since this kickoff?
  roll10 Phi of the last 10 minutes — how has he done recently?
  sum z  running sum of z, no 1/t and no Phi — pile-up since this kickoff.

LOWESS frac=0.5, each half fit alone. Sum, t, and the rolling window restart
at half-time. No sqrt(t). Do not map the raw sum through Phi.

Writes to RSP/. Leaves kinematic_profiles/profile_read/ figures alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess_resid

import match_player_analysis as mpa
import wc_position_profiles as wcp

PLAYERS = ["Amrabat", "Hakimi", "Modric", "Perisic", "Gvardiol"]
OUT_ROOT = mpa.ROOT / "RSP"
ROLL_DIR = OUT_ROOT / "roll10"
SPEED_DIR = ROLL_DIR / "speed"
CSV_PROFILES = mpa.ROOT / "kinematic_profiles" / "wc_position_profiles.csv"
CACHE_CSV = OUT_ROOT / "minutes_all.csv"

ROLL_W = 10
RESID_FRAC = 0.50
RESID_IT = 0
WHEN_COLOR = {"early": "#1a5276", "late": "#922b21"}
WHEN_LABEL = {"early": "early MAR–CRO", "late": "late CRO–MAR"}


def slug(name: str) -> str:
    return (
        name.replace("ć", "c")
        .replace("č", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace(" ", "_")
    )


def resolve_players():
    people = []
    for needle in PLAYERS:
        hits = mpa.find_player(needle)
        hits = hits.loc[hits["when"].isin(("early", "late"))].drop_duplicates(["when", "player_id"])
        if hits.empty:
            raise SystemExit(f"no roster hit for {needle}")
        pid = str(hits.iloc[0]["player_id"])
        hits = hits.loc[hits.player_id.astype(str) == pid]
        by_when = {}
        for _, r in hits.iterrows():
            by_when[r["when"]] = {
                "when": r["when"],
                "match_id": str(r["match_id"]),
                "player_id": pid,
                "nickname": str(r["nickname"]),
                "position": str(r["position"]).upper(),
            }
        missing = [w for w in ("early", "late") if w not in by_when]
        if missing:
            raise SystemExit(f"{needle} missing matches: {missing}")
        people.append(by_when)
    return people


def _roster_records(match: wcp.MatchFiles) -> list[dict]:
    meta = wcp._load_json(match.metadata)
    home_id = str((meta.get("homeTeam") or {}).get("id"))
    raw = wcp._load_json(match.roster)
    if isinstance(raw, dict):
        raw = raw.get("players") or raw.get("roster") or []
    out = []
    for rec in raw:
        pid = str((rec.get("player") or {}).get("id"))
        team_id = str((rec.get("team") or {}).get("id"))
        side = "home" if team_id == home_id else "away"
        jersey = str(rec.get("shirtNumber"))
        code = str(rec.get("positionGroupType") or "").upper()
        nick = str((rec.get("player") or {}).get("nickname") or pid)
        out.append(
            {
                "player_id": pid,
                "side": side,
                "jersey": jersey,
                "position": code,
                "nickname": nick,
                "key": (side, jersey),
            }
        )
    return out


def extract_match_minutes(match: wcp.MatchFiles, wanted_ids: set[str]) -> pd.DataFrame:
    roster = _roster_records(match)
    hits = [r for r in roster if r["player_id"] in wanted_ids]
    missing = wanted_ids - {r["player_id"] for r in hits}
    if missing:
        raise SystemExit(f"{match.match_id} missing player_ids {sorted(missing)}")
    index = {r["key"]: r["position"] for r in hits}
    print(f"  streaming {match.match_id}  ({len(hits)} players)", flush=True)
    bufs = wcp._stream_match(match, index)
    rows = []
    for rec in hits:
        buf = bufs[rec["key"]]
        if len(buf) < 10:
            print(f"    skip {rec['nickname']}: too few frames")
            continue
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
        vals = wcp._player_minute_values(df)
        for mi in range(wcp.N_MINUTES):
            for k, metric in enumerate(wcp.METRICS):
                rows.append(
                    {
                        "match_id": match.match_id,
                        "player_id": rec["player_id"],
                        "nickname": rec["nickname"],
                        "position": rec["position"],
                        "minute": mi + 1,
                        "metric": metric,
                        "value": float(vals[mi, k]) if np.isfinite(vals[mi, k]) else np.nan,
                    }
                )
        print(f"    {rec['nickname']}  {rec['position']}  ok", flush=True)
    del bufs
    return pd.DataFrame(rows)


def load_or_extract_minutes(people) -> pd.DataFrame:
    wanted = {p["early"]["player_id"] for p in people}
    if CACHE_CSV.exists():
        tab = pd.read_csv(CACHE_CSV)
        have = set(tab.player_id.astype(str).unique())
        if wanted <= have and set(tab.match_id.astype(str).unique()) >= {"3820", "10516"}:
            print(f"using cached minutes {CACHE_CSV}")
            return tab
    local = {m.match_id: m for m in wcp.discover_matches(".")}
    frames = []
    for mid in ("3820", "10516"):
        if mid not in local:
            raise FileNotFoundError(f"local tracking not found for {mid}")
        frames.append(extract_match_minutes(local[mid], wanted))
    tab = pd.concat(frames, ignore_index=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(CACHE_CSV, index=False)
    print(f"wrote {CACHE_CSV}  ({len(tab)} rows)")
    return tab


def lowess_half(minutes, values, frac=RESID_FRAC, it=RESID_IT):
    minutes = np.asarray(minutes, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.full(minutes.shape, np.nan)
    for lo, hi in ((1, 45), (46, 90)):
        mask = (minutes >= lo) & (minutes <= hi) & np.isfinite(values)
        n = int(mask.sum())
        if n < 4:
            continue
        frac_use = float(np.clip(frac, 3.0 / n, 1.0))
        sm = _lowess_resid(values[mask], minutes[mask], frac=frac_use, it=int(it), return_sorted=True)
        out[mask] = np.interp(minutes[mask], sm[:, 0], sm[:, 1])
    return out


def cumsum_half(minutes, residual):
    minutes = np.asarray(minutes, dtype=float)
    residual = np.asarray(residual, dtype=float)
    acc = np.full(minutes.shape, np.nan)
    t_half = np.full(minutes.shape, np.nan)
    for lo, hi in ((1, 45), (46, 90)):
        m = np.where((minutes >= lo) & (minutes <= hi) & np.isfinite(residual))[0]
        if len(m) == 0:
            continue
        order = m[np.argsort(minutes[m])]
        acc[order] = np.cumsum(residual[order])
        t_half[order] = np.arange(1, len(order) + 1, dtype=float)
    return acc, t_half


def rolling_mean_half(minutes, z, window):
    minutes = np.asarray(minutes, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.full(z.shape, np.nan)
    w = int(window)
    for lo, hi in ((1, 45), (46, 90)):
        idx = np.where((minutes >= lo) & (minutes <= hi) & np.isfinite(z))[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(minutes[idx])]
        vals = z[order]
        csum = np.cumsum(vals)
        n = len(vals)
        out = np.empty(n)
        for i in range(n):
            start = i - w + 1
            if start < 0:
                out[i] = csum[i] / float(i + 1)
            else:
                prev = csum[start - 1] if start > 0 else 0.0
                out[i] = (csum[i] - prev) / float(w)
        r[order] = out
    return r


def plot_halves(ax, minutes, y, **style):
    minutes = np.asarray(minutes, dtype=float)
    y = np.asarray(y, dtype=float)
    label = style.pop("label", None)
    first = True
    for lo, hi in ((1, 45), (46, 90)):
        m = (minutes >= lo) & (minutes <= hi) & np.isfinite(y)
        if not np.any(m):
            continue
        ax.plot(minutes[m], y[m], label=label if first else None, **style)
        first = False


def style_phi(ax, shade_w=None):
    ax.axhline(0.5, color="#333333", lw=0.8)
    ax.axhline(norm.cdf(-1.0), color="#888888", lw=0.7, ls=":")
    ax.axhline(norm.cdf(1.0), color="#888888", lw=0.7, ls=":")
    ax.axvline(45.5, color="#888888", lw=0.8, ls="--")
    if shade_w is not None:
        w = int(shade_w)
        ax.axvspan(0.5, w - 0.5, color="#9aa0a6", alpha=0.16, zorder=0, lw=0)
        ax.axvspan(45.5, 45.0 + w - 0.5, color="#9aa0a6", alpha=0.16, zorder=0, lw=0)
    ax.set_xlim(1, 90)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)


def scores_for_slice(player_tab, ref_tab):
    merged = player_tab[["minute", "value"]].merge(
        ref_tab[["minute", "mean", "sd"]], on="minute", how="inner",
    )
    minutes = merged.minute.to_numpy()
    player_fit = lowess_half(minutes, merged.value.to_numpy())
    ref_fit = lowess_half(minutes, merged["mean"].to_numpy())
    sd = merged["sd"].to_numpy(dtype=float)
    resid = player_fit - ref_fit
    z = np.divide(
        resid, sd, out=np.full_like(resid, np.nan, dtype=float),
        where=np.isfinite(sd) & (sd > 0),
    )
    acc, t_half = cumsum_half(minutes, z)
    mean_so_far = np.divide(
        acc, t_half, out=np.full_like(acc, np.nan, dtype=float),
        where=np.isfinite(t_half) & (t_half > 0),
    )
    r10 = rolling_mean_half(minutes, z, ROLL_W)
    return minutes, norm.cdf(mean_so_far), norm.cdf(r10), acc


def style_sum(ax):
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.axvline(45.5, color="#888888", lw=0.8, ls="--")
    ax.set_xlim(1, 90)
    ax.grid(True, alpha=0.25)


def series_from_panel(panel, kind):
    mins, phi_5a, phi_r10, acc, code, name = panel
    if kind == "5a":
        return mins, phi_5a
    if kind == "roll10":
        return mins, phi_r10
    return mins, acc


def savefig(fig, path):
    path = Path(path)
    try:
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print("wrote", path.name)
    except OSError as exc:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        fig.savefig(alt, dpi=140, bbox_inches="tight")
        print("wrote", alt.name, "(could not overwrite", path.name, ":", exc, ")")
    plt.close(fig)


def draw_all_players(people, panels, kind, whens, fname, ylabel, title_bit, phi=True):
    fig, axes = plt.subplots(
        len(people), 4, figsize=(14.6, 2.35 * len(people) + 1.1),
        sharex=True, sharey=phi, layout="constrained",
    )
    for row, person in enumerate(people):
        pid = person["early"]["player_id"]
        short = person["early"]["nickname"].split()[-1]
        if whens == ("early",):
            pos_note = person["early"]["position"]
        elif whens == ("late",):
            pos_note = person["late"]["position"]
        elif person["late"]["position"] != person["early"]["position"]:
            pos_note = f"{person['early']['position']}/{person['late']['position']}"
        else:
            pos_note = person["early"]["position"]
        for col, metric in enumerate(wcp.METRICS):
            ax = axes[row, col]
            if phi:
                style_phi(ax, shade_w=ROLL_W if kind == "roll10" else None)
            else:
                style_sum(ax)
            for when in whens:
                key = (pid, when, metric)
                if key not in panels:
                    continue
                mins, y = series_from_panel(panels[key], kind)
                plot_halves(ax, mins, y, color=WHEN_COLOR[when], lw=1.6, label=WHEN_LABEL[when])
            if row == 0:
                ax.set_title(metric)
            if row == len(people) - 1:
                ax.set_xlabel("regulation minute")
        axes[row, 0].set_ylabel(f"{short} ({pos_note})", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    if whens == ("early",):
        match_bit = "early MAR–CRO only"
    elif whens == ("late",):
        match_bit = "late CRO–MAR only"
    else:
        match_bit = "early and late"
    if phi:
        fig.suptitle(
            rf"RSP  |  {match_bit}  |  {title_bit}  |  {ylabel}  |  range (0, 1)"
            rf"  |  0.5 = on trend  |  dotted = $\Phi(\pm 1)\approx 0.16,\ 0.84$",
            fontsize=11,
        )
    else:
        fig.suptitle(
            f"RSP  |  {match_bit}  |  {title_bit}  |  {ylabel}  |  0 = on trend this half"
            "  |  restarts at half-time",
            fontsize=11,
        )
    savefig(fig, OUT_ROOT / fname)


def draw_player_roll10(panels, person, when, out_dir):
    """One player, one match, rolling w=10 only. Four metrics."""
    pid = person["early"]["player_id"]
    name = person["early"]["nickname"]
    pos = person[when]["position"]
    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.4), sharex=True, sharey=True, layout="constrained")
    for col, metric in enumerate(wcp.METRICS):
        ax = axes[col]
        style_phi(ax, shade_w=ROLL_W)
        key = (pid, when, metric)
        if key in panels:
            mins, y = series_from_panel(panels[key], "roll10")
            plot_halves(ax, mins, y, color=WHEN_COLOR[when], lw=1.8, label=WHEN_LABEL[when])
        ax.set_title(metric)
        ax.set_xlabel("regulation minute")
    axes[0].set_ylabel(rf"$\Phi$(rolling $z$), $w={ROLL_W}$")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"RSP  |  {name}  {WHEN_LABEL[when]}  ({pos})  |  rolling $w={ROLL_W}$"
        rf"  |  0.5 = on trend  |  dotted = $\Phi(\pm 1)$  |  shaded = warm-up",
        fontsize=10,
    )
    savefig(fig, Path(out_dir) / f"{slug(name)}_{when}_roll{ROLL_W}.png")


def draw_player_roll10_speed(panels, person, when, out_dir):
    """One player, one match, rolling w=10, speed only."""
    pid = person["early"]["player_id"]
    name = person["early"]["nickname"]
    pos = person[when]["position"]
    fig, ax = plt.subplots(figsize=(8.6, 3.7), layout="constrained")
    style_phi(ax, shade_w=ROLL_W)
    key = (pid, when, "speed")
    if key in panels:
        mins, y = series_from_panel(panels[key], "roll10")
        plot_halves(ax, mins, y, color=WHEN_COLOR[when], lw=1.9, label=WHEN_LABEL[when])
    ax.set_title("speed")
    ax.set_xlabel("regulation minute")
    ax.set_ylabel(rf"$\Phi$(rolling $z$), $w={ROLL_W}$")
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"RSP  |  {name}  {WHEN_LABEL[when]}  ({pos})  |  rolling $w={ROLL_W}$  |  speed only"
        rf"  |  0.5 = on trend  |  dotted = $\Phi(\pm 1)$  |  shaded = warm-up",
        fontsize=10,
    )
    savefig(fig, Path(out_dir) / f"{slug(name)}_{when}_roll{ROLL_W}_speed.png")


def draw_all_players_roll10_speed(people, panels, when, out_dir):
    fig, axes = plt.subplots(
        len(people), 1, figsize=(8.6, 2.15 * len(people) + 0.8),
        sharex=True, sharey=True, layout="constrained",
    )
    for row, person in enumerate(people):
        ax = axes[row]
        pid = person["early"]["player_id"]
        short = person["early"]["nickname"].split()[-1]
        pos = person[when]["position"]
        style_phi(ax, shade_w=ROLL_W)
        key = (pid, when, "speed")
        if key in panels:
            mins, y = series_from_panel(panels[key], "roll10")
            plot_halves(ax, mins, y, color=WHEN_COLOR[when], lw=1.7, label=WHEN_LABEL[when])
        ax.set_ylabel(f"{short} ({pos})", fontsize=8)
        if row == 0:
            ax.set_title("speed")
            ax.legend(fontsize=7, frameon=False)
        if row == len(people) - 1:
            ax.set_xlabel("regulation minute")
    match_bit = "early MAR–CRO only" if when == "early" else "late CRO–MAR only"
    fig.suptitle(
        rf"RSP  |  {match_bit}  |  last {ROLL_W} minutes  |  speed  |  $\Phi$(rolling $z$)"
        rf"  |  0.5 = on trend  |  dotted = $\Phi(\pm 1)$  |  shaded = warm-up",
        fontsize=11,
    )
    savefig(fig, Path(out_dir) / f"all_players_{when}_roll{ROLL_W}_speed.png")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(CSV_PROFILES)
    people = resolve_players()
    minutes_all = load_or_extract_minutes(people)
    minutes_all["player_id"] = minutes_all.player_id.astype(str)
    minutes_all["match_id"] = minutes_all.match_id.astype(str)

    panels = {}
    table_rows = []
    for person in people:
        name = person["early"]["nickname"]
        pid = person["early"]["player_id"]
        for when, info in person.items():
            sl_all = minutes_all.loc[
                (minutes_all.player_id == info["player_id"])
                & (minutes_all.match_id == info["match_id"])
            ]
            code = info["position"]
            for metric in wcp.METRICS:
                sl = sl_all.loc[sl_all.metric == metric].sort_values("minute")
                ref = profiles.loc[
                    (profiles.level == "detailed")
                    & (profiles.group == code)
                    & (profiles.metric == metric)
                ].sort_values("minute")
                if sl.empty or ref.empty:
                    print(f"  missing data {name} {when} {metric} {code}")
                    continue
                mins, phi_5a, phi_r10, acc = scores_for_slice(sl, ref)
                panels[(pid, when, metric)] = (mins, phi_5a, phi_r10, acc, code, name)
                for lo, hi, half in ((1, 45, "H1"), (46, 90, "H2")):
                    m = (mins >= lo) & (mins <= hi) & np.isfinite(phi_5a)
                    m_acc = (mins >= lo) & (mins <= hi) & np.isfinite(acc)
                    table_rows.append(
                        {
                            "player": name,
                            "match": WHEN_LABEL[when],
                            "position": code,
                            "metric": metric,
                            "half": half,
                            "RSP 5a end": float(phi_5a[m][-1]) if np.any(m) else np.nan,
                            "RSP roll10 end": float(phi_r10[m][-1]) if np.any(m) else np.nan,
                            "sum z end": float(acc[m_acc][-1]) if np.any(m_acc) else np.nan,
                        }
                    )
        print(
            f"{name}: early {person['early']['position']} / late {person['late']['position']}",
            flush=True,
        )

        fig, axes = plt.subplots(2, 4, figsize=(14.6, 7.2), sharex=True, sharey=True, layout="constrained")
        for col, metric in enumerate(wcp.METRICS):
            ax_a, ax_b = axes[0, col], axes[1, col]
            style_phi(ax_a)
            style_phi(ax_b, shade_w=ROLL_W)
            for when in ("early", "late"):
                key = (pid, when, metric)
                if key not in panels:
                    continue
                mins, phi_5a, phi_r10, acc, code, _ = panels[key]
                plot_halves(ax_a, mins, phi_5a, color=WHEN_COLOR[when], lw=1.7, label=WHEN_LABEL[when])
                plot_halves(ax_b, mins, phi_r10, color=WHEN_COLOR[when], lw=1.7, label=WHEN_LABEL[when])
            ax_a.set_title(metric)
            ax_b.set_xlabel("regulation minute")
        pos_note = person["early"]["position"]
        if person["late"]["position"] != pos_note:
            pos_note = f"{person['early']['position']} early / {person['late']['position']} late"
        axes[0, 0].set_ylabel(r"5a. $\Phi$(half-long mean)")
        axes[1, 0].set_ylabel(rf"6. $\Phi$(rolling $z$), $w={ROLL_W}$")
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(
            f"RSP  |  {name}  ({pos_note})  |  detailed-position LOWESS residual / global SD"
            rf"  |  0.5 = on trend  |  dotted = $\Phi(\pm 1)$  |  shaded = roll-{ROLL_W} warm-up",
            fontsize=10,
        )
        savefig(fig, OUT_ROOT / f"{slug(name)}_5a_roll{ROLL_W}.png")

        for when in ("early", "late"):
            fig, axes = plt.subplots(3, 4, figsize=(14.6, 9.6), sharex=True, layout="constrained")
            for col, metric in enumerate(wcp.METRICS):
                ax_a, ax_r, ax_s = axes[0, col], axes[1, col], axes[2, col]
                style_phi(ax_a)
                style_phi(ax_r, shade_w=ROLL_W)
                style_sum(ax_s)
                key = (pid, when, metric)
                if key in panels:
                    mins, phi_5a, phi_r10, acc, code, _ = panels[key]
                    plot_halves(ax_a, mins, phi_5a, color=WHEN_COLOR[when], lw=1.8, label=WHEN_LABEL[when])
                    plot_halves(ax_r, mins, phi_r10, color=WHEN_COLOR[when], lw=1.8, label=WHEN_LABEL[when])
                    plot_halves(ax_s, mins, acc, color=WHEN_COLOR[when], lw=1.8, label=WHEN_LABEL[when])
                ax_a.set_title(metric)
                ax_s.set_xlabel("regulation minute")
            axes[0, 0].set_ylabel(r"5a. $\Phi$(half-long mean)")
            axes[1, 0].set_ylabel(rf"6. $\Phi$(rolling $z$), $w={ROLL_W}$")
            axes[2, 0].set_ylabel(r"sum of $z$  (no $1/t$)")
            axes[0, 0].legend(fontsize=7, frameon=False)
            fig.suptitle(
                f"RSP  |  {name}  {WHEN_LABEL[when]}  ({person[when]['position']})"
                f"  |  5a, roll-{ROLL_W}, sum of z"
                rf"  |  top two: $\Phi$ on (0, 1)  |  bottom: raw running sum, restart at HT",
                fontsize=10,
            )
            savefig(fig, OUT_ROOT / f"{slug(name)}_{when}_5a_roll{ROLL_W}_sumz.png")

        roll_dir = ROLL_DIR
        roll_dir.mkdir(parents=True, exist_ok=True)
        for when in ("early", "late"):
            draw_player_roll10(panels, person, when, roll_dir)
        speed_dir = SPEED_DIR
        speed_dir.mkdir(parents=True, exist_ok=True)
        for when in ("early", "late"):
            draw_player_roll10_speed(panels, person, when, speed_dir)

    SPEED_DIR.mkdir(parents=True, exist_ok=True)
    for when in ("early", "late"):
        draw_all_players_roll10_speed(people, panels, when, SPEED_DIR)

    for when in ("early", "late"):
        draw_all_players(
            people, panels, "5a", (when,), f"all_players_{when}_5a.png",
            r"5a. $\Phi$(half-long mean)", "since start of half", phi=True,
        )
        draw_all_players(
            people, panels, "roll10", (when,), f"all_players_{when}_roll{ROLL_W}.png",
            rf"6. $\Phi$(rolling $z$), $w={ROLL_W}$", f"last {ROLL_W} minutes", phi=True,
        )
        draw_all_players(
            people, panels, "sum", (when,), f"all_players_{when}_sum_z.png",
            r"sum of $z$ (no $1/t$)", "pile-up since start of half", phi=False,
        )

    tbl = pd.DataFrame(table_rows)
    tbl_path = OUT_ROOT / "end_of_half.csv"
    tbl.to_csv(tbl_path, index=False)
    print(tbl.round(3).to_string(index=False))
    print("wrote", tbl_path)
    print("done ->", OUT_ROOT)


if __name__ == "__main__":
    main()
