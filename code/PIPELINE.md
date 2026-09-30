# Pipeline: events / tracking → the 7 active 0–1 series

Intermediate CSVs for the 14 cases are under `data/players/*/source/`.
Core kernels are vendored under `code/lib/`. Entry-point batch scripts live under `code/series/`.

Roots: `code/paths.py` → `package_root()` (this package) and `football_root()` (match data workspace; set `FOOTBALL_ROOT` to override).

## Shared XY / speed preprocessing

File: `code/lib/match_player_analysis.py` (+ `possession_states.py` for tracking extract)

| Step | Function | Detail |
|---|---|---|
| Load tracks | `extract_focal_tracking` / `load_outfield_panel` | Match JSONL → x,y (raw or smoothed depending on caller) |
| Speed / accel | `add_stride_kinematics` | Central differences, `STRIDE=12` |
| Path length | `add_step_distance` | Frame-to-frame `step_dist` |

RSP forces **raw** `homePlayers` / `awayPlayers` via `wc_position_profiles._stream_match` (see `series/rsp_raw_roll10_NOTES.txt`).

## 1. `tight_w3` — Synergy (saved PC + window post-process)

1. Pitch-control synergy (individual + team) is produced **outside** this package. **Do not recompute for verification.**
2. Cap ±80 / ±1450 → `m`, then centered 1/3/5/10 min windows → `01f_…timewin.csv` (`m_w3`).
3. Post-process only: `code/series/rebuild.py` → `synergy_tight_timewin_from_events`.

Example: `data/players/marcos_acuna_3835/source/01f_pc_synergy_m_80_1450_timewin.csv`

## 2. `pass_C` — Pass / shot quality

`code/series/run_pass_success_w5.py` (+ `code/xg/`).  
5-min hop-1 window; `C = (1−w)P + wQ`.

Example: `…/source/05_pass_shot_quality_w5.csv`

## 3. `standing` — Positional appropriateness

Source enriched timeseries (not rebuilt here):  
`cognitive_metrics/where_he_stands/fatigue_case_study_timeseries_all_frames_enriched.csv` under `football_root()`.

Norm: pooled median 0.0256 → 0.5, p95 0.112 → 1 (`rebuild.normalize_worst_vulnerability`).  
Mixer: `standing = 1 − worst_vulnerability_5min_norm`.

Example: `…/source/06_standing_position.csv`

## 4. `ols_distance` — Sprint Recovery Capacity

1. `detect_relative_sprint_events` → sprint `distance_m`, `recovery_before_s`
2. `add_rolling_line_fit` with frozen scales (`series/ols_global_scales.json`: N=6, min_n=4, Δx≥20s, s_x=200, s_y=42)
3. Batch: `series/run_ols_bstar_angle_settings.py`, `series/run_ols_bstar_n6_min4_5players.py`
4. Curve: `rebuild.ols_curve_from_fitted` → `02_ols_line_angle.csv`

Example: `…/source/02_ols_line_angle.csv`

## 5–6. `tm_cum` / `op_cum` — LWRD (cumulative)

1. `self_nearest_minute_table` — dynamic nearest-2 teammates/opponents, 1-min residuals `d_resid_*`
2. `rebuild.lwrd_lead_from_residuals` — **cumulative** `C = Σ r` per period (this is what the mixer uses)
3. Mixer clips `C` at ±350 m (suite `N` column uses ±600 m)

Example: `…/source/03_self_nearest_distance_lead.csv`

## 7. `speed_phi` — RSP

1. Profiles: `data/profiles/wc_position_profiles.csv` (or `football_root()/kinematic_profiles/…` if present)
2. `code/lib/run_batch_rsp.py` + `series/run_rsp_raw_roll10.py`  
   LOWESS frac=0.5 → z → roll-10 → `phi_roll10 = Φ(z)`
3. `rebuild.rsp_curve_from_residual` → `04_speed_roll10_phi.csv`

Example: `…/source/04_speed_roll10_phi.csv`

## Mixer

`mixer_fixed.py` interpolates active series onto a 1-minute grid per period, then:

- `cognitive` = weighted mean of tight_w3, pass_C, standing  
- `kinematic` = weighted mean of ols_distance, tm_cum, op_cum, speed_phi  
- `both` = mean of the two groups  
