# Beyond Physical Fatigue: Integrating Physical and Cognitive-Tactical Performance for Early Fatigue Detection in Football

Code and data associated with the abstract submission to the MIT Sloan Sports Analytics Conference.

This folder rebuilds the **final 0–1 cognitive and kinematic (fatigue) curves**, and includes the tracking preprocessing kernels used to form the intermediates.

## Active series (7)

| Metric | Series | Group | Weight | Source file |
|---|---|---|---:|---|
| Synergy Score | `tight_w3` | cognitive | 1.0 | `01f_pc_synergy_m_80_1450_timewin.csv` (`m_w3`) |
| Pass success / shot quality | `pass_C` | cognitive | 1.0 | `05_pass_shot_quality_w5.csv` (`C`) |
| Positional Appropriateness | `standing` | cognitive | 1.5 | `06_standing_position.csv` |
| Sprint Recovery Capacity (SRC) | `ols_distance` | kinematic | 1.0 | `02_ols_line_angle.csv` (`distance_m`, `roll_A`) |
| Local Work-Rate Differential (LWRD), teammates | `tm_cum` | kinematic | 1.0 | `03_self_nearest_distance_lead.csv` (teammates **cumulative** `C`) |
| Local Work-Rate Differential (LWRD), opponents | `op_cum` | kinematic | 1.0 | same file (opponents **cumulative** `C`) |
| Relative Speed Profile (RSP) | `speed_phi` | kinematic | 1.0 | `04_speed_roll10_phi.csv` (`phi_roll10`) |

Within each group (cognitive and kinematic separately), each series weight is divided by the sum of the weights of all contributing series in that group.

Mixer remap for LWRD: clip cumulative residual `C` at ±350 m, then `N = (C + 350) / 700`.
OLS: recover θ from `roll_A`, clip ±42.5°, `N = (θ + 42.5) / 85`.
Standing benefit: `1 − worst_vulnerability_5min_norm`.

## Layout

```
beyond_physical_fatigue/
  weights.json
  mixer_fixed.py                 remap + weighted mix
  data/
    cases.json
    profiles/wc_position_profiles.csv
    players/<slug_match>/source/   the 7 CSV inputs
  code/
    paths.py                     package_root() + football_root()
    PIPELINE.md
    lib/                         XY → speed / distance / nearest / OLS kernels
      match_player_analysis.py
      possession_states.py
      run_batch_rsp.py
      wc_position_profiles.py
    series/
      rebuild.py                 03/02/04/06/01f post-process from upstream tables
      run_pass_success_w5.py
      run_rsp_raw_roll10.py
      run_ols_bstar_angle_settings.py
      run_ols_bstar_n6_min4_5players.py
      ols_global_scales.json
      rsp_raw_roll10_NOTES.txt
    xg/                          pass+xG model used by pass_success_w5
```

## Paths

All rebuild scripts resolve roots through `code/paths.py` (no machine-local absolute paths):

- **`package_root()`** — this folder (`beyond_physical_fatigue/`)
- **`football_root()`** — workspace that contains match folders / caches  
  - override with env `FOOTBALL_ROOT`  
  - otherwise the nearest ancestor that has `3820_MAR_CRO`, `10516_CRO_MAR`, `possession_cache`, or `kinematic_profiles`  
  - fallback: the parent of this package

Match JSONL / caches are **not** bundled. Tracking-level rebuilds need that workspace beside (or pointed at by) this package.

## Synergy note

Pitch-control synergy is **not** recomputed here. Saved `01f_…timewin.csv` is the input. Standing `V_actual` / vulnerability values come from the enriched case-study CSV; this package remaps / filters them.

## Run mixer

```bash
python -m beyond_physical_fatigue.mixer_fixed --folder marcos_acuna_3835
python -m beyond_physical_fatigue.mixer_fixed --all
```
