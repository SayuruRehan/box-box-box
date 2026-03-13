# Solution: Box Box Box F1 Race Simulator

## Overview

This solution works by learning a **parametric lap-time model** from the 30,000 historical F1 races, then using that model to simulate each race deterministically and predict finishing positions.

## Files

| File | Purpose |
|---|---|
| `race_simulator.py` | **Main entry point.** Reads a race JSON from stdin, simulates it, and writes the finishing order to stdout. |
| `model.py` | Shared race simulation core: lap-time formula, per-driver time calculation, race simulation with correct tie-breaking. |
| `model_params.json` | Learned model parameters (produced by `fit_model.py`). |
| `fit_model.py` | **Offline fitter.** Load historical races, precompute analytical features, run coordinate descent to find the best parameters. |
| `validate_local.py` | Helper that runs all 100 local test cases through `race_simulator.py` and reports accuracy. |

## Running the solution

```bash
# Single test case
python solution/race_simulator.py < data/test_cases/inputs/test_001.json

# All 100 test cases (uses test_runner.sh)
./test_runner.sh

# Local accuracy report
python solution/validate_local.py
```

## Lap-time model

For each lap at **tire age** `a` (1-indexed, fresh tire starts at 1), the lap-time delta vs. base is:

```
delta = offset[C]  +  (a == 1 ? fresh_bonus[C] : 0)
        + temp_factor[C] * ( linear_deg[C] * (a-1)
                           + quad_deg[C] * (a-1)^2
                           + cubic_deg[C] * (a-1)^3 )
```

where `temp_factor[C] = max(0.2, 1 + temp_sensitivity[C] * (track_temp - 30) / 10)`.

Total driver time:

```
Σ_laps ( base_lap_time + delta )  +  num_pit_stops × pit_lane_time × pit_penalty_scale
```

Tie-breaking when two drivers have equal total time: **grid position** (lower start position wins).

## Re-fitting parameters

```bash
# Fit on 800 historical races for 3 minutes
python solution/fit_model.py --history-limit 800 --time-budget 180

# Fit on more data for higher accuracy
python solution/fit_model.py --history-limit 3000 --time-budget 300
```

Fitter design:
1. **Analytical stint aggregation** — computes `Σ(age-1)`, `Σ(age-1)^2`, and `Σ(age-1)^3` analytically per stint (O(1)), avoiding per-lap Python loops.
2. **Three-phase search** — wide coordinate descent → random restarts → tight refinement.
3. **No external dependencies** — pure Python, no numpy/scipy required.
