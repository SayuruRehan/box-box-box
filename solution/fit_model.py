#!/usr/bin/env python3
"""Offline parameter fitter for the Box Box Box race simulator.

Design:
  - Analytical closed-form stint summation (O(1) per stint, no per-lap loop)
  - Pure Python; no numpy/scipy required
  - Three-phase search: wide coordinate descent → random restarts → tight refinement
  - Matches the grid-position tie-break used in race_simulator.py / model.py
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from model import DEFAULT_PARAM_PATH, ModelParams, load_params, save_params


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DIR = REPO_ROOT / "data" / "historical_races"
TEST_INPUT_DIR = REPO_ROOT / "data" / "test_cases" / "inputs"
TEST_EXPECTED_DIR = REPO_ROOT / "data" / "test_cases" / "expected_outputs"

REFERENCE_TEMP = 30.0
COMPOUND_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

# ─── Parameter vector layout (16 values) ─────────────────────────────────────
# [0:3]   compound_offset   SOFT, MEDIUM, HARD
# [3:6]   linear_deg        SOFT, MEDIUM, HARD
# [6:9]   quadratic_deg     SOFT, MEDIUM, HARD
# [9:12]  cubic_deg         SOFT, MEDIUM, HARD
# [12:15] fresh_bonus       SOFT, MEDIUM, HARD
# [15:18] temp_sensitivity  SOFT, MEDIUM, HARD
# [18]    pit_penalty_scale
# ─────────────────────────────────────────────────────────────────────────────

BOUNDS = [
    (-2.2, -0.2),   # offset SOFT     (should be negative: SOFT is faster)
    (-0.4,  0.4),   # offset MEDIUM   (reference, allow small deviation)
    ( 0.2,  2.0),   # offset HARD     (should be positive: HARD is slower)
    ( 0.01, 0.18),  # linear_deg SOFT
    ( 0.005,0.10),  # linear_deg MEDIUM
    ( 0.002,0.06),  # linear_deg HARD
    ( 0.0,  0.012), # quad_deg SOFT
    ( 0.0,  0.006), # quad_deg MEDIUM
    ( 0.0,  0.004), # quad_deg HARD
    ( 0.0,  0.0008),# cubic_deg SOFT
    ( 0.0,  0.0003),# cubic_deg MEDIUM
    ( 0.0,  0.0001),# cubic_deg HARD
    (-0.8,  0.1),   # fresh_bonus SOFT
    (-0.4,  0.1),   # fresh_bonus MEDIUM
    (-0.2,  0.1),   # fresh_bonus HARD
    ( 0.0,  0.14),  # temp_sens SOFT
    ( 0.0,  0.09),  # temp_sens MEDIUM
    ( 0.0,  0.06),  # temp_sens HARD
    ( 0.90, 1.10),  # pit_penalty_scale
]

# Physically reasonable starting point
INITIAL = [
    -0.70, 0.00, 0.65,    # compound offsets
    0.055, 0.028, 0.013,  # linear_deg
    0.003, 0.001, 0.0005, # quad_deg
    0.00012, 0.00003, 0.00001, # cubic_deg
    -0.10, -0.04, -0.02,  # fresh_bonus
    0.030, 0.018, 0.010,  # temp_sensitivity
    1.000,                 # pit_penalty_scale
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_historical_races(limit):
    races = []
    for path in sorted(HISTORICAL_DIR.glob("races_*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            races.extend(json.load(fh))
        if limit and len(races) >= limit:
            return races[:limit]
    return races


def load_local_tests():
    races = []
    for path in sorted(TEST_INPUT_DIR.glob("test_*.json")):
        exp = TEST_EXPECTED_DIR / path.name
        if not exp.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            race = json.load(fh)
        with open(exp, encoding="utf-8") as fh:
            race["finishing_positions"] = json.load(fh)["finishing_positions"]
        races.append(race)
    return races


# ─── Analytical feature precomputation ───────────────────────────────────────

def build_stints(strategy, total_laps):
    """Return [(compound, num_laps), ...] for each stint in pit-stop order."""
    stops = sorted(strategy["pit_stops"], key=lambda s: s["lap"])
    stints = []
    compound = strategy["starting_tire"]
    lap_ptr = 1
    for stop in stops:
        n = stop["lap"] - lap_ptr + 1
        if n > 0:
            stints.append((compound, n))
        compound = stop["to_tire"]
        lap_ptr = stop["lap"] + 1
    remaining = total_laps - lap_ptr + 1
    if remaining > 0:
        stints.append((compound, remaining))
    return stints


def precompute_race(race):
    """Return (list_of_driver_tuples, finishing_positions).

    Each driver tuple:
    (base_time, pit_time, comp_laps[3], comp_stints[3],
     comp_lin[3], comp_quad[3], comp_cube[3], temp_delta, driver_id)
    Compound index: SOFT=0, MEDIUM=1, HARD=2
    """
    cfg = race["race_config"]
    total_laps = cfg["total_laps"]
    base_lap_time = cfg["base_lap_time"]
    pit_lane_time = cfg["pit_lane_time"]
    temp_delta = (cfg["track_temp"] - REFERENCE_TEMP) / 10.0

    driver_features = []
    for key in sorted(race["strategies"], key=lambda k: int(k[3:])):
        strat = race["strategies"][key]
        stints = build_stints(strat, total_laps)
        comp_laps   = [0, 0, 0]
        comp_stints = [0, 0, 0]
        comp_lin    = [0.0, 0.0, 0.0]
        comp_quad   = [0.0, 0.0, 0.0]
        comp_cube   = [0.0, 0.0, 0.0]
        for cmp, n in stints:
            ci = COMPOUND_IDX[cmp]
            comp_laps[ci]   += n
            comp_stints[ci] += 1
            comp_lin[ci]    += n * (n - 1) / 2.0               # Σ(age-1) for age 1..n
            comp_quad[ci]   += n * (n - 1) * (2 * n - 1) / 6.0 # Σ(age-1)² for age 1..n
            comp_cube[ci]   += (n * (n - 1) / 2.0) ** 2        # Σ(age-1)^3 for age 1..n
        driver_features.append((
            total_laps * base_lap_time,
            len(strat["pit_stops"]) * pit_lane_time,
            comp_laps, comp_stints, comp_lin, comp_quad, comp_cube,
            temp_delta, strat["driver_id"],
        ))
    return driver_features, race.get("finishing_positions", [])


# ─── Fast scoring ─────────────────────────────────────────────────────────────

def driver_time(feat, vec):
    """Compute a driver's total race time from precomputed features and param vector."""
    base, pit, comp_laps, comp_stints, comp_lin, comp_quad, comp_cube, td, _ = feat
    t = base + pit * vec[18]
    for ci in range(3):
        n  = comp_laps[ci]
        ns = comp_stints[ci]
        ls = comp_lin[ci]
        qs = comp_quad[ci]
        cs = comp_cube[ci]
        ts = vec[15 + ci]
        temp_factor = max(0.2, 1.0 + ts * td)
        t += (n * vec[ci]
              + ns * vec[12 + ci]
              + temp_factor * (vec[3 + ci] * ls + vec[6 + ci] * qs + vec[9 + ci] * cs))
    return t


def predict_order(driver_features, vec):
    """Return finishing order using (total_time, grid_pos) as sort key."""
    times = [
        (driver_time(f, vec), grid_pos, f[8])
        for grid_pos, f in enumerate(driver_features, start=1)
    ]
    times.sort()
    return [did for _, _, did in times]


def score(precomputed, vec):
    """Return (exact_matches, total_pairwise_inversions) for the given param vector."""
    exact = 0
    inversions = 0
    for driver_features, actual_order in precomputed:
        predicted = predict_order(driver_features, vec)
        if predicted == actual_order:
            exact += 1
        else:
            rank = {d: i for i, d in enumerate(actual_order)}
            for i, pi in enumerate(predicted):
                ri = rank[pi]
                for j in range(i + 1, len(predicted)):
                    if ri > rank[predicted[j]]:
                        inversions += 1
    return exact, inversions


def combined_score(exact, inversions):
    """A single scalar; higher is better."""
    return exact * 10000 - inversions


# ─── Parameter helpers ────────────────────────────────────────────────────────

def vec_to_params(vec):
    return ModelParams(
        compound_offset  ={c: vec[i]      for i, c in enumerate(COMPOUNDS)},
        linear_deg       ={c: vec[3 + i]  for i, c in enumerate(COMPOUNDS)},
        quadratic_deg    ={c: vec[6 + i]  for i, c in enumerate(COMPOUNDS)},
        cubic_deg        ={c: vec[9 + i]  for i, c in enumerate(COMPOUNDS)},
        fresh_bonus      ={c: vec[12 + i] for i, c in enumerate(COMPOUNDS)},
        temp_sensitivity ={c: vec[15 + i] for i, c in enumerate(COMPOUNDS)},
        pit_penalty_scale=float(vec[18]),
    )


def params_to_vec(params):
    vec = []
    for field_name in (
        "compound_offset",
        "linear_deg",
        "quadratic_deg",
        "cubic_deg",
        "fresh_bonus",
        "temp_sensitivity",
    ):
        values = getattr(params, field_name)
        vec.extend(values[compound] for compound in COMPOUNDS)
    vec.append(float(params.pit_penalty_scale))
    return vec


# ─── Coordinate descent with line search ─────────────────────────────────────

def line_search(idx, vec, precomputed, lo, hi, steps=12):
    best_val = vec[idx]
    ex0, inv0 = score(precomputed, vec)
    best_s = combined_score(ex0, inv0)
    candidate = list(vec)
    for k in range(steps + 1):
        v = lo + (hi - lo) * k / steps
        candidate[idx] = v
        ex, inv = score(precomputed, candidate)
        s = combined_score(ex, inv)
        if s > best_s:
            best_s = s
            best_val = v
    return best_val


def coord_descent_sweep(vec, precomputed, narrow_factor=1.0):
    """One sweep over all parameters; narrow_factor shrinks each search window."""
    vec = list(vec)
    for idx, (lo, hi) in enumerate(BOUNDS):
        span = (hi - lo) * narrow_factor
        centre = vec[idx]
        search_lo = max(lo, centre - span / 2)
        search_hi = min(hi, centre + span / 2)
        vec[idx] = line_search(idx, vec, precomputed, search_lo, search_hi)
    return vec


# ─── Main fit routine ─────────────────────────────────────────────────────────

def fit(precomputed, time_budget_s=90, seed=42, verbose=True, initial_vec=None):
    """Find best parameter vector using three-phase search."""
    rng = random.Random(seed)
    t0         = time.monotonic()
    phase1_end = t0 + time_budget_s * 0.40  # wide sweeps
    phase2_end = t0 + time_budget_s * 0.85  # restarts
    t_end      = t0 + time_budget_s         # tight refinement

    vec = list(initial_vec or INITIAL)
    ex, inv = score(precomputed, vec)
    best_s   = combined_score(ex, inv)
    best_vec = vec[:]
    if verbose:
        print(f"  start  exact={ex} inversions={inv} score={best_s}", flush=True)

    # Phase 1: wide coordinate descent
    sweep = 0
    while time.monotonic() < phase1_end:
        vec = coord_descent_sweep(vec, precomputed, narrow_factor=1.0)
        ex, inv = score(precomputed, vec)
        s = combined_score(ex, inv)
        sweep += 1
        if s > best_s:
            best_s = s
            best_vec = vec[:]
        if verbose:
            print(f"  wide {sweep:3d}  exact={ex} inv={inv} score={s}", flush=True)

    # Phase 2: random restarts
    vec = best_vec[:]
    restarts = 0
    while time.monotonic() < phase2_end:
        v_new = [
            min(hi, max(lo, best_vec[i] + rng.uniform(-0.20, 0.20) * (hi - lo)))
            for i, (lo, hi) in enumerate(BOUNDS)
        ]
        v_new = coord_descent_sweep(v_new, precomputed, narrow_factor=0.5)
        ex, inv = score(precomputed, v_new)
        s = combined_score(ex, inv)
        restarts += 1
        if s > best_s:
            best_s = s
            best_vec = v_new[:]
            vec = v_new[:]
        if verbose:
            print(f"  restart {restarts:3d}  exact={ex} inv={inv} score={s}", flush=True)

    # Phase 3: tight refinement from best position
    vec = best_vec[:]
    while time.monotonic() < t_end:
        vec = coord_descent_sweep(vec, precomputed, narrow_factor=0.10)
        ex, inv = score(precomputed, vec)
        s = combined_score(ex, inv)
        if s > best_s:
            best_s = s
            best_vec = vec[:]
        if verbose:
            print(f"  tight  exact={ex} inv={inv} score={s}", flush=True)

    return best_vec, best_s


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit Box Box Box race model parameters")
    parser.add_argument("--history-limit", type=int, default=800)
    parser.add_argument("--time-budget",   type=int, default=120,
                        help="Fitting time budget in seconds")
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_PARAM_PATH)
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    print("Loading data...", flush=True)
    historical  = load_historical_races(args.history_limit) if args.history_limit > 0 else []
    local_tests = load_local_tests()
    all_races   = historical + local_tests

    if not all_races:
        print("No training data found.", file=sys.stderr)
        sys.exit(1)

    print(f"Precomputing features for {len(all_races)} races...", flush=True)
    precomputed = [(df, fp) for df, fp in (precompute_race(r) for r in all_races) if fp]

    initial_vec = None
    if args.output.exists():
        try:
            initial_vec = params_to_vec(load_params(args.output))
            print(f"Warm-starting from {args.output}...", flush=True)
        except (OSError, ValueError, KeyError, TypeError):
            initial_vec = None

    print(f"Fitting on {len(precomputed)} labelled races "
          f"(budget={args.time_budget}s)...", flush=True)
    best_vec, best_s = fit(precomputed, time_budget_s=args.time_budget,
                           seed=args.seed, verbose=verbose, initial_vec=initial_vec)

    params = vec_to_params(best_vec)
    save_params(params, args.output)

    # Final evaluation
    ex_hist = ex_test = inv_test = 0
    if historical:
        pc_hist = [(df, fp) for df, fp in (precompute_race(r) for r in historical) if fp]
        ex_hist, _ = score(pc_hist, best_vec)
    if local_tests:
        pc_test = [(df, fp) for df, fp in (precompute_race(r) for r in local_tests) if fp]
        ex_test, inv_test = score(pc_test, best_vec)

    result = {
        "historical_races": len(historical),
        "local_tests":      len(local_tests),
        "history_exact":    ex_hist,
        "test_exact":       ex_test,
        "test_inversions":  inv_test,
        "output":           str(args.output),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
