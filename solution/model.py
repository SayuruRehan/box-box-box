#!/usr/bin/env python3

import json
from dataclasses import dataclass
from pathlib import Path


COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
REFERENCE_TEMP = 30.0
DEFAULT_PARAM_PATH = Path(__file__).with_name("model_params.json")


@dataclass(frozen=True)
class ModelParams:
    compound_offset: dict
    linear_deg: dict
    quadratic_deg: dict
    cubic_deg: dict
    fresh_bonus: dict
    temp_sensitivity: dict
    pit_penalty_scale: float


DEFAULT_PARAMS = ModelParams(
    compound_offset={"SOFT": -0.85, "MEDIUM": 0.0, "HARD": 0.62},
    linear_deg={"SOFT": 0.06, "MEDIUM": 0.038, "HARD": 0.028},
    quadratic_deg={"SOFT": 0.0042, "MEDIUM": 0.0022, "HARD": 0.0010},
    cubic_deg={"SOFT": 0.00012, "MEDIUM": 0.00003, "HARD": 0.00001},
    fresh_bonus={"SOFT": -0.10, "MEDIUM": -0.05, "HARD": -0.02},
    temp_sensitivity={"SOFT": 0.022, "MEDIUM": 0.015, "HARD": 0.010},
    pit_penalty_scale=1.0,
)


def params_to_dict(params):
    return {
        "compound_offset": dict(params.compound_offset),
        "linear_deg": dict(params.linear_deg),
        "quadratic_deg": dict(params.quadratic_deg),
        "cubic_deg": dict(params.cubic_deg),
        "fresh_bonus": dict(params.fresh_bonus),
        "temp_sensitivity": dict(params.temp_sensitivity),
        "pit_penalty_scale": params.pit_penalty_scale,
    }


def load_params(path=DEFAULT_PARAM_PATH):
    if not Path(path).exists():
        return DEFAULT_PARAMS
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ModelParams(
        compound_offset=data["compound_offset"],
        linear_deg=data["linear_deg"],
        quadratic_deg=data["quadratic_deg"],
        cubic_deg=data.get("cubic_deg", DEFAULT_PARAMS.cubic_deg),
        fresh_bonus=data["fresh_bonus"],
        temp_sensitivity=data["temp_sensitivity"],
        pit_penalty_scale=float(data.get("pit_penalty_scale", 1.0)),
    )


def save_params(params, path=DEFAULT_PARAM_PATH):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params_to_dict(params), handle, indent=2, sort_keys=True)
        handle.write("\n")


def iter_drivers(strategies):
    for grid_slot in sorted(strategies.keys(), key=lambda name: int(name[3:])):
        yield strategies[grid_slot]


def lap_time_delta(params, compound, tire_age, track_temp):
    age_index = tire_age - 1
    temp_factor = 1.0 + ((track_temp - REFERENCE_TEMP) * params.temp_sensitivity[compound] / 10.0)
    temp_factor = max(temp_factor, 0.2)
    linear_term = params.linear_deg[compound] * age_index
    quadratic_term = params.quadratic_deg[compound] * age_index * age_index
    cubic_term = params.cubic_deg[compound] * age_index * age_index * age_index
    bonus = params.fresh_bonus[compound] if tire_age == 1 else 0.0
    return params.compound_offset[compound] + bonus + temp_factor * (linear_term + quadratic_term + cubic_term)


def _build_stints(strategy, total_laps):
    """Return [(compound, num_laps), ...] in pit-stop order."""
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


def simulate_driver_time(race_config, strategy, params):
    """Compute total race time using closed-form stint summation.

    Closed-form avoids floating-point drift from per-lap loops. This means
    drivers with analytically equal total times produce identical IEEE-754
    values, so grid-position tie-breaking works correctly.
    """
    total_laps    = race_config["total_laps"]
    base_lap_time = race_config["base_lap_time"]
    pit_lane_time = race_config["pit_lane_time"] * params.pit_penalty_scale
    track_temp    = race_config["track_temp"]

    total_time = total_laps * base_lap_time
    total_time += len(strategy["pit_stops"]) * pit_lane_time

    for compound, n in _build_stints(strategy, total_laps):
        temp_factor = 1.0 + (track_temp - REFERENCE_TEMP) * params.temp_sensitivity[compound] / 10.0
        temp_factor = max(temp_factor, 0.2)
        # Closed-form: Σ(age-1) for age=1..n  = n*(n-1)/2
        # Closed-form: Σ(age-1)² for age=1..n = n*(n-1)*(2n-1)/6
        # Closed-form: Σ(age-1)^3 for age=1..n = (n*(n-1)/2)^2
        lin_sum  = n * (n - 1) / 2
        quad_sum = n * (n - 1) * (2 * n - 1) / 6
        cube_sum = lin_sum * lin_sum
        total_time += (n * params.compound_offset[compound]
                       + params.fresh_bonus[compound]
                       + temp_factor * (params.linear_deg[compound] * lin_sum
                           + params.quadratic_deg[compound] * quad_sum
                           + params.cubic_deg[compound] * cube_sum))

    return total_time


def simulate_race(race, params):
    race_config = race["race_config"]
    results = []
    for grid_pos, strategy in enumerate(iter_drivers(race["strategies"]), start=1):
        driver_id = strategy["driver_id"]
        total_time = simulate_driver_time(race_config, strategy, params)
        # Tie-break by grid position (lower grid slot = better when times are equal)
        results.append((total_time, grid_pos, driver_id))
    results.sort()
    return [driver_id for _, _, driver_id in results]


def inversion_count(predicted_order, actual_order):
    actual_index = {driver_id: index for index, driver_id in enumerate(actual_order)}
    inversions = 0
    for left_index, left_driver in enumerate(predicted_order):
        left_actual = actual_index[left_driver]
        for right_driver in predicted_order[left_index + 1 :]:
            if left_actual > actual_index[right_driver]:
                inversions += 1
    return inversions


def exact_match_count(races, params):
    matches = 0
    for race in races:
        if simulate_race(race, params) == race["finishing_positions"]:
            matches += 1
    return matches


def score_races(races, params):
    exact_matches = 0
    total_inversions = 0
    for race in races:
        predicted_order = simulate_race(race, params)
        actual_order = race["finishing_positions"]
        if predicted_order == actual_order:
            exact_matches += 1
        total_inversions += inversion_count(predicted_order, actual_order)
    return exact_matches, total_inversions