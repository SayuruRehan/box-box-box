#!/usr/bin/env python3

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

from model import DEFAULT_PARAM_PATH, DEFAULT_PARAMS, ModelParams, save_params, score_races


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DIR = REPO_ROOT / "data" / "historical_races"
TEST_INPUT_DIR = REPO_ROOT / "data" / "test_cases" / "inputs"
TEST_EXPECTED_DIR = REPO_ROOT / "data" / "test_cases" / "expected_outputs"


def load_historical_races(limit=None):
    races = []
    for path in sorted(HISTORICAL_DIR.glob("races_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            races.extend(json.load(handle))
        if limit is not None and len(races) >= limit:
            return races[:limit]
    return races


def load_local_tests():
    races = []
    for path in sorted(TEST_INPUT_DIR.glob("test_*.json")):
        expected_path = TEST_EXPECTED_DIR / path.name
        if not expected_path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            race = json.load(handle)
        with open(expected_path, "r", encoding="utf-8") as handle:
            race["finishing_positions"] = json.load(handle)["finishing_positions"]
        races.append(race)
    return races


def flatten_params(params):
    vector = []
    for name in ("compound_offset", "linear_deg", "quadratic_deg", "fresh_bonus", "temp_sensitivity"):
        param_map = getattr(params, name)
        for compound in ("SOFT", "MEDIUM", "HARD"):
            vector.append(param_map[compound])
    vector.append(params.pit_penalty_scale)
    return vector


def unflatten_params(vector):
    cursor = 0
    sections = {}
    for name in ("compound_offset", "linear_deg", "quadratic_deg", "fresh_bonus", "temp_sensitivity"):
        sections[name] = {}
        for compound in ("SOFT", "MEDIUM", "HARD"):
            sections[name][compound] = vector[cursor]
            cursor += 1
    return ModelParams(
        compound_offset=sections["compound_offset"],
        linear_deg=sections["linear_deg"],
        quadratic_deg=sections["quadratic_deg"],
        fresh_bonus=sections["fresh_bonus"],
        temp_sensitivity=sections["temp_sensitivity"],
        pit_penalty_scale=vector[cursor],
    )


def parameter_bounds():
    return [
        (-2.0, 0.0),
        (-0.3, 0.4),
        (0.0, 1.5),
        (0.0, 0.15),
        (0.0, 0.12),
        (0.0, 0.10),
        (0.0, 0.02),
        (0.0, 0.01),
        (0.0, 0.006),
        (-0.4, 0.0),
        (-0.2, 0.0),
        (-0.1, 0.0),
        (0.0, 0.08),
        (0.0, 0.05),
        (0.0, 0.04),
        (0.95, 1.05),
    ]


def evaluate(vector, training_races, validation_races):
    params = unflatten_params(vector)
    train_matches, train_inversions = score_races(training_races, params)
    validation_matches, validation_inversions = score_races(validation_races, params)
    train_score = (train_matches * 1000000) - train_inversions
    validation_score = (validation_matches * 1000000) - validation_inversions
    return {
        "params": params,
        "vector": list(vector),
        "train_matches": train_matches,
        "train_inversions": train_inversions,
        "validation_matches": validation_matches,
        "validation_inversions": validation_inversions,
        "train_score": train_score,
        "validation_score": validation_score,
        "overall_score": train_score + validation_score,
    }


def mutate(vector, bounds, magnitude, randomizer):
    candidate = list(vector)
    for index, (lower, upper) in enumerate(bounds):
        span = upper - lower
        step = randomizer.uniform(-magnitude, magnitude) * span
        candidate[index] = min(upper, max(lower, candidate[index] + step))
    return candidate


def random_vector(bounds, randomizer):
    return [randomizer.uniform(lower, upper) for lower, upper in bounds]


def search(training_races, validation_races, iterations, seed):
    bounds = parameter_bounds()
    randomizer = random.Random(seed)
    best = evaluate(flatten_params(DEFAULT_PARAMS), training_races, validation_races)

    for _ in range(40):
        candidate = evaluate(random_vector(bounds, randomizer), training_races, validation_races)
        if candidate["overall_score"] > best["overall_score"]:
            best = candidate

    for magnitude in (0.30, 0.18, 0.10, 0.05, 0.025):
        stagnant_rounds = 0
        for _ in range(iterations):
            candidate_vector = mutate(best["vector"], bounds, magnitude, randomizer)
            candidate = evaluate(candidate_vector, training_races, validation_races)
            if candidate["overall_score"] > best["overall_score"]:
                best = candidate
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
            if stagnant_rounds > max(30, iterations // 4):
                break

    return best


def build_datasets(history_limit, local_test_weight):
    historical_races = load_historical_races(limit=history_limit)
    split_index = max(1, int(len(historical_races) * 0.85))
    training_races = historical_races[:split_index]
    validation_races = historical_races[split_index:]
    local_tests = load_local_tests()
    if local_test_weight > 0 and local_tests:
        weighted_tests = []
        for _ in range(local_test_weight):
            weighted_tests.extend(deepcopy(local_tests))
        validation_races = validation_races + weighted_tests
    return training_races, validation_races, len(historical_races), len(local_tests)


def main():
    parser = argparse.ArgumentParser(description="Fit Box Box Box race model parameters")
    parser.add_argument("--history-limit", type=int, default=6000)
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local-test-weight", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_PARAM_PATH)
    args = parser.parse_args()

    training_races, validation_races, historical_count, local_test_count = build_datasets(
        args.history_limit,
        args.local_test_weight,
    )

    best = search(training_races, validation_races, args.iterations, args.seed)
    save_params(best["params"], args.output)

    print(
        json.dumps(
            {
                "historical_count": historical_count,
                "local_test_count": local_test_count,
                "train_matches": best["train_matches"],
                "train_inversions": best["train_inversions"],
                "validation_matches": best["validation_matches"],
                "validation_inversions": best["validation_inversions"],
                "params": {
                    "compound_offset": best["params"].compound_offset,
                    "linear_deg": best["params"].linear_deg,
                    "quadratic_deg": best["params"].quadratic_deg,
                    "fresh_bonus": best["params"].fresh_bonus,
                    "temp_sensitivity": best["params"].temp_sensitivity,
                    "pit_penalty_scale": best["params"].pit_penalty_scale,
                },
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()