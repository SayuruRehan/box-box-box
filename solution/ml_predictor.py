#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None


COMPOUND_INDEX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
DEFAULT_MODEL_PATH = Path(__file__).with_name("model_hgb.joblib")
DEFAULT_META_PATH = Path(__file__).with_name("model_hgb_meta.json")


def _build_driver_features(race_config, strategy, tracks, track_index):
    total_laps = race_config["total_laps"]
    track_temp = race_config["track_temp"]
    temp_delta = track_temp - 30.0
    base_lap_time = race_config["base_lap_time"]
    pit_lane_time = race_config["pit_lane_time"]

    laps = np.zeros(3, dtype=np.float32)
    stints = np.zeros(3, dtype=np.float32)
    age1 = np.zeros(3, dtype=np.float32)
    age2 = np.zeros(3, dtype=np.float32)
    abs1 = np.zeros(3, dtype=np.float32)

    current = strategy["starting_tire"]
    lap_ptr = 1
    for stop in sorted(strategy["pit_stops"], key=lambda s: s["lap"]):
        a, b = lap_ptr, stop["lap"]
        n = b - a + 1
        if n > 0:
            ci = COMPOUND_INDEX[current]
            laps[ci] += n
            stints[ci] += 1
            age1[ci] += n * (n - 1) / 2.0
            age2[ci] += n * (n - 1) * (2 * n - 1) / 6.0
            abs1[ci] += (a + b - 2) * n / 2.0
        current = stop["to_tire"]
        lap_ptr = stop["lap"] + 1

    a, b = lap_ptr, total_laps
    n = b - a + 1
    if n > 0:
        ci = COMPOUND_INDEX[current]
        laps[ci] += n
        stints[ci] += 1
        age1[ci] += n * (n - 1) / 2.0
        age2[ci] += n * (n - 1) * (2 * n - 1) / 6.0
        abs1[ci] += (a + b - 2) * n / 2.0

    feats = []
    feats += [
        float(total_laps),
        float(track_temp),
        float(base_lap_time),
        float(pit_lane_time),
        float(len(strategy["pit_stops"])),
    ]

    track_onehot = [0.0] * len(tracks)
    idx = track_index.get(race_config["track"])
    if idx is not None:
        track_onehot[idx] = 1.0
    feats += track_onehot

    for arr in (laps, stints, age1, age2, abs1):
        feats += arr.tolist()

    for arr in (laps, age1, abs1):
        feats += (arr * temp_delta).tolist()

    feats += (laps * pit_lane_time).tolist()
    return np.asarray(feats, dtype=np.float32)


def load_ml_model(model_path=DEFAULT_MODEL_PATH, meta_path=DEFAULT_META_PATH):
    if joblib is None:
        return None
    if not model_path.exists() or not meta_path.exists():
        return None

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    tracks = meta["tracks"]
    track_index = {name: i for i, name in enumerate(tracks)}
    return {
        "model": model,
        "tracks": tracks,
        "track_index": track_index,
    }


def predict_finishing_positions_ml(race, ml_bundle):
    race_config = race["race_config"]
    drivers = []
    for pos_key in sorted(race["strategies"], key=lambda s: int(s[3:])):
        strategy = race["strategies"][pos_key]
        features = _build_driver_features(
            race_config,
            strategy,
            ml_bundle["tracks"],
            ml_bundle["track_index"],
        )
        score = float(ml_bundle["model"].predict(features.reshape(1, -1))[0])
        drivers.append((score, strategy["driver_id"]))

    drivers.sort(key=lambda row: row[0])
    return [driver_id for _, driver_id in drivers]
