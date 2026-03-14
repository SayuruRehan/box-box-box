#!/usr/bin/env python3

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ml_predictor import _build_driver_features


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DIR = REPO_ROOT / "data" / "historical_races"
MODEL_PATH = Path(__file__).with_name("model_hgb.joblib")
META_PATH = Path(__file__).with_name("model_hgb_meta.json")


def iter_historical_races():
    for race_file in sorted(HISTORICAL_DIR.glob("races_*.json")):
        with open(race_file, "r", encoding="utf-8") as handle:
            races = json.load(handle)
        for race in races:
            yield race


def collect_tracks():
    tracks = set()
    for race in iter_historical_races():
        tracks.add(race["race_config"]["track"])
    return sorted(tracks)


def build_training_set(tracks):
    track_index = {name: i for i, name in enumerate(tracks)}
    X = []
    y = []

    for race in iter_historical_races():
        rank = {driver_id: i + 1 for i, driver_id in enumerate(race["finishing_positions"])}
        for pos_key in sorted(race["strategies"], key=lambda s: int(s[3:])):
            strategy = race["strategies"][pos_key]
            X.append(
                _build_driver_features(
                    race["race_config"],
                    strategy,
                    tracks,
                    track_index,
                )
            )
            y.append(rank[strategy["driver_id"]])

    return np.vstack(X), np.asarray(y, dtype=np.float32)


def main():
    tracks = collect_tracks()
    X, y = build_training_set(tracks)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_depth=8,
        max_iter=220,
        min_samples_leaf=40,
        random_state=0,
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump({"tracks": tracks, "feature_dim": int(X.shape[1])}, handle, indent=2)
        handle.write("\n")

    print(json.dumps({"model_path": str(MODEL_PATH), "meta_path": str(META_PATH), "samples": int(X.shape[0])}))


if __name__ == "__main__":
    main()
