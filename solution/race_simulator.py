#!/usr/bin/env python3

import json
import sys

from model import load_params, simulate_race

_params = None


def get_params():
    global _params
    if _params is None:
        _params = load_params()
    return _params


def main():
    race = json.load(sys.stdin)
    finishing_positions = simulate_race(race, get_params())
    print(json.dumps({"race_id": race["race_id"],
                      "finishing_positions": finishing_positions}))


if __name__ == "__main__":
    main()