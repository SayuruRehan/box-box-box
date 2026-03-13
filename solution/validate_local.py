#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = REPO_ROOT / "data" / "test_cases" / "expected_outputs"
COMMAND = [sys.executable, str(REPO_ROOT / "solution" / "race_simulator.py")]


def run_case(input_path):
    expected_path = EXPECTED_DIR / input_path.name
    with open(input_path, "rb") as handle:
        result = subprocess.run(COMMAND, stdin=handle, capture_output=True, check=True)
    predicted = json.loads(result.stdout)
    with open(expected_path, "r", encoding="utf-8") as handle:
        expected = json.load(handle)
    return predicted["finishing_positions"] == expected["finishing_positions"]


def main():
    input_paths = sorted(INPUT_DIR.glob("test_*.json"))
    passed = 0
    total = len(input_paths)
    for input_path in input_paths:
        case_name = input_path.stem.upper()
        is_match = run_case(input_path)
        status = "PASS" if is_match else "FAIL"
        print(f"{status} {case_name}")
        if is_match:
            passed += 1
    print(json.dumps({"passed": passed, "total": total, "pass_rate": passed / total if total else 0.0}))


if __name__ == "__main__":
    main()
