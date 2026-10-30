"""Fail when the fixed evaluation suite regresses below a committed floor.

``fibbrain_tune`` compares two weight sets against each other; this module is
the CI half of the same idea: the current code must not fall below the numbers
in ``evals/floor.json``. Raise the floor deliberately when the suite improves;
never lower it to make a build pass.

Usage::

    python -m evals.gate                # default floor and datasets
    python -m evals.gate --floor other.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.runner import DEFAULT_DATASET_DIR, run_evaluation

DEFAULT_FLOOR = Path(__file__).resolve().parent / "floor.json"


def check_floor(summary: dict[str, Any], floor: dict[str, Any]) -> list[str]:
    """Return one line per metric that violates the floor; empty means pass."""
    failures: list[str] = []
    for key, minimum in floor.get("min", {}).items():
        value = summary.get(key)
        if value is None or value < minimum:
            failures.append(f"{key}: {value} < floor {minimum}")
    for key, maximum in floor.get("max", {}).items():
        value = summary.get(key)
        if value is None or value > maximum:
            failures.append(f"{key}: {value} > ceiling {maximum}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--floor", type=Path, default=DEFAULT_FLOOR)
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args(argv)

    floor = json.loads(args.floor.read_text(encoding="utf-8"))
    baseline = floor.get("baseline", "fibmind_current")
    report = run_evaluation(args.datasets, baseline_names=[baseline])
    summary = report["baselines"][baseline]
    failures = check_floor(summary, floor)

    watched = sorted({*floor.get("min", {}), *floor.get("max", {})})
    for key in watched:
        print(f"{baseline}.{key} = {summary.get(key)}")

    if "gaps" in floor:
        from evals.gaps_runner import run_gaps_evaluation

        gaps_floor = floor["gaps"]
        gaps_summary = run_gaps_evaluation()
        print(f"gaps.verdict_accuracy = {gaps_summary.get('verdict_accuracy')}")
        print(f"gaps.false_known_rate = {gaps_summary.get('false_known_rate')}")

        max_false_known = gaps_floor.get("false_known_rate", 0.0)
        min_verdict_acc = gaps_floor.get("verdict_accuracy", 0.9)

        if gaps_summary.get("false_known_rate", 0.0) > max_false_known:
            failures.append(
                f"gaps.false_known_rate: {gaps_summary.get('false_known_rate')} > ceiling {max_false_known}"
            )
        if gaps_summary.get("verdict_accuracy", 0.0) < min_verdict_acc:
            failures.append(f"gaps.verdict_accuracy: {gaps_summary.get('verdict_accuracy')} < floor {min_verdict_acc}")

    if "kernel" in floor:
        from evals.kernel_runner import run_kernel_evaluation

        kernel_floor = floor["kernel"]
        kernel_summary = run_kernel_evaluation()
        for key in sorted(kernel_floor):
            val = kernel_summary.get(key, 0.0)
            print(f"kernel.{key} = {val}")
            minimum = kernel_floor[key]
            if val < minimum:
                failures.append(f"kernel.{key}: {val} < floor {minimum}")

    if failures:
        print("\nevaluation floor violated:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print("\nevaluation floor holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
