"""Command-line runner for FibMind's reproducible P0 evaluation suite."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evals.baselines import BASELINE_TYPES, Baseline, load_datasets
from evals.metrics import CaseMetrics, aggregate_metrics, evaluate_case


DEFAULT_DATASET_DIR = Path(__file__).with_name("datasets")


def run_evaluation(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    *,
    top_k: int = 5,
    max_context_chars: int = 5200,
    baseline_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run selected baselines and return a JSON-serializable report."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    datasets = load_datasets(dataset_dir)
    selected = set(baseline_names or [])
    known = {baseline_type.name for baseline_type in BASELINE_TYPES}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown baselines: {sorted(unknown)}")

    summaries: dict[str, dict[str, int | float | None]] = {}
    case_rows: list[dict[str, object]] = []
    for baseline_type in BASELINE_TYPES:
        if selected and baseline_type.name not in selected:
            continue
        baseline_metrics: list[CaseMetrics] = []
        for dataset in datasets:
            baseline: Baseline = baseline_type(
                dataset,
                max_context_chars=max_context_chars,
            )
            for case in dataset.cases:
                result = baseline.retrieve(case, top_k=top_k)
                metric = evaluate_case(case, result, top_k=top_k)
                baseline_metrics.append(metric)
                case_rows.append(metric.to_dict())
        summaries[baseline_type.name] = aggregate_metrics(baseline_metrics)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset_dir": str(Path(dataset_dir)),
            "dataset_count": len(datasets),
            "case_count": sum(len(dataset.cases) for dataset in datasets),
            "top_k": top_k,
            "max_context_chars": max_context_chars,
            "token_metric": "deterministic estimate; replace with model tokenizer in P3",
        },
        "baselines": summaries,
        "cases": case_rows,
    }


def render_table(report: dict[str, Any]) -> str:
    """Render the aggregate report as a compact terminal table."""
    headers = (
        "baseline",
        "R@K",
        "MRR",
        "answer",
        "no-rel",
        "irrelevant",
        "stale",
        "forbidden",
        "tokens",
        "p95-ms",
    )
    rows = []
    for name, summary in report["baselines"].items():
        rows.append(
            (
                name,
                _format(summary["recall_at_k"]),
                _format(summary["mrr"]),
                _format(summary["answerability_rate"]),
                _format(summary["no_relevant_accuracy"]),
                _format(summary["irrelevant_injection_rate"]),
                _format(summary["stale_pollution_rate"]),
                _format(summary["forbidden_case_rate"]),
                _format(summary["mean_estimated_tokens"], digits=2),
                _format(summary["latency_p95_ms"], digits=4),
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    rendered = [_render_row(headers, widths), _render_row(tuple("-" * width for width in widths), widths)]
    rendered.extend(_render_row(row, widths) for row in rows)
    return "\n".join(rendered)


def _render_row(row: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))


def _format(value: int | float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FibMind retrieval evaluations")
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=5200)
    parser.add_argument(
        "--baseline",
        action="append",
        choices=[baseline_type.name for baseline_type in BASELINE_TYPES],
        dest="baselines",
        help="Run only this baseline; repeat to select several",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--output", type=Path, help="Also write the full JSON report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = run_evaluation(
        args.datasets,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        baseline_names=args.baselines,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_table(report))


if __name__ == "__main__":
    main()
