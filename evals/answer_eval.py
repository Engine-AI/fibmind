"""Attach independently judged model answers and provider token usage to a retrieval report."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def score_answers(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    """Score recorded answers without treating context term matches as correctness.

    Answers are collected by an external harness. ``correct`` must be judged
    against the user's question, and token counts must come from the provider;
    the retrieval runner cannot infer either from its own context text.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = str(payload.get("model") or "").strip()
    judged_by = str(payload.get("judged_by") or "").strip()
    rows = payload.get("answers")
    if not model or not judged_by or not isinstance(rows, list) or not rows:
        raise ValueError("answer file needs model, judged_by, and non-empty answers")

    known = {(row["baseline"], row["case_id"]) for row in report["cases"]}
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("answer entries must be objects")
        key = (row.get("baseline"), row.get("case_id"))
        if key not in known or key in seen:
            raise ValueError(f"unknown or duplicate answer: {key}")
        seen.add(key)
        if not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError(f"answer {key} needs the model's non-empty answer text")
        if type(row.get("correct")) is not bool:
            raise ValueError(f"answer {key} needs an independently judged boolean correct")
        for field in ("input_tokens", "output_tokens"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"answer {key} needs a non-negative provider {field}")
        grouped.setdefault(str(key[0]), []).append(row)

    by_baseline = {}
    for baseline, judged in grouped.items():
        expected = sum(row["baseline"] == baseline for row in report["cases"])
        by_baseline[baseline] = {
            "judged_cases": len(judged),
            "coverage": round(len(judged) / expected, 4),
            "answer_accuracy": round(mean(row["correct"] for row in judged), 4),
            "mean_input_tokens": round(mean(row["input_tokens"] for row in judged), 2),
            "mean_output_tokens": round(mean(row["output_tokens"] for row in judged), 2),
            "total_input_tokens": sum(row["input_tokens"] for row in judged),
            "total_output_tokens": sum(row["output_tokens"] for row in judged),
        }
    return {"model": model, "judged_by": judged_by, "by_baseline": by_baseline}
