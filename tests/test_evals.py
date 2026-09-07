from pathlib import Path

import pytest

from evals.baselines import EvaluationCase, RetrievalResult, load_datasets
from evals.metrics import aggregate_metrics, estimate_tokens, evaluate_case
from evals.runner import DEFAULT_DATASET_DIR, render_table, run_evaluation


def test_bundled_datasets_are_valid_and_stable() -> None:
    datasets = load_datasets(DEFAULT_DATASET_DIR)

    assert [dataset.name for dataset in datasets] == [
        "conflicting_memory",
        "cross_project",
        "cross_session",
        "project_history",
        "session_review",
        "stale_memory",
    ]
    assert sum(len(dataset.cases) for dataset in datasets) == 17


def test_dataset_rejects_unknown_memory_reference(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        """{
          "name": "bad",
          "memories": [{"id": "known", "title": "Known", "content": "text"}],
          "cases": [{"id": "case", "query": "query", "relevant_ids": ["missing"]}]
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown memories"):
        load_datasets(tmp_path)


def test_estimated_tokens_are_deterministic_and_cjk_aware() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("记忆") == 2
    assert estimate_tokens("hello, world") == 5


def test_case_metrics_distinguish_relevance_staleness_and_forbidden_hits() -> None:
    case = EvaluationCase(
        id="dataset/case",
        query="current choice",
        relevant_ids=("current",),
        stale_ids=("old",),
        forbidden_ids=("old",),
        expected_answer_terms=("SQLite",),
    )
    result = RetrievalResult(
        baseline="test",
        case_id=case.id,
        retrieved_ids=("old", "current", "noise"),
        text="SQLite is current",
        latency_ms=2.0,
    )

    metric = evaluate_case(case, result, top_k=3)

    assert metric.recall_at_k == 1.0
    assert metric.reciprocal_rank == 0.5
    assert metric.answerability == 1.0
    assert metric.irrelevant_injection_rate == pytest.approx(2 / 3)
    assert metric.stale_pollution_rate == pytest.approx(1 / 3)
    assert metric.forbidden_hit == 1.0


def test_no_relevant_case_is_reported_separately() -> None:
    case = EvaluationCase(id="dataset/negative", query="none", relevant_ids=())
    result = RetrievalResult(
        baseline="test",
        case_id=case.id,
        retrieved_ids=(),
        text="",
        latency_ms=0.1,
    )

    metric = evaluate_case(case, result, top_k=5)
    summary = aggregate_metrics([metric])

    assert metric.recall_at_k is None
    assert metric.no_relevant_accuracy == 1.0
    assert summary["recall_at_k"] is None
    assert summary["no_relevant_accuracy"] == 1.0


def test_full_evaluation_compares_all_p0_baselines() -> None:
    report = run_evaluation(DEFAULT_DATASET_DIR, top_k=5)

    assert report["configuration"]["dataset_count"] == 6
    assert report["configuration"]["case_count"] == 17
    assert set(report["baselines"]) == {
        "no_memory",
        "agents_md",
        "hermes_hot",
        "fibmind_current",
        "fibmind_hybrid",
        "fibmind_budgeted",
    }
    assert report["baselines"]["no_memory"]["total_estimated_tokens"] == 0
    assert report["baselines"]["fibmind_current"]["recall_at_k"] == 1.0
    # Lifecycle filtering and strict owner visibility are pinned by the same
    # benchmark that originally exposed stale/forbidden-memory pollution.
    current = report["baselines"]["fibmind_current"]
    assert current["stale_pollution_rate"] == 0.0
    assert current["forbidden_case_rate"] == 0.0
    assert current["no_relevant_accuracy"] == 1.0
    assert current["irrelevant_injection_rate"] <= 0.2

    table = render_table(report)
    assert "fibmind_current" in table
    assert "forbidden" in table


def test_baseline_selection_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown baselines"):
        run_evaluation(DEFAULT_DATASET_DIR, baseline_names=["does-not-exist"])
