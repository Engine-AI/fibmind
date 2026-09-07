"""Dataset loading and comparable memory-retrieval baselines.

The baselines deliberately share one result shape. Future FTS5, embedding, and
hybrid retrievers can therefore be measured without changing the datasets or
metric implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from fibmind.context import build_context
from fibmind.embedding import EmbeddingCache, HashingEmbeddingProvider
from fibmind.graph import FibMind
from fibmind.models import MemoryScope


@dataclass(frozen=True, slots=True)
class MemoryFixture:
    """One stable, human-authored memory in an evaluation corpus."""

    id: str
    title: str
    content: str
    category: str = "general"
    tags: tuple[str, ...] = ()
    scope: str = MemoryScope.PERSONAL.value
    owner: str | None = None
    workspace_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    hot: bool = False
    agents_md: bool = False
    stale: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryFixture":
        required = ("id", "title", "content")
        missing = [key for key in required if not str(data.get(key, "")).strip()]
        if missing:
            raise ValueError(f"memory fixture is missing non-empty fields: {', '.join(missing)}")
        scope = str(data.get("scope", MemoryScope.PERSONAL.value))
        MemoryScope(scope)
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            content=str(data["content"]),
            category=str(data.get("category", "general")),
            tags=tuple(str(tag) for tag in data.get("tags", [])),
            scope=scope,
            owner=data.get("owner"),
            workspace_id=data.get("workspace_id"),
            project_id=data.get("project_id"),
            session_id=data.get("session_id"),
            task_id=data.get("task_id"),
            hot=bool(data.get("hot", False)),
            agents_md=bool(data.get("agents_md", False)),
            stale=bool(data.get("stale", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One query and the retrieval behaviour expected for it."""

    id: str
    query: str
    relevant_ids: tuple[str, ...]
    stale_ids: tuple[str, ...] = ()
    forbidden_ids: tuple[str, ...] = ()
    expected_answer_terms: tuple[str, ...] = ()
    owner: str | None = None
    workspace_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    scopes: tuple[str, ...] | None = None
    top_k: int | None = None

    @classmethod
    def from_dict(cls, dataset_name: str, data: dict[str, Any]) -> "EvaluationCase":
        case_id = str(data.get("id", "")).strip()
        query = str(data.get("query", "")).strip()
        if not case_id or not query:
            raise ValueError(f"dataset {dataset_name!r} contains a case without id or query")
        scopes = data.get("scopes")
        if scopes is not None:
            scopes = tuple(str(scope) for scope in scopes)
            for scope in scopes:
                MemoryScope(scope)
        top_k = data.get("top_k")
        if top_k is not None and int(top_k) < 1:
            raise ValueError(f"case {case_id!r} top_k must be positive")
        return cls(
            id=f"{dataset_name}/{case_id}",
            query=query,
            relevant_ids=tuple(str(value) for value in data.get("relevant_ids", [])),
            stale_ids=tuple(str(value) for value in data.get("stale_ids", [])),
            forbidden_ids=tuple(str(value) for value in data.get("forbidden_ids", [])),
            expected_answer_terms=tuple(
                str(value) for value in data.get("expected_answer_terms", [])
            ),
            owner=data.get("owner"),
            workspace_id=data.get("workspace_id"),
            project_id=data.get("project_id"),
            session_id=data.get("session_id"),
            task_id=data.get("task_id"),
            scopes=scopes,
            top_k=int(top_k) if top_k is not None else None,
        )


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A memory corpus and the queries evaluated against that corpus."""

    name: str
    description: str
    memories: tuple[MemoryFixture, ...]
    cases: tuple[EvaluationCase, ...]
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> "EvaluationDataset":
        data = json.loads(path.read_text(encoding="utf-8"))
        name = str(data.get("name", path.stem)).strip()
        memories = tuple(MemoryFixture.from_dict(item) for item in data.get("memories", []))
        cases = tuple(EvaluationCase.from_dict(name, item) for item in data.get("cases", []))
        if not memories:
            raise ValueError(f"dataset {name!r} has no memories")
        if not cases:
            raise ValueError(f"dataset {name!r} has no cases")

        memory_ids = [memory.id for memory in memories]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError(f"dataset {name!r} contains duplicate memory ids")
        known = set(memory_ids)
        for case in cases:
            referenced = set(case.relevant_ids + case.stale_ids + case.forbidden_ids)
            unknown = referenced - known
            if unknown:
                raise ValueError(
                    f"case {case.id!r} references unknown memories: {sorted(unknown)}"
                )
        return cls(
            name=name,
            description=str(data.get("description", "")),
            memories=memories,
            cases=cases,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Normalized output produced by every baseline."""

    baseline: str
    case_id: str
    retrieved_ids: tuple[str, ...]
    text: str
    latency_ms: float


def load_datasets(directory: str | Path) -> list[EvaluationDataset]:
    """Load and validate all JSON datasets in stable filename order."""
    dataset_dir = Path(directory)
    paths = sorted(dataset_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no evaluation datasets found in {dataset_dir}")
    datasets = [EvaluationDataset.from_path(path) for path in paths]
    names = [dataset.name for dataset in datasets]
    if len(names) != len(set(names)):
        raise ValueError("evaluation dataset names must be unique")
    return datasets


class Baseline:
    """Base class for one retrieval strategy over a single dataset."""

    name = "baseline"

    def __init__(self, dataset: EvaluationDataset, max_context_chars: int = 5200) -> None:
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be positive")
        self.dataset = dataset
        self.max_context_chars = max_context_chars

    def retrieve(self, case: EvaluationCase, top_k: int) -> RetrievalResult:
        raise NotImplementedError

    def _result(
        self,
        case: EvaluationCase,
        started_at: float,
        retrieved_ids: Iterable[str],
        text: str,
    ) -> RetrievalResult:
        return RetrievalResult(
            baseline=self.name,
            case_id=case.id,
            retrieved_ids=tuple(retrieved_ids),
            text=text,
            latency_ms=(perf_counter() - started_at) * 1000,
        )


class NoMemoryBaseline(Baseline):
    """Control group: answer without any persisted context."""

    name = "no_memory"

    def retrieve(self, case: EvaluationCase, top_k: int) -> RetrievalResult:
        started_at = perf_counter()
        return self._result(case, started_at, (), "")


class _StaticBaseline(Baseline):
    fixture_flag = ""

    def retrieve(self, case: EvaluationCase, top_k: int) -> RetrievalResult:
        started_at = perf_counter()
        selected = [
            memory
            for memory in self.dataset.memories
            if getattr(memory, self.fixture_flag) and _is_visible(memory, case)
        ]
        selected = selected[:top_k]
        ids, text = _render_memories(selected, self.max_context_chars)
        return self._result(case, started_at, ids, text)


class AgentsMdBaseline(_StaticBaseline):
    """Static project instructions injected for every query."""

    name = "agents_md"
    fixture_flag = "agents_md"


class HermesHotBaseline(_StaticBaseline):
    """Bounded, always-on hot-memory snapshot."""

    name = "hermes_hot"
    fixture_flag = "hot"


class FibMindCurrentBaseline(Baseline):
    """The repository's current keyword search plus graph context expansion."""

    name = "fibmind_current"

    def __init__(self, dataset: EvaluationDataset, max_context_chars: int = 5200) -> None:
        super().__init__(dataset, max_context_chars=max_context_chars)
        self.memory = FibMind()
        self.node_to_fixture: dict[str, str] = {}
        for fixture in dataset.memories:
            node_id = self.memory.append(
                category=fixture.category,
                title=fixture.title,
                content=fixture.content,
                tags=fixture.tags,
                metadata={
                    **fixture.metadata,
                    "eval_id": fixture.id,
                    "eval_stale": fixture.stale,
                },
                scope=MemoryScope(fixture.scope),
                owner=fixture.owner,
                workspace_id=fixture.workspace_id,
                project_id=fixture.project_id,
                session_id=fixture.session_id,
                task_id=fixture.task_id,
            )
            self.node_to_fixture[node_id] = fixture.id
            if fixture.stale:
                self.memory.mark_stale(node_id, "evaluation fixture marks this memory stale")

    def retrieve(self, case: EvaluationCase, top_k: int) -> RetrievalResult:
        started_at = perf_counter()
        scopes = (
            {MemoryScope(scope) for scope in case.scopes}
            if case.scopes is not None
            else None
        )
        pack = build_context(
            self.memory,
            case.query,
            top_k=case.top_k or top_k,
            depth=1,
            max_chars=self.max_context_chars,
            reinforce=False,
            scopes=scopes,
            owner=case.owner,
            workspace_id=case.workspace_id,
            project_id=case.project_id,
            session_id=case.session_id,
        )
        ids: list[str] = []
        for hit in pack.hits:
            fixture_id = hit.node.metadata.get("eval_id")
            if fixture_id is not None and fixture_id not in ids:
                ids.append(str(fixture_id))
        return self._result(case, started_at, ids, pack.text)


class FibMindHybridBaseline(FibMindCurrentBaseline):
    """The same engine with the offline hashing vector provider fused in.

    Measures whether the fusion path helps or hurts on the fixed datasets.
    The hashing provider is not semantic, so a real embedding model can only
    be judged on the imported real-session datasets, not here.
    """

    name = "fibmind_hybrid"

    def __init__(self, dataset: EvaluationDataset, max_context_chars: int = 5200) -> None:
        super().__init__(dataset, max_context_chars=max_context_chars)
        self.memory.embeddings = EmbeddingCache(HashingEmbeddingProvider())


class FibMindBudgetedBaseline(FibMindCurrentBaseline):
    """The lexical engine under a hard token budget, with the dataset's ``hot``
    fixtures rendered first as the L0 preamble.

    This is the P3 shape: a bounded preamble plus on-demand recall, measured in
    tokens. The budget is deliberately tight so the report shows what a
    budget costs in recall, not just that it is enforced.
    """

    name = "fibmind_budgeted"
    budget_tokens = 120

    def retrieve(self, case: EvaluationCase, top_k: int) -> RetrievalResult:
        started_at = perf_counter()
        scopes = (
            {MemoryScope(scope) for scope in case.scopes}
            if case.scopes is not None
            else None
        )
        hot_nodes = [
            self.memory.nodes[node_id]
            for node_id, fixture_id in self.node_to_fixture.items()
            if self._fixture(fixture_id).hot and _is_visible(self._fixture(fixture_id), case)
        ]
        pack = build_context(
            self.memory,
            case.query,
            top_k=case.top_k or top_k,
            depth=1,
            max_chars=self.max_context_chars,
            budget_tokens=self.budget_tokens,
            hot=hot_nodes,
            reinforce=False,
            scopes=scopes,
            owner=case.owner,
            workspace_id=case.workspace_id,
            project_id=case.project_id,
            session_id=case.session_id,
        )
        assert pack.budget is not None and pack.budget.used_tokens <= self.budget_tokens
        ids: list[str] = []
        for hit in [*pack.hot, *pack.hits]:
            fixture_id = hit.node.metadata.get("eval_id")
            if fixture_id is not None and fixture_id not in ids:
                ids.append(str(fixture_id))
        return self._result(case, started_at, ids, pack.text)

    def _fixture(self, fixture_id: str) -> MemoryFixture:
        return next(memory for memory in self.dataset.memories if memory.id == fixture_id)


BASELINE_TYPES: tuple[type[Baseline], ...] = (
    NoMemoryBaseline,
    AgentsMdBaseline,
    HermesHotBaseline,
    FibMindCurrentBaseline,
    FibMindHybridBaseline,
    FibMindBudgetedBaseline,
)


def _is_visible(memory: MemoryFixture, case: EvaluationCase) -> bool:
    if case.scopes is not None and memory.scope not in case.scopes:
        return False
    if memory.scope != MemoryScope.KNOWLEDGE.value and memory.owner != case.owner:
        return False
    if memory.scope == MemoryScope.SESSION.value:
        if not memory.session_id or not case.session_id or memory.session_id != case.session_id:
            return False
    if memory.scope == MemoryScope.KNOWLEDGE.value:
        if memory.workspace_id is not None and memory.workspace_id != case.workspace_id:
            return False
        if memory.project_id is not None and memory.project_id != case.project_id:
            return False
        return True
    if memory.workspace_id != case.workspace_id:
        return False
    if memory.project_id != case.project_id:
        return False
    return True


def _render_memories(
    memories: Iterable[MemoryFixture], max_chars: int
) -> tuple[tuple[str, ...], str]:
    ids: list[str] = []
    lines: list[str] = []
    used = 0
    for memory in memories:
        line = f"- [{memory.category}] {memory.title}: {' '.join(memory.content.split())}"
        separator = 1 if lines else 0
        remaining = max_chars - used - separator
        if remaining <= 0:
            break
        rendered = line[:remaining]
        lines.append(rendered)
        ids.append(memory.id)
        used += len(rendered) + separator
    return tuple(ids), "\n".join(lines)
