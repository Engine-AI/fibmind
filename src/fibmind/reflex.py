"""System 1 Fast Reflex Engine: zero-LLM direct knowledge answers and verified procedure execution.

When a query or task matches a high-confidence verified memory or an established/verified
procedure whose environmental preconditions still hold, ``run_reflex`` returns a
deterministic response or executes the procedure directly (`bypass_llm=True`).

If environmental preconditions fail (e.g., target file SHA-256 hash drifted, target
file was deleted, memory is in dialectical conflict, or model-specific heuristic
belongs to a different model generation) or if execution fails physical verification,
the circuit breaker trips (`reflect(verdict="refuted")`) and seamlessly falls back
to System 2 (`bypass_llm=False`) with a full ``recall`` context pack for the LLM.
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fibmind.distill import StructuredSummarizer
from fibmind.models import MemoryKind, MemoryNode, Verdict
from fibmind.planning import GOAL_CATEGORY
from fibmind.procedure import PROCEDURE_CATEGORY
from fibmind.review import EPISODE_CATEGORY
from fibmind.service import _require_text
from fibmind.verify import diagnose_verification_outcome

if TYPE_CHECKING:
    from fibmind.brain import BrainState, FibBrain

HOT_CATEGORY = "hot"
_MATURITY_ORDER = {"candidate": 0, "verified": 1, "established": 2}


def compute_file_hashes(
    files: list[str] | tuple[str, ...],
    cwd: str | Path | None = None,
) -> dict[str, str]:
    """Compute SHA-256 hex digests for existing files relative to ``cwd``."""
    base = Path(cwd or ".").resolve()
    hashes: dict[str, str] = {}
    for rel in files:
        if not rel or not str(rel).strip():
            continue
        path = (base / str(rel)).resolve()
        if path.is_file():
            hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def check_node_preconditions(
    node: MemoryNode,
    *,
    cwd: str | Path | None = None,
    model_id: str | None = None,
    require_files_exist: bool = False,
) -> tuple[bool, str]:
    """Verify environmental, conflict, and model-generation preconditions before System 1 reuse.

    Returns ``(ok, reason)``.
    """
    meta = node.metadata or {}

    # 1. Dialectical conflict check
    if meta.get("conflicted"):
        reason = meta.get("conflict_reason") or "unresolved contradiction"
        return False, f"conflicted_memory: {reason}"

    # 2. Model generation / heuristic isolation check
    if meta.get("grounding") == "model_heuristic":
        recorded_model = meta.get("model_id")
        if model_id and recorded_model and str(recorded_model) != str(model_id):
            return (
                False,
                f"model_heuristic_mismatch: recorded for {recorded_model}, current model is {model_id}",
            )

    base = Path(cwd or ".").resolve()

    # 3. File SHA-256 drift check
    file_hashes = meta.get("file_hashes")
    if isinstance(file_hashes, dict) and file_hashes:
        for rel_path, expected_hash in file_hashes.items():
            target = (base / str(rel_path)).resolve()
            if not target.is_file():
                return False, f"missing_hashed_file: {rel_path}"
            actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual_hash != str(expected_hash):
                return False, f"file_hash_mismatch: {rel_path}"

    # 4. Target file existence check
    if require_files_exist or bool(meta.get("require_files_exist")):
        files = meta.get("files") or meta.get("changed_files")
        if isinstance(files, (list, tuple)):
            for rel_path in files:
                if not rel_path:
                    continue
                target = (base / str(rel_path)).resolve()
                if not target.exists():
                    return False, f"missing_target_file: {rel_path}"

    return True, "preconditions_satisfied"


@dataclass(frozen=True, slots=True)
class ReflexDecision:
    """Outcome of a System 1 Fast Reflex evaluation."""

    bypass_llm: bool
    mode: str
    goal: str
    node_id: str | None = None
    confidence: float = 0.0
    answer: str | None = None
    distillation: dict[str, Any] | None = None
    procedure: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    precondition_warnings: list[dict[str, Any]] = field(default_factory=list)
    fallback_context: dict[str, Any] | None = None
    state: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bypass_llm": self.bypass_llm,
            "mode": self.mode,
            "goal": self.goal,
            "node_id": self.node_id,
            "confidence": round(self.confidence, 4),
            "answer": self.answer,
            "distillation": self.distillation,
            "procedure": self.procedure,
            "execution": self.execution,
            "precondition_warnings": list(self.precondition_warnings),
            "fallback_context": self.fallback_context,
            "state": dict(self.state),
        }


def _render_command_template(template: str, inputs: dict[str, str] | None) -> str:
    """Substitute ``{key}`` placeholders in ``template`` using ``inputs``."""
    if not inputs:
        return template
    rendered = template
    for key, val in inputs.items():
        placeholder = "{" + str(key) + "}"
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, shlex.quote(str(val)))
    return rendered


def run_reflex(
    brain: FibBrain,
    goal: str,
    *,
    state: BrainState | None = None,
    execute: bool = False,
    inputs: dict[str, str] | None = None,
    min_confidence: float = 0.75,
    min_maturity: str = "verified",
    cwd: str | Path | None = None,
    model_id: str | None = None,
    require_files_exist: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Attempt a System 1 fast-path response or procedure execution before calling an LLM."""
    from fibmind.brain import AdviseVerdict, _identity_kwargs
    from fibmind.brain import BrainState as _BrainState

    goal = _require_text(goal, "goal")
    if min_maturity not in _MATURITY_ORDER:
        raise ValueError(f"min_maturity must be one of {sorted(_MATURITY_ORDER)}")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0.0 and 1.0")

    effective_state = state or _BrainState()
    identity = effective_state.identity()
    effective_model_id = model_id or getattr(effective_state, "model_id", None)
    warnings: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fast Path B: Distilled Procedure Reflex
    procedures = brain._matching_procedures(goal, identity)  # noqa: SLF001
    for proc in procedures:
        proc_id = proc["node_id"]
        node = brain.memory.read(lambda m, nid=proc_id: m.nodes.get(nid))
        if node is None:
            continue
        if _MATURITY_ORDER.get(proc["maturity"], 0) < _MATURITY_ORDER[min_maturity]:
            continue
        ok, reason = check_node_preconditions(
            node,
            cwd=cwd,
            model_id=effective_model_id,
            require_files_exist=require_files_exist,
        )
        if not ok:
            warnings.append({"node_id": node.id, "title": node.title, "reason": reason})
            continue

        # Check advise gate on procedure tools/steps
        blocked_by_advise = False
        for action_name in proc.get("tools") or proc.get("steps") or []:
            decision = brain.advise(str(action_name), state=effective_state)
            if decision["verdict"] == AdviseVerdict.REJECT.value:
                warnings.append(
                    {
                        "node_id": node.id,
                        "title": node.title,
                        "reason": f"advise_rejected: {decision['reason']}",
                    }
                )
                blocked_by_advise = True
                break
        if blocked_by_advise:
            continue

        rendered_steps = [_render_command_template(step, inputs) for step in proc["steps"]]
        if not execute:
            brain.observe(
                "reflex",
                f"procedure_ready {node.title} ({node.id})",
                {"node_id": node.id, "mode": "procedure_ready"},
                state=effective_state,
            )
            return ReflexDecision(
                bypass_llm=True,
                mode="procedure_ready",
                goal=goal,
                node_id=node.id,
                confidence=node.confidence,
                answer="\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(rendered_steps)),
                procedure={**proc, "rendered_steps": rendered_steps},
                precondition_warnings=warnings,
                state=identity,
            ).to_dict()

        # Execute procedure steps sequentially with post-verification and circuit breaker
        step_results: list[dict[str, Any]] = []
        failed_step: dict[str, Any] | None = None
        for step_cmd in rendered_steps:
            cmd_args = shlex.split(step_cmd)
            if not cmd_args:
                continue
            start_t = time.monotonic()
            proc_run = subprocess.run(
                cmd_args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = round(time.monotonic() - start_t, 2)
            verdict, diag_reason = diagnose_verification_outcome(
                proc_run.returncode,
                proc_run.stdout,
                proc_run.stderr,
            )
            step_info = {
                "command": step_cmd,
                "exit_code": proc_run.returncode,
                "stdout": proc_run.stdout,
                "stderr": proc_run.stderr,
                "duration_sec": duration,
                "verdict": verdict,
                "reason": diag_reason,
            }
            step_results.append(step_info)
            if proc_run.returncode != 0:
                failed_step = step_info
                break

        if failed_step is not None:
            # Trip circuit breaker when physical execution fails (refuted)
            reflect_verdict = (
                Verdict.REFUTED.value if failed_step["verdict"] == Verdict.REFUTED.value else Verdict.INCONCLUSIVE.value
            )
            reflection = brain.reflect(
                reflect_verdict,
                source="fibbrain_reflex",
                node_id=node.id,
                note=f"Reflex step failed ({failed_step['command']}): {failed_step['reason']}",
                state=effective_state,
            )
            fallback = brain.recall(goal, state=effective_state)
            return ReflexDecision(
                bypass_llm=False,
                mode="circuit_breaker_tripped",
                goal=goal,
                node_id=node.id,
                confidence=reflection["confidence"],
                procedure=proc,
                execution={"steps": step_results, "reflection": reflection},
                precondition_warnings=warnings,
                fallback_context=fallback,
                state=identity,
            ).to_dict()

        reflection = brain.reflect(
            Verdict.CONFIRMED.value,
            source="fibbrain_reflex",
            node_id=node.id,
            note=f"Reflex executed {len(step_results)} step(s) successfully",
            state=effective_state,
        )
        combined_out = "\n".join(s["stdout"].strip() for s in step_results if s["stdout"].strip())
        return ReflexDecision(
            bypass_llm=True,
            mode="procedure_executed",
            goal=goal,
            node_id=node.id,
            confidence=reflection["confidence"],
            answer=combined_out or f"Executed procedure: {node.title}",
            procedure=proc,
            execution={"steps": step_results, "reflection": reflection},
            precondition_warnings=warnings,
            state=identity,
        ).to_dict()

    # ------------------------------------------------------------------
    # Fast Path A: Direct Declarative Knowledge Answer
    search_kwargs = _identity_kwargs(identity)
    found = brain.memory.search(goal, top_k=5, **search_kwargs)
    summarizer = StructuredSummarizer()
    excluded_cats = {EPISODE_CATEGORY, HOT_CATEGORY, GOAL_CATEGORY, PROCEDURE_CATEGORY}

    for hit in found["results"]:
        if hit["category"] in excluded_cats or hit.get("memory_kind") == MemoryKind.PROCEDURE.value:
            continue
        hit_id = hit["node_id"]
        node = brain.memory.read(lambda m, nid=hit_id: m.nodes.get(nid))
        if node is None or node.confidence < min_confidence:
            continue
        ok, reason = check_node_preconditions(
            node,
            cwd=cwd,
            model_id=effective_model_id,
            require_files_exist=require_files_exist,
        )
        if not ok:
            warnings.append({"node_id": node.id, "title": node.title, "reason": reason})
            continue

        distilled = summarizer.distill_node(node)
        brain.observe(
            "reflex",
            f"direct_answer {node.title} ({node.id})",
            {"node_id": node.id, "mode": "direct_answer", "confidence": node.confidence},
            state=effective_state,
        )
        return ReflexDecision(
            bypass_llm=True,
            mode="direct_answer",
            goal=goal,
            node_id=node.id,
            confidence=node.confidence,
            answer=distilled.to_text(),
            distillation={
                "title": distilled.title,
                "directive": distilled.directive,
                "conditions": list(distilled.conditions),
                "exceptions": list(distilled.exceptions),
                "verified_by": list(distilled.verified_by),
                "risk_level": distilled.risk_level,
            },
            precondition_warnings=warnings,
            state=identity,
        ).to_dict()

    # ------------------------------------------------------------------
    # Slow Path Fallback (System 2: LLM Required)
    fallback = brain.recall(goal, state=effective_state)
    return ReflexDecision(
        bypass_llm=False,
        mode="llm_fallback",
        goal=goal,
        precondition_warnings=warnings,
        fallback_context=fallback,
        state=identity,
    ).to_dict()
