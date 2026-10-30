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

import ast
import hashlib
import inspect
import re
import shlex
import subprocess
import time
from collections.abc import Iterable
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


POLYGLOT_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}


def _strip_polyglot_comments(text: str) -> str:
    """Strip /* ... */ block comments and // ... line comments, preserving string literals."""

    def replacer(match: re.Match) -> str:
        s = match.group(0)
        if s.startswith("/"):
            return " "
        return s

    pattern = re.compile(
        r'//[^\r\n]*|/\*.*?\*/|\x27(?:\\.|[^\\\x27])*\x27|"(?:\\.|[^\\"])*"|`(?:\\.|[^\\`])*`',
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub(replacer, text)


def _mask_polyglot_comments_preserve_lines(text: str) -> str:
    """Mask /* ... */ block comments and // ... line comments with spaces, preserving newlines and char offsets."""

    def replacer(match: re.Match) -> str:
        s = match.group(0)
        if s.startswith("/"):
            return "".join("\n" if c == "\n" else ("\r" if c == "\r" else " ") for c in s)
        return s

    pattern = re.compile(
        r'//[^\r\n]*|/\*.*?\*/|\x27(?:\\.|[^\\\x27])*\x27|"(?:\\.|[^\\"])*"|`(?:\\.|[^\\`])*`',
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub(replacer, text)


def _extract_brace_balanced_span(text: str, start_pos: int) -> tuple[int, int] | None:
    """From start_pos, return (start_pos, end_pos) matching the declaration and balanced block or terminating ';'."""
    first_brace = text.find("{", start_pos)
    first_semi = text.find(";", start_pos)

    if first_semi != -1 and (first_brace == -1 or first_semi < first_brace):
        return (start_pos, first_semi + 1)

    if first_brace == -1:
        eol = text.find("\n", start_pos)
        end_pos = eol if eol != -1 else len(text)
        return (start_pos, end_pos)

    depth = 0
    in_string = None
    i = first_brace
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_string:
                in_string = None
        else:
            if ch in ("'", '"', "`"):
                in_string = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return (start_pos, i + 1)
        i += 1
    return (start_pos, len(text))


def _extract_brace_balanced_block(text: str, start_pos: int) -> str | None:
    """From start_pos, extract the declaration up to its balanced '{...}' block or terminating ';'."""
    span = _extract_brace_balanced_span(text, start_pos)
    if span is None:
        return None
    return text[span[0] : span[1]].strip()


def _extract_polyglot_symbol_normalized(source_text: str, symbol_path: str) -> str | None:
    """Extract and normalize a symbol declaration/body for TS/JS/Go/Rust."""
    clean_text = _strip_polyglot_comments(source_text)
    block: str | None = None

    if "." in symbol_path:
        parent_name, method_name = symbol_path.split(".", 1)
        parent_name = parent_name.strip()
        method_name = method_name.strip()

        # Direct search for Go receiver methods: func (.*ClassName) methodName(...)
        go_receiver_pattern = re.compile(
            rf"\bfunc\s*\([^)]*\b{re.escape(parent_name)}\b[^)]*\)\s*{re.escape(method_name)}\b"
        )
        match_go = go_receiver_pattern.search(clean_text)
        if match_go:
            block = _extract_brace_balanced_block(clean_text, match_go.start())
        else:
            # Locate enclosing class, interface, struct, impl block
            parent_pattern = re.compile(
                rf"\b(?:class|interface|struct|impl(?:\s+[^{{]+?\s+for)?|type)\s+{re.escape(parent_name)}\b"
                rf"|\btype\s+{re.escape(parent_name)}\s+struct\b"
            )
            match_parent = parent_pattern.search(clean_text)
            if match_parent:
                parent_block = _extract_brace_balanced_block(clean_text, match_parent.start())
                if parent_block:
                    method_pattern = re.compile(
                        rf"(?:(?:async|public|private|protected|static|fn|pub\s+fn)\s+)*\b{re.escape(method_name)}\b\s*(?:<[^>]*>)?\s*\("
                        rf"|\b{re.escape(method_name)}\s*[:=]\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z0-9_]+)\s*=>"
                        rf"|\b{re.escape(method_name)}\s*\([^)]*\)\s*(?:\{{|:[^{{]+|\b)"
                    )
                    match_method = method_pattern.search(parent_block)
                    if match_method:
                        block = _extract_brace_balanced_block(parent_block, match_method.start())
    else:
        symbol_name = symbol_path.strip()
        decl_pattern = re.compile(
            rf"(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const|let|var|enum|"
            rf"fn|pub\s+fn|struct|pub\s+struct|impl|trait|func)\s+(?:\([^)]*\)\s*)?{re.escape(symbol_name)}\b"
            rf"|\b{re.escape(symbol_name)}\s*(?:<[^>]*>)?\s*\("
            rf"|\b{re.escape(symbol_name)}\s*[:=]"
        )
        match_decl = decl_pattern.search(clean_text)
        if match_decl:
            block = _extract_brace_balanced_block(clean_text, match_decl.start())

    if block is not None:
        normalized = " ".join(block.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return None


def compute_symbol_hashes(
    specs: list[str] | tuple[str, ...],
    cwd: str | Path | None = None,
) -> dict[str, str]:
    """Compute normalized AST structural hashes for Python symbols in specs.

    Each spec has the format ``relative/path.py::SymbolName`` or
    ``relative/path.py::ClassName.method_name``.
    """
    base = Path(cwd or ".").resolve()
    hashes: dict[str, str] = {}
    ast_cache: dict[Path, ast.AST | None] = {}
    polyglot_cache: dict[Path, str | None] = {}

    for spec in specs:
        if not spec or not isinstance(spec, str) or "::" not in spec:
            continue
        rel_path, symbol_path = spec.split("::", 1)
        rel_path = rel_path.strip()
        symbol_path = symbol_path.strip()
        if not rel_path or not symbol_path:
            continue

        target_path = (base / rel_path).resolve()
        if not target_path.is_file():
            continue

        suffix = target_path.suffix.lower()
        if suffix in POLYGLOT_EXTENSIONS:
            if target_path not in polyglot_cache:
                try:
                    polyglot_cache[target_path] = target_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    polyglot_cache[target_path] = None
            source_text = polyglot_cache[target_path]
            if source_text is not None:
                sym_hash = _extract_polyglot_symbol_normalized(source_text, symbol_path)
                if sym_hash is not None:
                    hashes[spec] = sym_hash
            continue

        if suffix != ".py":
            continue

        if target_path not in ast_cache:
            try:
                source_text = target_path.read_text(encoding="utf-8")
                ast_cache[target_path] = ast.parse(source_text, filename=str(target_path))
            except (SyntaxError, OSError, UnicodeDecodeError):
                ast_cache[target_path] = None

        tree = ast_cache[target_path]
        if tree is None:
            continue

        target_node: ast.AST | None = None
        if "." in symbol_path:
            class_name, method_name = symbol_path.split(".", 1)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for sub in node.body:
                        if (
                            isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                            and sub.name == method_name
                        ):
                            target_node = sub
                            break
                    break
        else:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol_path:
                    target_node = node
                    break
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == symbol_path:
                            target_node = node
                            break
                    if target_node is not None:
                        break

            if target_node is None:
                for node in ast.walk(tree):
                    if (
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and getattr(node, "name", None) == symbol_path
                    ):
                        target_node = node
                        break

        if target_node is not None:
            dumped = ast.dump(target_node, include_attributes=False)
            hashes[spec] = hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    return hashes


def auto_ground_metadata(
    metadata: dict[str, Any] | None,
    cwd: str | Path | None = None,
) -> dict[str, Any] | None:
    """Auto-populate ``file_hashes``, ``symbol_hashes``, and ``git_commit`` in metadata from workspace files."""
    if metadata is None:
        return None
    meta = dict(metadata)
    if cwd is None:
        return meta

    # 1. File Hashes Auto-Capture
    candidate_files: list[str] = []
    seen_files: set[str] = set()
    for key in ("files", "changed_files"):
        val = meta.get(key)
        if isinstance(val, (list, tuple, set)):
            for item in val:
                if isinstance(item, str) and item.strip():
                    cleaned = item.strip()
                    if cleaned not in seen_files:
                        seen_files.add(cleaned)
                        candidate_files.append(cleaned)
        elif isinstance(val, str) and val.strip():
            cleaned = val.strip()
            if cleaned not in seen_files:
                seen_files.add(cleaned)
                candidate_files.append(cleaned)

    if candidate_files and "file_hashes" not in meta:
        computed_files = compute_file_hashes(candidate_files, cwd=cwd)
        if computed_files:
            meta["file_hashes"] = computed_files

    # 2. Symbol Hashes Auto-Capture
    candidate_symbols: list[str] = []
    sym_val = meta.get("symbols")
    if isinstance(sym_val, (list, tuple, set)):
        for item in sym_val:
            if isinstance(item, str) and item.strip():
                candidate_symbols.append(item.strip())
    elif isinstance(sym_val, str) and sym_val.strip():
        candidate_symbols.append(sym_val.strip())

    code_extensions = (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs")
    code_files = [f for f in candidate_files if any(f.endswith(ext) for ext in code_extensions)]
    normalized_specs: list[str] = []
    seen_specs: set[str] = set()

    for sym in candidate_symbols:
        sym_str = sym.strip()
        if "::" in sym_str:
            if sym_str not in seen_specs:
                seen_specs.add(sym_str)
                normalized_specs.append(sym_str)
        elif code_files:
            for code_file in code_files:
                spec = f"{code_file}::{sym_str}"
                if spec not in seen_specs:
                    seen_specs.add(spec)
                    normalized_specs.append(spec)

    if normalized_specs and "symbol_hashes" not in meta:
        computed_syms = compute_symbol_hashes(normalized_specs, cwd=cwd)
        if computed_syms:
            meta["symbol_hashes"] = computed_syms

    # 3. Temporal Git Commit Auto-Capture
    if "git_commit" not in meta:
        from fibmind.working_context import resolve_git_commit

        commit = resolve_git_commit(cwd)
        if commit is not None:
            meta["git_commit"] = commit

    return meta


def diagnose_temporal_drift(
    node: MemoryNode,
    cwd: str | Path | None = None,
) -> dict[str, Any] | None:
    """Diagnose cross-commit temporal drift between node's anchored git_commit and workspace HEAD."""
    if cwd is None:
        return None
    meta = node.metadata or {}
    anchored_commit = meta.get("git_commit")
    if not anchored_commit:
        return None

    from fibmind.working_context import resolve_git_commit

    current_commit = resolve_git_commit(cwd)
    if current_commit and str(anchored_commit).strip().lower() != str(current_commit).strip().lower():
        return {
            "anchored_commit": str(anchored_commit).strip().lower(),
            "current_commit": str(current_commit).strip().lower(),
            "commit_changed": True,
        }
    return None


def locate_symbol_spans(
    symbols: Iterable[str],
    cwd: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Locate 1-based start/end line spans and signatures for symbols in workspace files."""
    base = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    spans: dict[str, dict[str, Any]] = {}
    ast_cache: dict[Path, tuple[str, ast.AST] | None] = {}
    polyglot_cache: dict[Path, str | None] = {}

    for spec in symbols:
        if not spec or not isinstance(spec, str) or "::" not in spec:
            continue
        rel_path, symbol_path = spec.split("::", 1)
        rel_path = rel_path.strip()
        symbol_path = symbol_path.strip()
        if not rel_path or not symbol_path:
            continue

        target_path = (base / rel_path).resolve()
        if not target_path.is_file():
            continue

        # Get structural SHA-256
        sym_hashes = compute_symbol_hashes([spec], cwd=base)
        sha_hex = sym_hashes.get(spec)
        if not sha_hex:
            continue

        suffix = target_path.suffix.lower()

        # 1. Polyglot symbols
        if suffix in POLYGLOT_EXTENSIONS:
            if target_path not in polyglot_cache:
                try:
                    polyglot_cache[target_path] = target_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    polyglot_cache[target_path] = None
            source = polyglot_cache[target_path]
            if source is None:
                continue

            masked = _mask_polyglot_comments_preserve_lines(source)
            span_res: tuple[int, int] | None = None

            if "." in symbol_path:
                parent_name, method_name = symbol_path.split(".", 1)
                parent_name = parent_name.strip()
                method_name = method_name.strip()

                go_receiver_pattern = re.compile(
                    rf"\bfunc\s*\([^)]*\b{re.escape(parent_name)}\b[^)]*\)\s*{re.escape(method_name)}\b"
                )
                m_go = go_receiver_pattern.search(masked)
                if m_go:
                    span_res = _extract_brace_balanced_span(masked, m_go.start())
                else:
                    parent_pattern = re.compile(
                        rf"\b(?:class|interface|struct|impl(?:\s+[^{{]+?\s+for)?|type)\s+{re.escape(parent_name)}\b"
                        rf"|\btype\s+{re.escape(parent_name)}\s+struct\b"
                    )
                    m_parent = parent_pattern.search(masked)
                    if m_parent:
                        parent_span = _extract_brace_balanced_span(masked, m_parent.start())
                        if parent_span:
                            method_pattern = re.compile(
                                rf"(?:(?:async|public|private|protected|static|fn|pub\s+fn)\s+)*\b{re.escape(method_name)}\b\s*(?:<[^>]*>)?\s*\("
                                rf"|\b{re.escape(method_name)}\s*[:=]\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z0-9_]+)\s*=>"
                                rf"|\b{re.escape(method_name)}\s*\([^)]*\)\s*(?:\{{|:[^{{]+|\b)"
                            )
                            m_method = method_pattern.search(masked, parent_span[0], parent_span[1])
                            if m_method:
                                span_res = _extract_brace_balanced_span(masked, m_method.start())
            else:
                symbol_name = symbol_path.strip()
                decl_pattern = re.compile(
                    rf"(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const|let|var|enum|"
                    rf"fn|pub\s+fn|struct|pub\s+struct|impl|trait|func)\s+(?:\([^)]*\)\s*)?{re.escape(symbol_name)}\b"
                    rf"|\b{re.escape(symbol_name)}\s*(?:<[^>]*>)?\s*\("
                    rf"|\b{re.escape(symbol_name)}\s*[:=]"
                )
                m_decl = decl_pattern.search(masked)
                if m_decl:
                    span_res = _extract_brace_balanced_span(masked, m_decl.start())

            if span_res is not None:
                start_idx, end_idx = span_res
                start_line = source.count("\n", 0, start_idx) + 1
                end_line = source.count("\n", 0, max(start_idx, end_idx - 1)) + 1
                raw_decl = source[start_idx:end_idx]
                first_part = raw_decl.split("{", 1)[0].split(";", 1)[0]
                signature = " ".join(first_part.split())[:160].strip()
                spans[spec] = {
                    "file": rel_path,
                    "symbol": symbol_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "signature": signature,
                    "sha256": sha_hex,
                }
            continue

        if suffix != ".py":
            continue

        # 2. Python symbols
        if target_path not in ast_cache:
            try:
                src_text = target_path.read_text(encoding="utf-8")
                tree = ast.parse(src_text, filename=str(target_path))
                ast_cache[target_path] = (src_text, tree)
            except (SyntaxError, OSError, UnicodeDecodeError):
                ast_cache[target_path] = None

        cached = ast_cache[target_path]
        if cached is None:
            continue
        source, tree = cached

        target_node: ast.AST | None = None
        if "." in symbol_path:
            class_name, method_name = symbol_path.split(".", 1)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for sub in node.body:
                        if (
                            isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                            and sub.name == method_name
                        ):
                            target_node = sub
                            break
                    break
        else:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol_path:
                    target_node = node
                    break
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == symbol_path:
                            target_node = node
                            break
                    if target_node is not None:
                        break

            if target_node is None:
                for node in ast.walk(tree):
                    if (
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and getattr(node, "name", None) == symbol_path
                    ):
                        target_node = node
                        break

        if target_node is not None:
            start_line = int(getattr(target_node, "lineno", 1))
            end_line = int(getattr(target_node, "end_lineno", start_line))
            src_lines = source.splitlines()
            sig_lines = []
            for line in src_lines[start_line - 1 : end_line]:
                sig_lines.append(line.strip())
                if ":" in line:
                    break
            signature = " ".join(sig_lines)[:160].strip()
            spans[spec] = {
                "file": rel_path,
                "symbol": symbol_path,
                "start_line": start_line,
                "end_line": end_line,
                "signature": signature,
                "sha256": sha_hex,
            }

    return spans


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

    # 3. Symbol-level AST hash check
    symbol_verified_files: set[str] = set()
    symbol_hashes = meta.get("symbol_hashes")
    if isinstance(symbol_hashes, dict) and symbol_hashes:
        for spec, expected_sym_hash in symbol_hashes.items():
            if "::" not in spec:
                continue
            rel_path, _sym_name = spec.split("::", 1)
            rel_path = rel_path.strip()
            target_file = (base / rel_path).resolve()
            if not target_file.is_file():
                return False, f"missing_hashed_file: {rel_path}"
            actual_map = compute_symbol_hashes([spec], cwd=base)
            if spec not in actual_map:
                return False, f"missing_symbol: {spec}"
            if actual_map[spec] != str(expected_sym_hash):
                return False, f"symbol_hash_mismatch: {spec}"
            symbol_verified_files.add(str(rel_path))

    # 4. File SHA-256 drift check
    file_hashes = meta.get("file_hashes")
    if isinstance(file_hashes, dict) and file_hashes:
        for rel_path, expected_hash in file_hashes.items():
            target = (base / str(rel_path)).resolve()
            if not target.is_file():
                return False, f"missing_hashed_file: {rel_path}"
            if str(rel_path) in symbol_verified_files:
                # Target symbol(s) already verified intact; skip whole-file drift
                continue
            actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual_hash != str(expected_hash):
                return False, f"file_hash_mismatch: {rel_path}"

    # 5. Target file existence check
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


def _run_step_command(
    step_cmd: str,
    *,
    cwd: str | Path | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Run one shell command safely with timeout and environment failure handling."""
    cmd_args = shlex.split(step_cmd)
    if not cmd_args:
        return {
            "command": step_cmd,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration_sec": 0.0,
            "verdict": Verdict.INCONCLUSIVE.value,
            "reason": "empty command",
        }
    start_t = time.monotonic()
    try:
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
        return {
            "command": step_cmd,
            "exit_code": proc_run.returncode,
            "stdout": proc_run.stdout,
            "stderr": proc_run.stderr,
            "duration_sec": duration,
            "verdict": verdict,
            "reason": diag_reason,
        }
    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - start_t, 2)
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "command": step_cmd,
            "exit_code": 124,
            "stdout": stdout,
            "stderr": stderr,
            "duration_sec": duration,
            "verdict": Verdict.INCONCLUSIVE.value,
            "reason": f"command timed out after {timeout}s",
        }
    except (FileNotFoundError, PermissionError, OSError) as exc:
        duration = round(time.monotonic() - start_t, 2)
        return {
            "command": step_cmd,
            "exit_code": 127 if isinstance(exc, FileNotFoundError) else 126,
            "stdout": "",
            "stderr": str(exc),
            "duration_sec": duration,
            "verdict": Verdict.INCONCLUSIVE.value,
            "reason": f"command failed to start ({type(exc).__name__}: {exc})",
        }


def _looks_like_executable_command(verify_spec: str, cwd: str | Path | None = None) -> bool:
    """Return True if ``verify_spec`` looks like a runnable shell command rather than prose."""
    import shutil

    spec = verify_spec.strip()
    if not spec:
        return False
    try:
        tokens = shlex.split(spec)
    except ValueError:
        return False
    if not tokens:
        return False
    first = tokens[0]
    base = Path(cwd or ".").resolve()
    if "/" in first or "\\" in first or first.startswith("."):
        return (base / first).exists() or Path(first).exists()
    return shutil.which(first) is not None


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
    min_relevance: float = 0.25,
    cwd: str | Path | None = None,
    model_id: str | None = None,
    require_files_exist: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Attempt a System 1 fast-path response or procedure execution before calling an LLM."""
    from fibmind.brain import AdviseVerdict, _identity_kwargs
    from fibmind.brain import BrainState as _BrainState
    from fibmind.ranking import tokenize

    goal = _require_text(goal, "goal")
    if min_maturity not in _MATURITY_ORDER:
        raise ValueError(f"min_maturity must be one of {sorted(_MATURITY_ORDER)}")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    if not 0.0 <= min_relevance <= 1.0:
        raise ValueError("min_relevance must be between 0.0 and 1.0")

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
            if not step_cmd or not step_cmd.strip():
                continue
            step_info = _run_step_command(step_cmd, cwd=cwd, timeout=timeout)
            step_results.append(step_info)
            if step_info["exit_code"] != 0:
                failed_step = step_info
                break

        # If all steps succeeded and procedure declares an executable verify command, run it too
        verify_spec = _render_command_template(str(proc.get("verify") or "").strip(), inputs)
        if failed_step is None and verify_spec and _looks_like_executable_command(verify_spec, cwd=cwd):
            v_info = _run_step_command(verify_spec, cwd=cwd, timeout=timeout)
            v_info["is_verify"] = True
            step_results.append(v_info)
            if v_info["exit_code"] != 0 or v_info["verdict"] != Verdict.CONFIRMED.value:
                failed_step = v_info

        if not step_results:
            warnings.append({"node_id": node.id, "title": node.title, "reason": "empty_procedure_execution"})
            continue

        if failed_step is not None:
            # Trip circuit breaker when physical execution fails (refuted or inconclusive)
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

        # Distinguish inconclusive (e.g. collected 0 items) from confirmed success
        any_inconclusive = any(s["verdict"] != Verdict.CONFIRMED.value for s in step_results)
        final_verdict = Verdict.INCONCLUSIVE.value if any_inconclusive else Verdict.CONFIRMED.value
        reflection = brain.reflect(
            final_verdict,
            source="fibbrain_reflex",
            node_id=node.id,
            note=(
                f"Reflex executed {len(step_results)} step(s) with inconclusive verification"
                if any_inconclusive
                else f"Reflex executed {len(step_results)} step(s) successfully"
            ),
            state=effective_state,
        )
        if any_inconclusive:
            fallback = brain.recall(goal, state=effective_state)
            return ReflexDecision(
                bypass_llm=False,
                mode="inconclusive_execution",
                goal=goal,
                node_id=node.id,
                confidence=reflection["confidence"],
                procedure=proc,
                execution={"steps": step_results, "reflection": reflection},
                precondition_warnings=warnings,
                fallback_context=fallback,
                state=identity,
            ).to_dict()

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
    assessment = brain.assess(
        goal,
        state=effective_state,
        workspace_root=cwd,
        observe_event=False,
    )
    contradicted = assessment.get("contradicted", [])
    if contradicted:
        for c in contradicted:
            warnings.append(
                {
                    "node_id": c.get("node_id"),
                    "title": c.get("title", ""),
                    "reason": f"metacognition veto: contradicted memory ({c.get('reason', 'refuted')})",
                }
            )
        fallback = brain.recall(goal, state=effective_state)
        return ReflexDecision(
            bypass_llm=False,
            mode="llm_fallback",
            goal=goal,
            precondition_warnings=warnings,
            fallback_context=fallback,
            state=identity,
        ).to_dict()

    weak_ids = {w["node_id"] for w in assessment.get("weak", []) if "node_id" in w}
    conflicted_uncertain_ids = {
        u["node_id"] for u in assessment.get("uncertain", []) if "node_id" in u and "conflicted" in u.get("reasons", [])
    }

    search_kwargs = _identity_kwargs(identity)
    found = brain.memory.search(goal, top_k=5, **search_kwargs)
    summarizer = StructuredSummarizer()
    excluded_cats = {EPISODE_CATEGORY, HOT_CATEGORY, GOAL_CATEGORY, PROCEDURE_CATEGORY}
    query_terms = set(tokenize(goal))

    for hit in found["results"]:
        if hit["category"] in excluded_cats or hit.get("memory_kind") == MemoryKind.PROCEDURE.value:
            continue
        hit_id = hit["node_id"]
        if hit_id in weak_ids:
            continue
        if hit_id in conflicted_uncertain_ids:
            warnings.append(
                {
                    "node_id": hit_id,
                    "title": hit.get("title", ""),
                    "reason": "metacognition veto: candidate in uncertain conflicted set",
                }
            )
            fallback = brain.recall(goal, state=effective_state)
            return ReflexDecision(
                bypass_llm=False,
                mode="llm_fallback",
                goal=goal,
                precondition_warnings=warnings,
                fallback_context=fallback,
                state=identity,
            ).to_dict()

        matched = set(hit.get("matched_terms") or ())
        term_coverage = (len(matched) / len(query_terms)) if query_terms else 0.0
        if term_coverage < min_relevance and hit.get("score", 0.0) < min_relevance * 2.0:
            continue
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

        try:
            sig = inspect.signature(summarizer.distill_node)
            if "goal" in sig.parameters:
                distilled = summarizer.distill_node(node, goal=goal)
            else:
                distilled = summarizer.distill_node(node)
        except (TypeError, ValueError):
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
                "assessment_verdict": assessment.get("verdict", "unknown"),
                "known_coverage": assessment.get("coverage", {}).get("known", 0.0),
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
