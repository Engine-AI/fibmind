"""Entity world model projection for files, modules, and symbols.

Aggregates invariants, mature procedures, pitfalls (refuted / error memories),
conflicts, verification commands, and environmental drift into a single
structured entity profile.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fibmind.models import MemoryKind, MemoryNode, MemoryScope, MemoryStatus, RelationType
from fibmind.procedure import is_procedure_node
from fibmind.reflex import check_node_preconditions, diagnose_temporal_drift, locate_symbol_spans

if TYPE_CHECKING:
    from fibmind.graph import FibMind

CODE_DOC_EXTENSIONS = (
    ".py",
    ".js",
    ".ts",
    ".json",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".sh",
    ".rs",
    ".go",
)


DRIFT_REASON_PREFIXES = (
    "missing_hashed_file",
    "file_hash_mismatch",
    "missing_target_file",
    "missing_symbol",
    "symbol_hash_mismatch",
)


def _node_entry(node: MemoryNode) -> dict[str, Any]:
    return {
        "node_id": node.id,
        "title": node.title,
        "content": node.content,
        "category": node.category,
        "status": node.status.value,
        "confidence": round(node.confidence, 4),
        "metadata": dict(node.metadata) if node.metadata else {},
    }


def _pitfall_entry(node: MemoryNode) -> dict[str, Any]:
    meta = node.metadata or {}
    reason = node.status_reason or meta.get("conflict_reason") or meta.get("reason") or "refuted or known error"
    return {
        "node_id": node.id,
        "title": node.title,
        "content": node.content,
        "category": node.category,
        "status": node.status.value,
        "reason": str(reason),
        "confidence": round(node.confidence, 4),
    }


def _conflict_entry(node: MemoryNode) -> dict[str, Any]:
    meta = node.metadata or {}
    return {
        "node_id": node.id,
        "title": node.title,
        "category": node.category,
        "conflicted": bool(meta.get("conflicted")),
        "conflict_reason": meta.get("conflict_reason"),
        "potential_conflicts": list(meta.get("potential_conflicts") or ()),
    }


def extract_python_imports(file_path: str, cwd: str | Path) -> list[str]:
    """Extract local workspace-relative Python file paths imported by ``file_path``."""
    base = Path(cwd).resolve()
    target = (base / file_path.strip()).resolve()
    if not target.is_file() or target.suffix != ".py":
        return []

    try:
        source_text = target.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(target))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return []

    candidate_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    candidate_modules.add(alias.name.strip())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                candidate_modules.add(node.module.strip())
                for alias in node.names:
                    if alias.name:
                        candidate_modules.add(f"{node.module.strip()}.{alias.name.strip()}")
            elif node.level > 0:
                try:
                    rel_parent = target.parent.relative_to(base).as_posix()
                    for alias in node.names:
                        if alias.name:
                            name = alias.name.strip()
                            candidate_modules.add(
                                f"{rel_parent.replace('/', '.')}.{name}" if rel_parent and rel_parent != "." else name
                            )
                except Exception:
                    pass

    results: list[str] = []
    for mod in candidate_modules:
        if not mod:
            continue
        rel_mod = mod.replace(".", "/")
        probes = [
            f"{rel_mod}.py",
            f"src/{rel_mod}.py",
            f"{rel_mod}/__init__.py",
            f"src/{rel_mod}/__init__.py",
        ]
        if target.parent != base:
            try:
                rel_parent = target.parent.relative_to(base).as_posix()
                if rel_parent and rel_parent != ".":
                    probes.extend(
                        [
                            f"{rel_parent}/{rel_mod}.py",
                            f"{rel_parent}/{rel_mod}/__init__.py",
                        ]
                    )
            except Exception:
                pass

        for candidate_rel in probes:
            cand_path = (base / candidate_rel).resolve()
            if cand_path.is_file() and cand_path != target:
                results.append(candidate_rel)
                break

    return sorted(set(results))


def extract_code_imports(file_path: str, cwd: str | Path) -> list[str]:
    """Extract local workspace-relative file paths imported by ``file_path``.

    Supports Python (.py), TypeScript/JavaScript (.ts, .tsx, .js, .jsx),
    Rust (.rs), and Go (.go).
    """
    base = Path(cwd).resolve()
    clean_path = str(file_path).strip()
    target = (base / clean_path).resolve()
    if not target.is_file():
        return []

    ext = target.suffix.lower()
    if ext == ".py":
        return extract_python_imports(clean_path, cwd)

    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    results: list[str] = []

    # 1. TypeScript & JavaScript (.ts, .tsx, .js, .jsx, .mjs, .cjs)
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        raw_specs: set[str] = set()
        for m in re.finditer(r"""\bfrom\s+['"]([^'"]+)['"]""", content):
            raw_specs.add(m.group(1).strip())
        for m in re.finditer(r"""\bimport\s+['"]([^'"]+)['"]""", content):
            raw_specs.add(m.group(1).strip())
        for m in re.finditer(r"""\b(?:require|import)\s*\(\s*['"]([^'"]+)['"]\s*\)""", content):
            raw_specs.add(m.group(1).strip())

        ts_exts = (".ts", ".tsx", ".js", ".jsx")
        for spec in raw_specs:
            if not spec:
                continue
            candidates: list[Path] = []
            if spec.startswith("."):
                cand_base = (target.parent / spec).resolve()
                candidates.append(cand_base)
            else:
                clean_spec = spec.removeprefix("@/").removeprefix("~/")
                candidates.append((base / clean_spec).resolve())
                candidates.append((base / "src" / clean_spec).resolve())
                candidates.append((target.parent / clean_spec).resolve())

            for cand in candidates:
                probes: list[Path] = []
                if cand.suffix.lower() in ts_exts and cand.is_file():
                    probes.append(cand)
                elif cand.suffix.lower() == ".js":
                    probes.extend([cand.with_suffix(".ts"), cand.with_suffix(".tsx"), cand])
                else:
                    for te in ts_exts:
                        probes.append(cand.with_name(cand.name + te))
                        probes.append(cand / f"index{te}")

                found = False
                for p in probes:
                    if p.is_file() and p != target:
                        try:
                            if p.is_relative_to(base):
                                results.append(p.relative_to(base).as_posix())
                                found = True
                                break
                        except Exception:
                            pass
                if found:
                    break

    # 2. Rust (.rs)
    elif ext == ".rs":
        mod_names: set[str] = set()
        for m in re.finditer(r"""\b(?:pub\s+)?mod\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;""", content):
            mod_names.add(m.group(1).strip())
        for m in re.finditer(r"""\buse\s+crate::([a-zA-Z_][a-zA-Z0-9_]*)""", content):
            mod_names.add(m.group(1).strip())

        for name in mod_names:
            if not name:
                continue
            probes = [
                target.parent / f"{name}.rs",
                target.parent / name / "mod.rs",
                base / "src" / f"{name}.rs",
                base / "src" / name / "mod.rs",
                base / f"{name}.rs",
                base / name / "mod.rs",
            ]
            for p in probes:
                cand = p.resolve()
                if cand.is_file() and cand != target:
                    try:
                        if cand.is_relative_to(base):
                            results.append(cand.relative_to(base).as_posix())
                            break
                    except Exception:
                        pass

    # 3. Go (.go)
    elif ext == ".go":
        go_specs: set[str] = set()
        for m in re.finditer(r"""\bimport\s+(?:[a-zA-Z_][a-zA-Z0-9_]*\s+)?["']([^"']+)["']""", content):
            go_specs.add(m.group(1).strip())
        for block in re.finditer(r"""\bimport\s*\(([^)]*)\)""", content, re.DOTALL):
            for m in re.finditer(r"""["']([^"']+)["']""", block.group(1)):
                go_specs.add(m.group(1).strip())

        for spec in go_specs:
            if not spec:
                continue
            cand_dirs: list[Path] = []
            if spec.startswith("."):
                cand_dirs.append((target.parent / spec).resolve())
            else:
                cand_dirs.append((base / spec).resolve())
                parts = spec.split("/")
                for i in range(1, len(parts)):
                    subpath = "/".join(parts[i:])
                    cand_dirs.append((base / subpath).resolve())
                    cand_dirs.append((base / "src" / subpath).resolve())

            found = False
            for d in cand_dirs:
                if d.is_dir():
                    go_files = sorted(d.glob("*.go"))
                    for gf in go_files:
                        cand = gf.resolve()
                        if cand.is_file() and cand != target:
                            try:
                                if cand.is_relative_to(base):
                                    results.append(cand.relative_to(base).as_posix())
                                    found = True
                            except Exception:
                                pass
                    if found:
                        break

    return sorted(set(results))


def resolve_transitive_imports(
    file_path: str,
    cwd: str | Path,
    *,
    max_hops: int = 2,
) -> list[dict[str, Any]]:
    """Resolve direct and transitive imported files using BFS up to max_hops.

    Returns a list of dicts:
        {"file": <rel_path>, "hop": <int>, "via": <str | None>}
    """
    clean_start = str(file_path).strip()
    if not clean_start or max_hops < 1:
        return []

    visited: set[str] = {clean_start}
    results: list[dict[str, Any]] = []

    queue: list[tuple[str, int]] = []
    hop1_files = extract_code_imports(clean_start, cwd)
    for imp in hop1_files:
        if imp not in visited:
            visited.add(imp)
            entry = {"file": imp, "hop": 1, "via": None}
            results.append(entry)
            queue.append((imp, 1))

    while queue:
        curr_file, hop = queue.pop(0)
        if hop >= max_hops:
            continue
        next_hop = hop + 1
        sub_imports = extract_code_imports(curr_file, cwd)
        for imp in sub_imports:
            if imp not in visited:
                visited.add(imp)
                entry = {"file": imp, "hop": next_hop, "via": curr_file}
                results.append(entry)
                queue.append((imp, next_hop))

    return results


def build_entity_profile(
    memory: FibMind,
    target: str,
    *,
    identity: dict[str, str | None],
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate an entity world model view for a file path or symbol."""
    clean_target = str(target).strip()
    target_lower = clean_target.lower()
    target_base = Path(clean_target).name.lower()

    target_file = ""
    target_sym = ""
    raw_target_file = ""
    if "::" in clean_target:
        raw_file, raw_sym = clean_target.split("::", 1)
        raw_target_file = raw_file.strip()
        target_file = raw_file.strip().lower()
        target_sym = raw_sym.strip().lower()

    lookup_file = raw_target_file if raw_target_file else clean_target
    if cwd is not None and lookup_file:
        transitive_list = resolve_transitive_imports(lookup_file, cwd, max_hops=2)
        imported_files = [item["file"] for item in transitive_list if item["hop"] == 1]
    else:
        transitive_list = []
        imported_files = []

    anchor_node_ids = memory.lexical.nodes_for_anchor(clean_target)
    if target_file:
        anchor_node_ids |= memory.lexical.nodes_for_anchor(target_file)
    if target_sym:
        anchor_node_ids |= memory.lexical.nodes_for_anchor(target_sym)

    matched_nodes: list[MemoryNode] = []
    for node in memory.nodes.values():
        if node.folded_into is not None:
            continue
        if node.status not in (MemoryStatus.ACTIVE, MemoryStatus.REFUTED, MemoryStatus.PENDING):
            continue

        meta = node.metadata or {}

        # Check association with target
        is_anchor_match = node.id in anchor_node_ids
        title_lower = node.title.lower()
        is_title_match = target_lower in title_lower or (bool(target_base) and target_base in title_lower)
        if target_sym and target_sym in title_lower:
            is_title_match = True
        if target_file and Path(target_file).name.lower() in title_lower:
            is_title_match = True

        is_tag_match = any(target_lower == tag.lower() for tag in node.tags)
        if target_sym and any(target_sym == tag.lower() for tag in node.tags):
            is_tag_match = True

        # Check symbol_hashes keys in node metadata
        is_sym_match = False
        node_sym_hashes = meta.get("symbol_hashes")
        if isinstance(node_sym_hashes, dict) and node_sym_hashes:
            for spec in node_sym_hashes:
                spec_lower = spec.lower()
                if target_lower == spec_lower or (target_lower and target_lower in spec_lower):
                    is_sym_match = True
                    break
                if "::" in spec:
                    spec_file, spec_sym = spec.split("::", 1)
                    spec_file = spec_file.strip().lower()
                    spec_sym = spec_sym.strip().lower()
                    if target_lower == spec_sym:
                        is_sym_match = True
                        break
                    if (
                        target_file
                        and target_sym
                        and (
                            (target_file == spec_file or Path(target_file).name.lower() == Path(spec_file).name.lower())
                            and target_sym == spec_sym
                        )
                    ):
                        is_sym_match = True
                        break

        # Check file_hashes / files / changed_files in node metadata
        is_file_match = False
        node_file_hashes = meta.get("file_hashes")
        if isinstance(node_file_hashes, dict):
            for f in node_file_hashes:
                f_lower = str(f).lower()
                if target_lower == f_lower or (bool(target_base) and target_base == Path(f_lower).name.lower()):
                    is_file_match = True
                    break
        if not is_file_match:
            for f_key in ("files", "changed_files"):
                f_val = meta.get(f_key)
                if isinstance(f_val, (list, tuple, set)):
                    for f in f_val:
                        f_lower = str(f).lower()
                        if target_lower == f_lower or (bool(target_base) and target_base == Path(f_lower).name.lower()):
                            is_file_match = True
                            break
                elif isinstance(f_val, str) and f_val.strip():
                    f_lower = f_val.strip().lower()
                    if target_lower == f_lower or (bool(target_base) and target_base == Path(f_lower).name.lower()):
                        is_file_match = True
                        break

        if not (is_anchor_match or is_title_match or is_tag_match or is_sym_match or is_file_match):
            continue

        # Check visibility
        if not memory.is_visible(
            node,
            owner=identity.get("owner"),
            statuses={node.status},
            workspace_id=identity.get("workspace_id"),
            project_id=identity.get("project_id"),
            session_id=identity.get("session_id"),
            task_id=identity.get("task_id"),
        ):
            continue

        matched_nodes.append(node)

    invariants: list[dict[str, Any]] = []
    procedures: list[dict[str, Any]] = []
    pitfalls: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    verification_commands: list[str] = []
    seen_commands: set[str] = set()
    drift: list[dict[str, Any]] = []

    for node in matched_nodes:
        meta = node.metadata or {}

        # Collect verification commands
        cmd_sources = [
            meta.get("command"),
            meta.get("verification"),
        ]
        raw_cmds = meta.get("commands")
        if isinstance(raw_cmds, (list, tuple)):
            cmd_sources.extend(raw_cmds)
        raw_steps = meta.get("steps")
        if isinstance(raw_steps, (list, tuple)) and (node.category == "procedure" or is_procedure_node(node)):
            cmd_sources.extend(raw_steps)

        for cmd in cmd_sources:
            if isinstance(cmd, str) and cmd.strip():
                clean_cmd = cmd.strip()
                if clean_cmd not in seen_commands:
                    seen_commands.add(clean_cmd)
                    verification_commands.append(clean_cmd)

        # 1. Pitfalls: refuted memories, error, or risk memories
        if (
            node.status == MemoryStatus.REFUTED
            or node.category in ("error", "risk")
            or node.memory_kind == MemoryKind.ERROR
        ):
            pitfalls.append(_pitfall_entry(node))

        # 2. Conflicts
        if meta.get("conflicted") or bool(meta.get("potential_conflicts")):
            conflicts.append(_conflict_entry(node))

        # 3. Active procedures
        if node.status == MemoryStatus.ACTIVE:
            is_proc = (
                node.category == "procedure" or node.memory_kind == MemoryKind.PROCEDURE or is_procedure_node(node)
            )
            if is_proc:
                procedures.append(_node_entry(node))

            # 4. Active invariants (decisions, requirements, knowledge)
            is_conflicted = meta.get("conflicted") or bool(meta.get("potential_conflicts"))
            is_inv = (
                node.category in ("decision", "requirement")
                or node.scope == MemoryScope.KNOWLEDGE
                or node.memory_kind in (MemoryKind.DECISION, MemoryKind.REQUIREMENT)
            )
            if is_inv and not is_conflicted:
                invariants.append(_node_entry(node))

            # 5. File / Symbol drift check
            if cwd is not None:
                has_file_precond = (
                    bool(meta.get("file_hashes"))
                    or bool(meta.get("symbol_hashes"))
                    or bool(meta.get("require_files_exist"))
                )
                if has_file_precond:
                    ok, reason = check_node_preconditions(node, cwd=cwd)
                    if not ok and reason.startswith(DRIFT_REASON_PREFIXES):
                        drift_entry: dict[str, Any] = {
                            "kind": "drift",
                            "node_id": node.id,
                            "title": node.title,
                            "reason": reason,
                        }
                        temporal = diagnose_temporal_drift(node, cwd=cwd)
                        if temporal:
                            drift_entry["anchored_commit"] = temporal["anchored_commit"]
                            drift_entry["current_commit"] = temporal["current_commit"]
                        drift.append(drift_entry)

    # 6. Transitive drift propagation across DERIVED_FROM, SUMMARY_OF, VERSION_OF, RELATED_TO
    direct_drift_ids = {item["node_id"] for item in drift}
    if cwd is not None:
        transitive_relations = {
            RelationType.DERIVED_FROM,
            RelationType.SUMMARY_OF,
            RelationType.VERSION_OF,
            RelationType.RELATED_TO,
            "derived_from",
            "summary_of",
            "version_of",
            "related_to",
        }
        for node in matched_nodes:
            if node.status != MemoryStatus.ACTIVE or node.id in direct_drift_ids:
                continue

            edge_ids = memory.adjacency.get(node.id, set()) | memory.reverse_adjacency.get(node.id, set())
            for edge_id in sorted(edge_ids):
                edge = memory.edges.get(edge_id)
                if edge is None or edge.relation_type not in transitive_relations:
                    continue
                nb_id = edge.to_node_id if edge.from_node_id == node.id else edge.from_node_id
                if nb_id == node.id:
                    continue
                nb = memory.nodes.get(nb_id)
                if nb is None or nb.folded_into is not None:
                    continue
                if not memory.is_visible(
                    nb,
                    owner=identity.get("owner"),
                    statuses={nb.status},
                    workspace_id=identity.get("workspace_id"),
                    project_id=identity.get("project_id"),
                    session_id=identity.get("session_id"),
                    task_id=identity.get("task_id"),
                ):
                    continue
                nb_meta = nb.metadata or {}
                has_precond = (
                    bool(nb_meta.get("file_hashes"))
                    or bool(nb_meta.get("symbol_hashes"))
                    or bool(nb_meta.get("require_files_exist"))
                )
                if not has_precond:
                    continue
                ok, reason = check_node_preconditions(nb, cwd=cwd)
                if not ok and reason.startswith(DRIFT_REASON_PREFIXES):
                    drift_entry = {
                        "kind": "drift",
                        "node_id": node.id,
                        "title": node.title,
                        "reason": f"transitive_drift via {nb.id}: {reason}",
                        "via_node_id": nb.id,
                    }
                    temporal = diagnose_temporal_drift(node, cwd=cwd)
                    if temporal:
                        drift_entry["anchored_commit"] = temporal["anchored_commit"]
                        drift_entry["current_commit"] = temporal["current_commit"]
                    drift.append(drift_entry)
                    break

    # 6b. Physical AST Import-Graph Drift Propagation (Multi-Hop)
    if cwd is not None and transitive_list:
        existing_drift_ids = {item["node_id"] for item in drift}
        for item in transitive_list:
            imp_file = item["file"]
            imp_lower = imp_file.lower()
            imp_base_lower = Path(imp_file).name.lower()
            imp_anchor_ids = memory.lexical.nodes_for_anchor(imp_file)
            if imp_base_lower != imp_lower:
                imp_anchor_ids |= memory.lexical.nodes_for_anchor(imp_base_lower)

            for upstream_node in memory.nodes.values():
                if upstream_node.folded_into is not None:
                    continue
                if upstream_node.status not in (MemoryStatus.ACTIVE, MemoryStatus.REFUTED):
                    continue
                if not memory.is_visible(
                    upstream_node,
                    owner=identity.get("owner"),
                    statuses={upstream_node.status},
                    workspace_id=identity.get("workspace_id"),
                    project_id=identity.get("project_id"),
                    session_id=identity.get("session_id"),
                    task_id=identity.get("task_id"),
                ):
                    continue

                meta = upstream_node.metadata or {}
                file_hashes = meta.get("file_hashes") or {}
                sym_hashes = meta.get("symbol_hashes") or {}
                meta_files = meta.get("files") or ()
                meta_changed = meta.get("changed_files") or ()

                is_associated = (
                    upstream_node.id in imp_anchor_ids
                    or imp_lower in upstream_node.title.lower()
                    or (bool(imp_base_lower) and imp_base_lower in upstream_node.title.lower())
                    or imp_file in file_hashes
                    or imp_base_lower in file_hashes
                    or any(k.startswith(f"{imp_file}::") or k.startswith(f"{imp_base_lower}::") for k in sym_hashes)
                    or (
                        isinstance(meta_files, (list, tuple, set))
                        and (imp_file in meta_files or imp_base_lower in meta_files)
                    )
                    or (
                        isinstance(meta_changed, (list, tuple, set))
                        and (imp_file in meta_changed or imp_base_lower in meta_changed)
                    )
                )
                if not is_associated:
                    continue

                if upstream_node.id in existing_drift_ids:
                    continue

                drift_reason: str | None = None
                if meta.get("drift_detected"):
                    drift_reason = f"drift_detected: {meta['drift_detected']}"
                elif upstream_node.status == MemoryStatus.REFUTED:
                    drift_reason = "refuted_memory"
                else:
                    has_precond = bool(file_hashes) or bool(sym_hashes) or bool(meta.get("require_files_exist"))
                    if has_precond:
                        ok, reason = check_node_preconditions(upstream_node, cwd=cwd)
                        if not ok and reason.startswith(DRIFT_REASON_PREFIXES):
                            drift_reason = reason

                if drift_reason is not None:
                    drift_entry = {
                        "kind": "drift",
                        "node_id": upstream_node.id,
                        "title": upstream_node.title,
                        "reason": f"import_drift via {imp_file}: {drift_reason}",
                        "via_file": imp_file,
                        "hop": item["hop"],
                        "via": item["via"],
                    }
                    temporal = diagnose_temporal_drift(upstream_node, cwd=cwd)
                    if temporal:
                        drift_entry["anchored_commit"] = temporal["anchored_commit"]
                        drift_entry["current_commit"] = temporal["current_commit"]
                    drift.append(drift_entry)
                    existing_drift_ids.add(upstream_node.id)

    # 7. Multi-Agent IPC active leases
    active_leases: list[dict[str, Any]] = []
    target_terms: list[str] = [target_lower] if target_lower else []
    if len(target_base) >= 4 and target_base not in target_terms:
        target_terms.append(target_base)
    if target_sym and len(target_sym) >= 3 and target_sym not in target_terms:
        target_terms.append(target_sym)
    if target_file:
        tf_base = Path(target_file).name.lower()
        if len(tf_base) >= 4 and tf_base not in target_terms:
            target_terms.append(tf_base)
    if len(target_base) >= 4 and target_base not in target_terms:
        target_terms.append(target_base)

    if target_terms:
        for node in memory.nodes.values():
            if node.status != MemoryStatus.ACTIVE or node.folded_into is not None:
                continue
            if node.category not in ("goal", "episode"):
                continue
            if (
                node.owner != identity.get("owner")
                or node.workspace_id != identity.get("workspace_id")
                or node.project_id != identity.get("project_id")
            ):
                continue
            if node.session_id is None:
                continue
            caller_session = identity.get("session_id")
            if caller_session is not None and node.session_id == caller_session:
                continue

            if node.category == "goal":
                try:
                    goal_data = json.loads(node.content)
                    if isinstance(goal_data, dict) and goal_data.get("status") == "complete":
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

            try:
                meta_str = json.dumps(node.metadata or {}, ensure_ascii=False).lower()
            except (TypeError, ValueError):
                meta_str = str(node.metadata or {}).lower()

            text_to_search = f"{node.title.lower()} {node.content.lower()} {meta_str}"
            if any(term in text_to_search for term in target_terms):
                active_leases.append(
                    {
                        "node_id": node.id,
                        "session_id": node.session_id,
                        "task_id": node.task_id,
                        "kind": node.category,
                        "summary": node.title,
                    }
                )

    # Sort invariants by confidence descending, then title
    invariants.sort(key=lambda item: (item["confidence"], item["title"]), reverse=True)
    procedures.sort(key=lambda item: (item["confidence"], item["title"]), reverse=True)

    # Collect candidate symbol specs for span localization
    candidate_specs: list[str] = []
    seen_cand_specs: set[str] = set()

    def _add_spec(sp: str) -> None:
        cleaned = sp.strip()
        if cleaned and cleaned not in seen_cand_specs:
            seen_cand_specs.add(cleaned)
            candidate_specs.append(cleaned)

    if "::" in clean_target:
        _add_spec(clean_target)

    lookup_base = Path(lookup_file).name if lookup_file else ""
    for node in matched_nodes:
        meta = node.metadata or {}
        sym_hashes = meta.get("symbol_hashes")
        if isinstance(sym_hashes, dict):
            for k in sym_hashes:
                k_str = str(k).strip()
                if "::" in k_str:
                    f_part = k_str.split("::", 1)[0].strip()
                    if f_part == lookup_file or (lookup_base and Path(f_part).name == lookup_base):
                        _add_spec(k_str)

        sym_items = meta.get("symbols")
        if isinstance(sym_items, (list, tuple, set)):
            for s in sym_items:
                s_str = str(s).strip()
                if not s_str:
                    continue
                if "::" in s_str:
                    f_part = s_str.split("::", 1)[0].strip()
                    if f_part == lookup_file or (lookup_base and Path(f_part).name == lookup_base):
                        _add_spec(s_str)
                elif lookup_file:
                    _add_spec(f"{lookup_file}::{s_str}")

    symbol_spans = locate_symbol_spans(candidate_specs, cwd=cwd) if (cwd is not None and candidate_specs) else {}

    return {
        "entity": clean_target,
        "invariants": invariants,
        "procedures": procedures,
        "pitfalls": pitfalls,
        "conflicts": conflicts,
        "verification_commands": verification_commands,
        "drift": drift,
        "warnings": drift,
        "active_leases": active_leases,
        "imported_files": imported_files,
        "transitive_imports": transitive_list,
        "symbol_spans": symbol_spans,
        "state": identity,
    }


def extract_entity_targets(
    memory: FibMind,
    goal: str,
    task_nodes: Iterable[MemoryNode] = (),
    *,
    limit: int = 3,
) -> list[str]:
    """Extract candidate entity targets (paths / symbols) from goal text and task nodes."""
    if limit <= 0:
        return []

    targets: list[str] = []
    seen: set[str] = set()

    # 1. Candidate file paths / module symbols from goal tokens
    for token in goal.split():
        clean = token.strip(".,;:?!'\"`()[]{}")
        if not clean or clean in ("/", "\\"):
            continue
        clean_lower = clean.lower()
        is_candidate = (
            "/" in clean
            or any(clean_lower.endswith(ext) for ext in CODE_DOC_EXTENSIONS)
            or bool(memory.lexical.nodes_for_anchor(clean))
        )
        if is_candidate and clean not in seen:
            seen.add(clean)
            targets.append(clean)
            if len(targets) >= limit:
                return targets

    # 2. Extract file paths from node metadata files / changed_files across task_nodes
    for node in task_nodes:
        meta = node.metadata or {}
        for key in ("files", "changed_files"):
            val = meta.get(key)
            if isinstance(val, (list, tuple, set)):
                for item in val:
                    item_str = str(item).strip()
                    if item_str and item_str not in seen:
                        seen.add(item_str)
                        targets.append(item_str)
                        if len(targets) >= limit:
                            return targets
            elif isinstance(val, str) and val.strip():
                item_str = val.strip()
                if item_str and item_str not in seen:
                    seen.add(item_str)
                    targets.append(item_str)
                    if len(targets) >= limit:
                        return targets

    return targets
