"""The dsh session importer turns tool/call + tool/result pairs into a dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from evals.baselines import EvaluationDataset
from evals.import_dsh_session import (
    brain_tool_name,
    build_report,
    import_session,
    pair_tool_exchanges,
    read_events,
)


def _event(seq: int, kind: str, data: dict) -> str:
    return json.dumps({"seq": seq, "type": kind, "time": 1_725_000_000_000 + seq, "data": data})


def _text_result(call_id: str, payload: dict, turn: int = 1, step: int = 1) -> dict:
    return {
        "turn": turn,
        "step": step,
        "message": {
            "role": "tool",
            "callId": call_id,
            "content": [{"type": "text", "text": json.dumps(payload)}],
        },
    }


def _sample_log_lines() -> list[str]:
    identity = {"owner": "iroha", "workspace_id": "fibmind", "project_id": "fibmind", "session_id": "s-1"}
    lines = [
        json.dumps({"version": 0, "id": "s-1", "cwd": "/tmp/fibmind"}),  # header, no type
        _event(1, "turn/start", {"turn": 1}),
        _event(
            2,
            "tool/call",
            {
                "turn": 1,
                "step": 1,
                "callId": "c1",
                "name": "mcp__fibbrain__fibbrain_remember",
                "arguments": json.dumps(
                    {
                        "category": "preference",
                        "title": "Validation drink",
                        "content": "The validation drink is lapsang-42.",
                        "tags": ["drink"],
                        **identity,
                    }
                ),
            },
        ),
        _event(3, "tool/result", _text_result("c1", {"verdict": "write", "node_id": "node_aaa", "title": "Validation drink"})),
        _event(
            4,
            "tool/call",
            {
                "turn": 1,
                "step": 2,
                "callId": "c2",
                "name": "mcp__fibbrain__fibbrain_remember",
                "arguments": json.dumps(
                    {"category": "preference", "title": "Validation drink", "content": "The validation drink is lapsang-42.", **identity}
                ),
            },
        ),
        _event(5, "tool/result", _text_result("c2", {"verdict": "skip", "reason": "duplicate"}, step=2)),
        _event(6, "tool/call", {"turn": 1, "step": 3, "callId": "c3", "name": "read_file", "arguments": "{\"path\":\"x\"}"}),
        _event(7, "tool/result", _text_result("c3", {"ok": True}, step=3)),
        _event(8, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        _event(9, "turn/start", {"turn": 2}),
        _event(
            10,
            "tool/call",
            {
                "turn": 2,
                "step": 1,
                "callId": "c4",
                "name": "mcp__fibbrain__fibbrain_recall",
                "arguments": json.dumps({"goal": "What is my validation drink?", **identity}),
            },
        ),
        _event(
            11,
            "tool/result",
            _text_result(
                "c4",
                {"goal": "What is my validation drink?", "text": "- [preference] Validation drink: ...", "hits": [{"node_id": "node_aaa", "title": "Validation drink"}]},
                turn=2,
            ),
        ),
        _event(12, "turn/end", {"turn": 2, "reason": {"kind": "completed"}}),
        '{"seq": 13, "type": "tool/call", "data": {"turn": 3, "step": 1, "callId": "c5"',  # torn tail
    ]
    return lines


class ToolNameTests(unittest.TestCase):
    def test_strips_dsh_prefix(self) -> None:
        self.assertEqual(brain_tool_name("mcp__fibbrain__fibbrain_recall"), "fibbrain_recall")
        self.assertEqual(brain_tool_name("mcp__my-memory__fibmind_search"), "fibmind_search")
        self.assertEqual(brain_tool_name("fibbrain_plan"), "fibbrain_plan")
        self.assertIsNone(brain_tool_name("read_file"))
        self.assertIsNone(brain_tool_name("mcp__github__list_issues"))


class ImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "session.jsonl"
        self.path.write_text("\n".join(_sample_log_lines()) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_reads_events_and_ignores_header_and_torn_tail(self) -> None:
        events = read_events(self.path)
        self.assertEqual([event["seq"] for event in events], list(range(1, 13)))

    def test_pairs_calls_with_results(self) -> None:
        exchanges = pair_tool_exchanges(read_events(self.path))
        self.assertEqual([item.call_id for item in exchanges], ["c1", "c2", "c3", "c4"])
        self.assertEqual(exchanges[0].result["verdict"], "write")
        self.assertEqual(exchanges[3].arguments["goal"], "What is my validation drink?")

    def test_builds_dataset_with_auto_labelled_case(self) -> None:
        dataset, report = import_session(self.path, name="probe")
        self.assertEqual(report.tool_calls_seen, 3)  # c1, c2, c4 — not read_file
        self.assertEqual(len(dataset["memories"]), 1)
        self.assertEqual(len(dataset["cases"]), 1)
        memory = dataset["memories"][0]
        self.assertEqual(memory["id"], "validation_drink")
        self.assertEqual(memory["owner"], "iroha")
        self.assertEqual(memory["metadata"]["dsh_node_id"], "node_aaa")
        case = dataset["cases"][0]
        self.assertEqual(case["relevant_ids"], ["validation_drink"])
        self.assertEqual(case["workspace_id"], "fibmind")
        self.assertTrue(any("duplicate" in line for line in report.skipped))

    def test_output_is_a_valid_evaluation_dataset(self) -> None:
        dataset, _ = import_session(self.path, name="probe")
        target = Path(self.tmp.name) / "probe.json"
        target.write_text(json.dumps(dataset), encoding="utf-8")
        loaded = EvaluationDataset.from_path(target)
        self.assertEqual(loaded.name, "probe")
        self.assertEqual(loaded.cases[0].relevant_ids, ("validation_drink",))

    def test_default_owner_fills_missing_identity(self) -> None:
        exchanges = pair_tool_exchanges(read_events(self.path))
        for exchange in exchanges:
            exchange.arguments.pop("owner", None)
        report = build_report(exchanges, str(self.path), default_owner="fallback")
        self.assertEqual(report.memories[0]["owner"], "fallback")
        self.assertEqual(report.cases[0]["owner"], "fallback")

    @unittest.skipIf(shutil.which("zstd") is None, "zstd CLI not installed")
    def test_reads_zstd_compressed_log(self) -> None:
        compressed = self.path.with_suffix(".jsonl.zstd")
        subprocess.run(["zstd", "-q", "-f", str(self.path), "-o", str(compressed)], check=True)
        events = read_events(compressed)
        self.assertEqual(len(events), 12)


if __name__ == "__main__":
    unittest.main()
