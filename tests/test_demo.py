"""The README Quick Start runs ``examples/demo.py``; it must keep working."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_end_to_end(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo.py"), "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "demo-brain.json").is_file()
    assert (tmp_path / "demo-memory.json").is_file()
    assert "Search from fix node:" in result.stdout
