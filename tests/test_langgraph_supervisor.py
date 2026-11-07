import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from agent_ledger import Ledger  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "langgraph_supervisor",
    Path(__file__).resolve().parents[1] / "examples" / "langgraph_supervisor.py",
)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def _run(tmp_path: Path, drafts: list[str], max_attempts: int = 3):
    ws = tmp_path / "ws"
    ws.mkdir()
    return example.run(ws, path="price.txt", expected="12.50", drafts=drafts,
                       ledger=Ledger(), max_attempts=max_attempts), ws


def test_supervisor_writes_then_reviews_the_committed_file(tmp_path: Path) -> None:
    (final, ledger), ws = _run(tmp_path, drafts=["12.50"])
    assert (ws / "price.txt").read_text(encoding="utf-8") == "12.50"
    assert final["committed"] and final["reviewed"]
    assert final["review"] == "ok"
    assert final["rejections"] == []
    assert [c.key for c in ledger.committed()] == ["file"]


def test_rejected_draft_is_rolled_back_and_the_writer_retries(tmp_path: Path) -> None:
    (final, ledger), ws = _run(tmp_path, drafts=["wrong", "12.50"])
    assert (ws / "price.txt").read_text(encoding="utf-8") == "12.50"
    assert len(final["rejections"]) == 1
    assert final["review"] == "ok"
    assert len(ledger.committed()) == 1


def test_supervisor_gives_up_after_max_attempts_and_does_not_review(tmp_path: Path) -> None:
    (final, ledger), ws = _run(tmp_path, drafts=["wrong", "still wrong", "nope"], max_attempts=3)
    assert not (ws / "price.txt").exists()
    assert not final["committed"]
    assert not final["reviewed"]
    assert len(final["rejections"]) == 3
    assert ledger.committed() == []


def test_path_outside_the_workspace_is_refused_and_leaves_the_file_alone(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    final, _ledger = example.run(ws, path="../outside.txt", expected="12.50", drafts=["12.50"],
                                 ledger=Ledger(), max_attempts=2)
    assert outside.read_text(encoding="utf-8") == "original"
    assert not final["committed"]
    assert final["rejections"]
