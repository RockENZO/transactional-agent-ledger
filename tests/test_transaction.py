import os
from pathlib import Path

import pytest

from agent_ledger import (
    FileExistsVerifier,
    JsonFieldVerifier,
    Ledger,
    run_step,
)


def _write(ws: Path, rel: str, text: str) -> None:
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_passing_step_commits_claims_and_keeps_changes(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = Ledger()

    def step(w: Path):
        _write(w, "report.txt", "done")
        return [("files", ["report.txt"], {"tool_call": "write_1"})]

    result = run_step(ledger, ws, step, FileExistsVerifier())

    assert result.committed
    assert (ws / "report.txt").read_text() == "done"
    assert ledger.is_done("files")
    assert ledger.committed("files")[0].provenance == {"tool_call": "write_1"}


def test_failing_verifier_rolls_back_and_rejects(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "keep.txt", "before")
    ledger = Ledger()

    def step(w: Path):
        _write(w, "keep.txt", "after")
        return [("files", ["missing.txt"], {"tool_call": "write_2"})]

    result = run_step(ledger, ws, step, FileExistsVerifier())

    assert not result.committed
    assert "missing file" in result.verdict.reason
    assert (ws / "keep.txt").read_text() == "before"
    assert not ledger.is_done("files")
    assert ledger.rejected_count() == 1


def test_step_crash_rolls_back(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "a.txt", "x")
    ledger = Ledger()

    def step(w: Path):
        _write(w, "a.txt", "changed")
        raise RuntimeError("tool failed")

    result = run_step(ledger, ws, step, FileExistsVerifier())

    assert not result.committed
    assert "step error" in result.verdict.reason
    assert (ws / "a.txt").read_text() == "x"


def test_crashing_verifier_counts_as_failure(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = Ledger()

    class Broken:
        def check(self, workspace, claims):
            raise ValueError("boom")

    result = run_step(ledger, ws, lambda w: [("k", 1, {})], Broken())

    assert not result.committed
    assert "verifier error" in result.verdict.reason


def test_json_field_verifier(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = Ledger()

    def step(w: Path):
        _write(w, "cfg.json", '{"status": "draft"}')
        return [("config", "status", {"tool_call": "write_3"})]

    result = run_step(ledger, ws, step, JsonFieldVerifier("cfg.json", "status", "final"))

    assert not result.committed
    assert "expected 'final'" in result.verdict.reason
    assert not (ws / "cfg.json").exists()


def test_file_claim_escaping_workspace_is_rejected(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("host file")
    ledger = Ledger()

    result = run_step(
        ledger, ws,
        lambda w: [("files", ["../outside.txt"], {})],
        FileExistsVerifier(),
    )

    assert not result.committed
    assert "escapes workspace" in result.verdict.reason


def test_symlink_escaping_workspace_is_rejected(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("host file")
    os.symlink(outside, ws / "link.txt")
    ledger = Ledger()

    result = run_step(
        ledger, ws,
        lambda w: [("files", ["link.txt"], {})],
        FileExistsVerifier(),
    )

    assert not result.committed
    assert "escapes workspace" in result.verdict.reason


def test_symlinks_survive_rollback(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "target.txt").write_text("t")
    os.symlink("target.txt", ws / "link.txt")
    os.symlink("does-not-exist", ws / "dangling")
    ledger = Ledger()

    def step(w: Path):
        (w / "new.txt").write_text("x")
        return [("files", ["new.txt"], {})]

    result = run_step(ledger, ws, step, JsonFieldVerifier("missing.json", "k", 1))

    assert not result.committed
    assert os.path.islink(ws / "link.txt")
    assert os.readlink(ws / "link.txt") == "target.txt"
    assert os.path.islink(ws / "dangling")
    assert not (ws / "new.txt").exists()


def test_step_deleting_workspace_is_restored(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "keep.txt", "precious")
    ledger = Ledger()

    def step(w: Path):
        import shutil
        shutil.rmtree(w)
        raise RuntimeError("after deleting")

    result = run_step(ledger, ws, step, FileExistsVerifier())

    assert not result.committed
    assert (ws / "keep.txt").read_text() == "precious"


def test_keyboard_interrupt_restores_and_propagates(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "a.txt", "before")
    ledger = Ledger()

    def step(w: Path):
        _write(w, "a.txt", "half-done")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_step(ledger, ws, step, FileExistsVerifier())

    assert (ws / "a.txt").read_text() == "before"


def test_ledger_inside_workspace_is_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = Ledger(ws / "task.db")

    with pytest.raises(ValueError, match="outside the workspace"):
        run_step(ledger, ws, lambda w: [], FileExistsVerifier())
    ledger.close()
