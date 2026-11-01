from pathlib import Path

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
    claim = ledger.committed("files")[0]
    assert claim.provenance == {"tool_call": "write_1"}


def test_failing_verifier_rolls_back_and_rejects(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "keep.txt", "before")
    ledger = Ledger()

    def step(w: Path):
        _write(w, "keep.txt", "after")
        # Claims a file that the step never wrote: a false completion.
        return [("files", ["missing.txt"], {"tool_call": "write_2"})]

    result = run_step(ledger, ws, step, FileExistsVerifier())

    assert not result.committed
    assert "missing file" in result.verdict.reason
    assert (ws / "keep.txt").read_text() == "before"  # rolled back
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
    assert not (ws / "cfg.json").exists()  # rolled back to empty workspace
