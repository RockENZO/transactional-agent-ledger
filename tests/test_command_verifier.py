import os
import signal
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

from agent_ledger import AllOf, CommandVerifier, ExpectedStateVerifier, Ledger, run_step


def py(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def check(verifier, ws: Path):
    return verifier.check(ws, [])


def test_zero_exit_passes(tmp_path: Path) -> None:
    verdict = check(CommandVerifier(py("pass")), tmp_path)
    assert verdict.ok
    assert "exited 0" in verdict.reason


def test_nonzero_exit_fails_and_keeps_the_output_tail(tmp_path: Path) -> None:
    verdict = check(CommandVerifier(py("import sys; print('boom: 1 failed'); sys.exit(3)")), tmp_path)
    assert not verdict.ok
    assert "exited 3, expected 0" in verdict.reason
    assert "boom: 1 failed" in verdict.reason


def test_stderr_is_included(tmp_path: Path) -> None:
    verdict = check(CommandVerifier(py("import sys; sys.stderr.write('bad thing'); sys.exit(1)")), tmp_path)
    assert "bad thing" in verdict.reason


def test_expected_returncode_can_be_nonzero(tmp_path: Path) -> None:
    assert check(CommandVerifier(py("import sys; sys.exit(2)"), expected_returncode=2), tmp_path).ok


def test_command_runs_in_the_workspace(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("x")
    source = "import os, sys; sys.exit(0 if os.path.exists('marker.txt') else 1)"
    assert check(CommandVerifier(py(source)), tmp_path).ok


def test_long_output_keeps_the_end_where_failures_are(tmp_path: Path) -> None:
    source = "print('a' * 10000); print('FINAL LINE'); raise SystemExit(1)"
    verdict = check(CommandVerifier(py(source), max_output_chars=100), tmp_path)
    assert "FINAL LINE" in verdict.reason
    assert verdict.reason.count("a") < 200
    assert "[...]" in verdict.reason


def test_missing_command_is_a_failed_verification_not_an_exception(tmp_path: Path) -> None:
    verdict = check(CommandVerifier(["definitely-not-a-real-command-xyz"]), tmp_path)
    assert not verdict.ok
    assert "command not found" in verdict.reason


def test_timeout_fails_and_kills_the_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "escaped"
    child = f"import time; from pathlib import Path; time.sleep(1.5); Path({str(marker)!r}).touch()"
    parent = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
    verdict = check(CommandVerifier(py(parent), timeout=0.5), tmp_path)
    assert not verdict.ok
    assert "timed out after 0.5s" in verdict.reason
    time.sleep(2.0)
    assert not marker.exists(), "a grandchild survived the timeout"


def test_bytecode_is_not_written_by_default(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n")
    assert check(CommandVerifier(py("import mod")), tmp_path).ok
    assert not (tmp_path / "__pycache__").exists()


def test_rejects_empty_argv_and_bad_timeout() -> None:
    with pytest.raises(ValueError):
        CommandVerifier([])
    with pytest.raises(ValueError):
        CommandVerifier(py("pass"), timeout=0)


def test_allof_reports_the_first_failure(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("right")
    both = AllOf(
        ExpectedStateVerifier({"a.txt": "right"}),
        CommandVerifier(py("import sys; sys.exit(1)")),
        ExpectedStateVerifier({"a.txt": "never checked"}),
    )
    verdict = check(both, tmp_path)
    assert not verdict.ok
    assert "exited 1" in verdict.reason
    with pytest.raises(ValueError):
        AllOf()


def test_failing_test_run_rolls_the_workspace_back(tmp_path: Path) -> None:
    """The scenario this verifier exists for: a bad edit is undone, a good one is kept."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (ws / "check.py").write_text("from calc import add\nassert add(2, 3) == 5\n")
    ledger = Ledger()
    verifier = CommandVerifier([sys.executable, "check.py"])

    def break_it(w: Path):
        (w / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        return [("edit", ["calc.py"], {"tool_call": "edit_1"})]

    bad = run_step(ledger, ws, break_it, verifier)
    assert not bad.committed
    assert "AssertionError" in bad.verdict.reason
    assert (ws / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert not ledger.is_done("edit")

    def improve(w: Path):
        (w / "calc.py").write_text("def add(a, b):\n    return b + a\n")
        return [("edit", ["calc.py"], {"tool_call": "edit_2"})]

    good = run_step(ledger, ws, improve, verifier)
    assert good.committed
    assert "b + a" in (ws / "calc.py").read_text()
    assert ledger.is_done("edit")


@pytest.mark.skipif(os.name != "posix", reason="process groups and SIGALRM")
def test_interruption_kills_the_command_before_propagating(tmp_path: Path) -> None:
    """Ctrl-C during verification must not leave the command running to write after rollback."""
    marker = tmp_path / "late-write"
    source = f"import time; from pathlib import Path; time.sleep(1.5); Path({str(marker)!r}).write_text('x')"

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    old = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.3)
    try:
        with pytest.raises(KeyboardInterrupt):
            CommandVerifier(py(source), timeout=30).check(tmp_path, [])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    time.sleep(2.0)
    assert not marker.exists(), "the command kept running after the interruption"


def test_memory_stays_bounded_for_noisy_commands(tmp_path: Path) -> None:
    source = "import sys; sys.stdout.write('x' * (16 * 1024 * 1024)); sys.stdout.write('END'); sys.exit(1)"
    tracemalloc.start()
    try:
        verdict = check(CommandVerifier(py(source), max_output_chars=100), tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert not verdict.ok and verdict.reason.endswith("END")
    assert peak < 2 * 1024 * 1024, f"peak allocations {peak} bytes"


@pytest.mark.skipif(os.name != "posix", reason="process groups")
@pytest.mark.parametrize("exit_code", [0, 1])
def test_background_children_are_stopped_when_the_command_exits(tmp_path: Path, exit_code: int) -> None:
    """A command that leaves a child running must not let it write after the verdict."""
    marker = tmp_path / "late-write"
    child = f"import time; from pathlib import Path; time.sleep(1.0); Path({str(marker)!r}).write_text('x')"
    parent = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}]); sys.exit({exit_code})"
    verdict = check(CommandVerifier(py(parent)), tmp_path)
    assert verdict.ok == (exit_code == 0)
    time.sleep(1.8)
    assert not marker.exists(), "a background child outlived the verifier"


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_a_surviving_child_cannot_undo_a_rollback(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "data.txt").write_text("original")
    target = ws / "data.txt"
    child = f"import time; from pathlib import Path; time.sleep(1.0); Path({str(target)!r}).write_text('overwritten')"
    command = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}]); sys.exit(1)"

    def step(w: Path):
        (w / "data.txt").write_text("changed by the step")
        return [("edit", ["data.txt"], {})]

    result = run_step(Ledger(), ws, step, CommandVerifier(py(command)))
    assert not result.committed
    time.sleep(1.8)
    assert target.read_text() == "original"


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_eperm_from_killpg_on_a_finished_group_is_not_a_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS returns EPERM, not ESRCH, for a group holding only the unreaped, finished command."""
    real = os.killpg

    def killpg(pgid: int, sig: int) -> None:
        real(pgid, sig)
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", killpg)
    assert check(CommandVerifier(py("pass")), tmp_path).ok
    assert not check(CommandVerifier(py("import sys; sys.exit(2)")), tmp_path).ok
