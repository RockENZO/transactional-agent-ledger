"""Deterministic verifiers. A verifier sees the workspace and claims, never the builder's reasoning.

Every path is resolved before it is checked. A path that resolves outside the
workspace, including through a symbolic link, fails verification.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_ledger.ledger import Claim


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str


class Verifier(Protocol):
    def check(self, workspace: Path, claims: list[Claim]) -> Verdict: ...


def resolve_inside(workspace: Path, rel: str) -> Path | None:
    """Return the resolved path if it lies inside the workspace, else None."""
    root = workspace.resolve()
    candidate = (root / rel).resolve()
    return candidate if candidate.is_relative_to(root) else None


class FileExistsVerifier:
    """Passes when every relative path listed in the claim's value is a file inside the workspace."""

    def check(self, workspace: Path, claims: list[Claim]) -> Verdict:
        for claim in claims:
            for rel in claim.value:
                target = resolve_inside(workspace, rel)
                if target is None:
                    return Verdict(False, f"path escapes workspace: {rel}")
                if not target.is_file():
                    return Verdict(False, f"missing file: {rel}")
        return Verdict(True, "all files exist")


class JsonFieldVerifier:
    """Passes when a JSON file inside the workspace has the expected value at a top-level key."""

    def __init__(self, path: str, field: str, expected: object) -> None:
        self.path = path
        self.field = field
        self.expected = expected

    def check(self, workspace: Path, claims: list[Claim]) -> Verdict:
        target = resolve_inside(workspace, self.path)
        if target is None:
            return Verdict(False, f"path escapes workspace: {self.path}")
        if not target.is_file():
            return Verdict(False, f"missing file: {self.path}")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return Verdict(False, f"invalid JSON in {self.path}: {exc.msg}")
        if not isinstance(data, dict) or self.field not in data:
            return Verdict(False, f"field {self.field!r} not found in {self.path}")
        if data[self.field] != self.expected:
            return Verdict(False, f"{self.field!r} is {data[self.field]!r}, expected {self.expected!r}")
        return Verdict(True, f"{self.field!r} matches")


class ExpectedStateVerifier:
    """Passes when each path holds the expected text, or is absent when the expected value is None."""

    def __init__(self, expected: dict[str, str | None]) -> None:
        self.expected = expected

    def check(self, workspace: Path, claims: list[Claim]) -> Verdict:
        for rel, want in self.expected.items():
            target = resolve_inside(workspace, rel)
            if target is None:
                return Verdict(False, f"path escapes workspace: {rel}")
            if want is None:
                # Check the entry itself, not its target: a dangling symlink still exists.
                entry = workspace.resolve() / rel
                if entry.exists() or entry.is_symlink() or target.exists():
                    return Verdict(False, f"should be deleted: {rel}")
                continue
            if not target.is_file():
                return Verdict(False, f"missing file: {rel}")
            if target.read_text(encoding="utf-8") != want:
                return Verdict(False, f"wrong content: {rel}")
        return Verdict(True, "expected state reached")


class AllOf:
    """Passes when every verifier passes. Stops at the first failure and reports its reason."""

    def __init__(self, *verifiers: Verifier) -> None:
        if not verifiers:
            raise ValueError("AllOf needs at least one verifier")
        self.verifiers = verifiers

    def check(self, workspace: Path, claims: list[Claim]) -> Verdict:
        reasons = []
        for verifier in self.verifiers:
            verdict = verifier.check(workspace, claims)
            if not verdict.ok:
                return verdict
            reasons.append(verdict.reason)
        return Verdict(True, "; ".join(reasons))


class CommandVerifier:
    """Passes when a command, run with the workspace as its working directory, exits as expected.

    This is what lets the ledger gate a coding agent: the command is usually the project's
    test suite, for example ``["python", "-m", "pytest", "-q"]``. The command is an argument
    list and is not run through a shell.

    The failure reason ends with the tail of the command's combined output, so it can be
    handed back to the agent as the reason a claim was rejected.

    Limits:
    - The command is not sandboxed. It runs with the caller's privileges and can reach
      anything they can; only changes inside the workspace are rolled back on failure.
    - Files the command writes into the workspace (caches, bytecode) are kept when the
      step commits. ``PYTHONDONTWRITEBYTECODE=1`` is set by default to limit this.
    - A verifier only checks what the command checks. An agent that can edit the tests can
      make them pass trivially; combine this with ``ExpectedStateVerifier`` (via ``AllOf``)
      to pin the test files.
    - When the command exits, on timeout, and on interruption (for example Ctrl-C), the
      command's process group is killed (POSIX), so a background child it left behind cannot
      write into the workspace after the verdict. Elsewhere only the direct child is killed.
      Where ``os.waitid`` is missing (macOS) the group is killed after the child is reaped,
      which leaves a vanishingly small window for process-id reuse.
    - Output is spooled to a temporary file and only its tail is read, so memory stays bounded;
      disk use while the command runs is not.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 120.0,
        expected_returncode: int = 0,
        max_output_chars: int = 4000,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not argv:
            raise ValueError("argv must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.argv = list(argv)
        self.timeout = timeout
        self.expected_returncode = expected_returncode
        self.max_output_chars = max_output_chars
        self.env = {"PYTHONDONTWRITEBYTECODE": "1", **(env or {})}

    def check(self, workspace: Path, claims: list[Claim]) -> Verdict:
        label = " ".join(self.argv)
        # Output goes to a temporary file, not a pipe, so a noisy command cannot fill this
        # process's memory; only the tail is read back.
        with tempfile.TemporaryFile() as sink:
            try:
                process = subprocess.Popen(
                    self.argv,
                    cwd=workspace,
                    env={**os.environ, **self.env},
                    stdin=subprocess.DEVNULL,
                    stdout=sink,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name == "posix"),
                )
            except FileNotFoundError:
                return Verdict(False, f"command not found: {self.argv[0]}")
            except OSError as exc:
                return Verdict(False, f"could not run {label!r}: {exc}")
            try:
                self._wait_then_stop_descendants(process)
            except subprocess.TimeoutExpired:
                self._stop(process)
                return Verdict(False, f"{label!r} timed out after {self.timeout:g}s\n{self._tail(sink)}".rstrip())
            except BaseException:
                # An interruption (Ctrl-C, cancellation) must not leave the command running:
                # the caller is about to roll the workspace back and the command could still write to it.
                self._stop(process)
                raise
            if process.returncode == self.expected_returncode:
                return Verdict(True, f"{label!r} exited {process.returncode}")
            return Verdict(
                False,
                f"{label!r} exited {process.returncode}, expected {self.expected_returncode}\n{self._tail(sink)}".rstrip(),
            )

    def _wait_then_stop_descendants(self, process: subprocess.Popen) -> None:
        """Wait for the command to exit, then kill anything it left running in its process group.

        A command can start a background child and exit; that child could still write into the
        workspace after the verdict, racing a rollback or a commit. Where ``os.waitid`` is
        available the direct child is waited for without reaping it, so its process-group id
        cannot be reused before the group is killed.
        """
        deadline = time.monotonic() + self.timeout
        if os.name == "posix" and hasattr(os, "waitid") and hasattr(os, "WNOWAIT"):
            while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(self.argv, self.timeout)
                time.sleep(0.01)
        else:
            process.wait(timeout=self.timeout)
        self._kill_group(process)
        process.wait()

    @staticmethod
    def _kill_group(process: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # ESRCH on Linux. macOS answers EPERM when the only member of the group is the
            # finished, not yet reaped command itself, so nothing is left to stop.
            pass

    def _tail(self, sink) -> str:
        """Last ``max_output_chars`` characters of the captured output."""
        size = sink.seek(0, os.SEEK_END)
        window = self.max_output_chars * 4  # four bytes is the widest UTF-8 character
        sink.seek(max(0, size - window))
        text = sink.read().decode("utf-8", errors="replace").strip()
        if size > window or len(text) > self.max_output_chars:
            return "[...]\n" + text[-self.max_output_chars:]
        return text

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        """Kill the command's process group (POSIX) and reap the direct child."""
        CommandVerifier._kill_group(process)
        if os.name != "posix":
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
