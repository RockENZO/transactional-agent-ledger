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
    - On timeout or interruption (for example Ctrl-C) the command's process group is killed
      (POSIX) and the direct child reaped before control returns. Elsewhere only the direct
      child is killed.
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
                process.wait(timeout=self.timeout)
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
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
