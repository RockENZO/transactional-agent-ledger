"""Deterministic verifiers. A verifier sees the workspace and claims, never the builder's reasoning."""

from __future__ import annotations

import json
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


class FileExistsVerifier:
    """Passes when every relative path listed in the claim's value exists under the workspace."""

    def check(self, workspace: Path, claims: list[Claim]) -> Verdict:
        for claim in claims:
            for rel in claim.value:
                if not (workspace / rel).is_file():
                    return Verdict(False, f"missing file: {rel}")
        return Verdict(True, "all files exist")


class JsonFieldVerifier:
    """Passes when a JSON file has the expected value at a top-level key."""

    def __init__(self, path: str, field: str, expected: object) -> None:
        self.path = path
        self.field = field
        self.expected = expected

    def check(self, workspace: Path, claims: list[Claim]) -> Verdict:
        target = workspace / self.path
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
