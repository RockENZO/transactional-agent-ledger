"""Run one agent step as a transaction on a workspace directory.

Flow: snapshot the workspace, run the step, record its claims as pending,
ask a verifier. On pass, claims are committed. On fail, the workspace is
restored from the snapshot and claims are rejected with the verifier's reason.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent_ledger.ledger import Claim, ClaimStatus, Ledger
from agent_ledger.verifier import Verdict, Verifier

# A step receives the workspace and returns (claim_key, value, provenance) tuples.
StepFn = Callable[[Path], list[tuple[str, Any, dict[str, Any]]]]


@dataclass(frozen=True)
class StepResult:
    committed: bool
    verdict: Verdict
    claim_ids: list[int]


def _snapshot(workspace: Path) -> Path:
    snap = Path(tempfile.mkdtemp(prefix="ledger-snap-"))
    shutil.copytree(workspace, snap / "ws")
    return snap / "ws"


def _restore(workspace: Path, snapshot: Path) -> None:
    shutil.rmtree(workspace)
    shutil.copytree(snapshot, workspace)


def run_step(
    ledger: Ledger,
    workspace: Path,
    step: StepFn,
    verifier: Verifier,
    verify_keys: list[str] | None = None,
) -> StepResult:
    """Run step on workspace; commit only if verifier passes, otherwise roll back.

    verify_keys limits which claim keys the verifier receives; None means all
    claims produced by this step.
    """
    workspace = workspace.resolve()
    snapshot = _snapshot(workspace)
    try:
        produced = step(workspace)
        ids = [ledger.add_claim(k, v, p) for k, v, p in produced]
        claims: list[Claim] = [ledger.get(i) for i in ids]
        scoped = [c for c in claims if verify_keys is None or c.key in verify_keys]
        try:
            verdict = verifier.check(workspace, scoped)
        except Exception as exc:  # a crashing verifier is a failed verification
            verdict = Verdict(False, f"verifier error: {exc}")

        if verdict.ok:
            for cid in ids:
                ledger.set_status(cid, ClaimStatus.COMMITTED, verdict.reason)
            return StepResult(True, verdict, ids)

        _restore(workspace, snapshot)
        for cid in ids:
            ledger.set_status(cid, ClaimStatus.REJECTED, verdict.reason)
        return StepResult(False, verdict, ids)
    except Exception as exc:
        # Step itself crashed: roll back, record nothing as committed.
        _restore(workspace, snapshot)
        verdict = Verdict(False, f"step error: {exc}")
        return StepResult(False, verdict, [])
    finally:
        shutil.rmtree(snapshot.parent, ignore_errors=True)
