"""Run one agent step as a transaction on a workspace directory.

Flow: snapshot the workspace (symlinks preserved), run the step, record its
claims as pending, ask a verifier. On pass, claims are committed. On fail, or
if the step raises, the workspace is restored and claims are rejected. An
interruption (KeyboardInterrupt, SystemExit) also restores the workspace before
it propagates. The snapshot is removed only after a successful restore.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent_ledger.ledger import ClaimStatus, Ledger
from agent_ledger.verifier import Verdict, Verifier

# A step receives the workspace and returns (claim_key, value, provenance) tuples.
StepFn = Callable[[Path], list[tuple[str, Any, dict[str, Any]]]]


@dataclass(frozen=True)
class StepResult:
    committed: bool
    verdict: Verdict
    claim_ids: list[int]


def _restore(workspace: Path, snapshot: Path) -> None:
    if workspace.is_symlink() or workspace.is_file():
        workspace.unlink()
    elif workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(snapshot, workspace, symlinks=True)


def _check_ledger_outside(ledger: Ledger, workspace: Path) -> None:
    if ledger.path == ":memory:":
        return
    db = Path(ledger.path).resolve()
    if db.is_relative_to(workspace):
        raise ValueError(
            f"ledger database must live outside the workspace: {db}"
        )


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
    _check_ledger_outside(ledger, workspace)

    snapshot_root = Path(tempfile.mkdtemp(prefix="ledger-snap-"))
    snapshot = snapshot_root / "ws"
    shutil.copytree(workspace, snapshot, symlinks=True)
    restored = False
    try:
        ids: list[int] = []
        try:
            produced = step(workspace)
            ids = ledger.add_claims(produced)
        except Exception as exc:
            verdict = Verdict(False, f"step error: {exc}")
        else:
            scoped = [
                c for c in ledger.get_many(ids)
                if verify_keys is None or c.key in verify_keys
            ]
            try:
                verdict = verifier.check(workspace, scoped)
            except Exception as exc:  # a crashing verifier is a failed verification
                verdict = Verdict(False, f"verifier error: {exc}")

        if verdict.ok:
            ledger.finalize(ids, ClaimStatus.COMMITTED, verdict.reason)
            return StepResult(True, verdict, ids)

        _restore(workspace, snapshot)
        restored = True
        ledger.finalize(ids, ClaimStatus.REJECTED, verdict.reason)
        return StepResult(False, verdict, ids)
    except BaseException:
        # Interrupted or failed mid-flight: roll back before propagating.
        if not restored:
            _restore(workspace, snapshot)
            restored = True
        raise
    finally:
        if restored:
            shutil.rmtree(snapshot_root, ignore_errors=True)
