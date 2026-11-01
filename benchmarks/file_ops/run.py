"""Harness self-test for the file-operations tasks.

This runs scripted agents, not language models. It checks that the ledger
accepts correct work, rejects false completions, and that a no-verifier
baseline would accept them. It says nothing about model task success.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

from agent_ledger import ExpectedStateVerifier, Ledger, run_step

from benchmarks.file_ops.tasks import Task, build_tasks

# An agent maps (workspace, task) to the list of (key, value, provenance) claims it makes.
Agent = Callable[[Path, Task], list[tuple[str, object, dict]]]


def materialize(task: Task, root: Path) -> Path:
    ws = root / task.id
    for rel, text in task.setup.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return ws


def oracle_agent(ws: Path, task: Task):
    """Does the work correctly, then claims done."""
    for rel, want in task.expected.items():
        if want is None:
            (ws / rel).unlink(missing_ok=True)
        else:
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            (ws / rel).write_text(want, encoding="utf-8")
    return [(task.id, list(task.expected), {"agent": "oracle"})]


def false_completion_agent(ws: Path, task: Task):
    """Claims done without touching the workspace."""
    return [(task.id, list(task.expected), {"agent": "skip"})]


def partial_agent(ws: Path, task: Task):
    """Applies every expected change except the last, then claims done."""
    items = list(task.expected.items())
    for rel, want in items[:-1]:
        if want is None:
            (ws / rel).unlink(missing_ok=True)
        else:
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            (ws / rel).write_text(want, encoding="utf-8")
    return [(task.id, list(task.expected), {"agent": "partial"})]


AGENTS: dict[str, Agent] = {
    "oracle": oracle_agent,
    "skip": false_completion_agent,
    "partial": partial_agent,
}


def run_with_ledger(agent: Agent, tasks: list[Task], root: Path) -> dict:
    committed = rejected = 0
    rollbacks = 0
    for task in tasks:
        ws = materialize(task, root)
        ledger = Ledger()
        result = run_step(
            ledger, ws,
            lambda w, t=task: agent(w, t),
            ExpectedStateVerifier(task.expected),
        )
        committed += result.committed
        rejected += not result.committed
        rollbacks += not result.committed
        ledger.close()
    return {"tasks": len(tasks), "committed": committed,
            "rejected": rejected, "rollbacks": rollbacks}


def run_baseline(agent: Agent, tasks: list[Task], root: Path) -> dict:
    """No verifier: every claimed completion is accepted."""
    accepted = 0
    for task in tasks:
        ws = materialize(task, root)
        agent(ws, task)
        accepted += 1
    return {"tasks": len(tasks), "accepted_without_check": accepted}


def main(argv: list[str]) -> int:
    tasks = build_tasks()
    root = Path(tempfile.mkdtemp(prefix="file-ops-bench-"))
    try:
        report = {}
        for name, agent in AGENTS.items():
            report[name] = {
                "ledger": run_with_ledger(agent, tasks, root / f"ledger-{name}"),
                "baseline": run_baseline(agent, tasks, root / f"baseline-{name}"),
            }
        out = json.dumps(report, indent=2)
        print(out)
        if argv:
            Path(argv[0]).write_text(out + "\n", encoding="utf-8")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
