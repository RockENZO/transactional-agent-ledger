"""Twenty deterministic file-operation tasks.

Each task has a setup (files to create), a goal in plain words, and the
expected final state. Expected None means the path must not exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    goal: str
    setup: dict[str, str]
    expected: dict[str, str | None] = field(default_factory=dict)


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def build_tasks() -> list[Task]:
    tasks: list[Task] = []

    # 1-5: rename a file
    for i in range(1, 6):
        body = f"note {i}\n"
        tasks.append(Task(
            id=f"rename-{i:02d}",
            goal=f"Rename notes/draft{i}.txt to notes/final{i}.txt.",
            setup={f"notes/draft{i}.txt": body},
            expected={f"notes/draft{i}.txt": None, f"notes/final{i}.txt": body},
        ))

    # 6-10: edit one JSON field
    for i in range(1, 6):
        setup = {"config.json": _json({"name": f"svc{i}", "version": i, "debug": True})}
        expected_obj = {"name": f"svc{i}", "version": i, "debug": False}
        tasks.append(Task(
            id=f"json-{i:02d}",
            goal="Set debug to false in config.json and keep the other fields.",
            setup=setup,
            expected={"config.json": _json(expected_obj)},
        ))

    # 11-15: append a line to a log
    for i in range(1, 6):
        before = f"start {i}\n"
        tasks.append(Task(
            id=f"append-{i:02d}",
            goal=f"Append 'done {i}' as a new last line of logs/run{i}.log.",
            setup={f"logs/run{i}.log": before},
            expected={f"logs/run{i}.log": before + f"done {i}\n"},
        ))

    # 16-20: concatenate two inputs into one summary file
    for i in range(1, 6):
        a, b = f"alpha {i}\n", f"beta {i}\n"
        tasks.append(Task(
            id=f"concat-{i:02d}",
            goal=f"Write summary{i}.txt containing a.txt then b.txt, in that order.",
            setup={f"in{i}/a.txt": a, f"in{i}/b.txt": b},
            expected={f"summary{i}.txt": a + b},
        ))

    assert len(tasks) == 20
    return tasks
