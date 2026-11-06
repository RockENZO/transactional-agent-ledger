"""A LangGraph agent whose every step is a ledger transaction.

The graph runs one planned action per node visit. Each action is applied to the workspace
through ``run_step``: the step's claim is committed only if the verifier passes, and otherwise
the workspace is restored and the rejection reason is kept in the graph state. LangGraph's
SQLite checkpointer saves the state after each node, so a run stopped midway resumes from the
last completed step with the same ``thread_id``.

No model is called. The "planner" is a fixed list of actions, which keeps the example
deterministic and runnable without an API key.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent_ledger import AllOf, ExpectedStateVerifier, Ledger, Verifier, run_step


class AgentState(TypedDict):
    step: int
    rejections: list[str]
    committed: list[str]


Action = tuple[str, str]  # (relative path, text to write)


def build_graph(
    *,
    ledger: Ledger,
    workspace: Path,
    actions: list[Action],
    verifier: Verifier,
    checkpointer: Any,
    max_attempts: int = 3,
    attempt_fn: Callable[[int], bool] | None = None,
    interrupt_after_act: bool = False,
):
    """Build the graph. ``attempt_fn(step)`` may return False to simulate a bad attempt (tests)."""

    def act(state: AgentState) -> dict:
        index = state["step"]
        path, text = actions[index]

        def step(ws: Path):
            if attempt_fn is not None and not attempt_fn(index):
                text_to_write = "wrong"  # a deliberately bad attempt the verifier must reject
            else:
                text_to_write = text
            target = ws / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text_to_write, encoding="utf-8")
            return [("file", [path], {"step": index})]

        # The step must also leave exactly this text in the file, so a bad attempt is rejected.
        check = AllOf(verifier, ExpectedStateVerifier({path: text}))
        result = run_step(ledger, workspace, step, check)
        if result.committed:
            return {"step": index + 1, "committed": state["committed"] + [path]}
        return {"rejections": state["rejections"] + [f"step {index}: {result.verdict.reason}"]}

    def route(state: AgentState) -> str:
        if state["step"] >= len(actions):
            return END
        if len(state["rejections"]) >= max_attempts:
            return END
        return "act"

    graph = StateGraph(AgentState)
    graph.add_node("act", act)
    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", route, {"act": "act", END: END})
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["act"] if interrupt_after_act else None,
    )


def open_checkpointer(path: str | Path) -> SqliteSaver:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
