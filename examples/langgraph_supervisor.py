"""A LangGraph supervisor that routes between a writer and a reviewer worker.

The supervisor is a routing function over the graph state. It sends work to the writer until a
write is committed, then to the reviewer, then ends. The writer's file write is a ledger
transaction: a write whose verifier fails is rolled back and recorded as a rejection, and the
writer tries again with its next draft, up to ``max_attempts``. The reviewer reads the committed
file back and checks it against the expected value.

The routing rule is fixed, so the example runs offline without a model. A real supervisor would
make the same routing decision with an LLM. The graph structure, state, and ledger calls stay the
same.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_ledger import ExpectedStateVerifier, Ledger, run_step


class SupervisorState(TypedDict):
    attempts: int
    rejections: list[str]
    committed: bool
    reviewed: bool
    review: str


def build_graph(
    *,
    ledger: Ledger,
    workspace: Path,
    path: str,
    expected: str,
    drafts: list[str],
    max_attempts: int = 3,
    checkpointer: Any = None,
):
    """Build the supervisor graph. ``drafts`` holds the writer's successive attempts."""
    ws = workspace.resolve()

    def writer(state: SupervisorState) -> dict:
        text = drafts[state["attempts"]] if state["attempts"] < len(drafts) else ""

        def step(w: Path):
            target = w / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            return [("file", [path], {"worker": "writer", "attempt": state["attempts"]})]

        result = run_step(ledger, ws, step, ExpectedStateVerifier({path: expected}))
        update: dict = {"attempts": state["attempts"] + 1}
        if result.committed:
            update["committed"] = True
        else:
            update["rejections"] = state["rejections"] + [result.verdict.reason]
        return update

    def reviewer(state: SupervisorState) -> dict:
        target = (ws / path).resolve()
        if not target.is_relative_to(ws) or not target.is_file():
            return {"reviewed": True, "review": f"missing {path}"}
        text = target.read_text(encoding="utf-8")
        verdict = "ok" if text == expected else f"mismatch: {text!r}"
        return {"reviewed": True, "review": verdict}

    def route(state: SupervisorState) -> str:
        if state["reviewed"]:
            return END
        if state["committed"]:
            return "reviewer"
        if state["attempts"] >= max_attempts:
            return END
        return "writer"

    graph = StateGraph(SupervisorState)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    targets = {"writer": "writer", "reviewer": "reviewer", END: END}
    graph.add_conditional_edges(START, route, targets)
    graph.add_conditional_edges("writer", route, targets)
    graph.add_edge("reviewer", END)
    return graph.compile(checkpointer=checkpointer)


def run(workspace: Path, *, path: str, expected: str, drafts: list[str], ledger: Ledger | None = None,
        max_attempts: int = 3) -> tuple[SupervisorState, Ledger]:
    ledger = ledger or Ledger()
    graph = build_graph(ledger=ledger, workspace=workspace, path=path, expected=expected,
                        drafts=drafts, max_attempts=max_attempts)
    initial: SupervisorState = {
        "attempts": 0, "rejections": [], "committed": False, "reviewed": False, "review": "",
    }
    final = graph.invoke(initial)
    return final, ledger


if __name__ == "__main__":
    import sys
    import tempfile

    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="supervisor-ws-"))
    ws.mkdir(parents=True, exist_ok=True)
    final, ledger = run(ws, path="price.txt", expected="12.50", drafts=["wrong", "12.50"])
    print("review:", final["review"])
    print("rejections:", final["rejections"])
    print("committed:", [c.key for c in ledger.committed()])
