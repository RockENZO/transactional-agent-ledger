import importlib.util
from pathlib import Path

from agent_ledger import FileExistsVerifier, Ledger

_SPEC = importlib.util.spec_from_file_location(
    "langgraph_agent",
    Path(__file__).resolve().parents[1] / "examples" / "langgraph_agent.py",
)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


ACTIONS = [("a.txt", "one"), ("b.txt", "two")]


def _run(tmp_path: Path, attempt_fn=None, thread="t1", verifier=None):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    ledger = Ledger(tmp_path / "task.db")
    checkpointer = example.open_checkpointer(tmp_path / "graph.db")
    graph = example.build_graph(
        ledger=ledger,
        workspace=ws,
        actions=ACTIONS,
        verifier=verifier or FileExistsVerifier(),
        checkpointer=checkpointer,
        attempt_fn=attempt_fn,
    )
    config = {"configurable": {"thread_id": thread}}
    state = graph.invoke({"step": 0, "rejections": [], "committed": []}, config)
    return state, ws, ledger


def test_all_steps_commit_through_the_graph(tmp_path: Path) -> None:
    state, ws, ledger = _run(tmp_path)
    assert state["committed"] == ["a.txt", "b.txt"]
    assert state["rejections"] == []
    assert (ws / "b.txt").read_text(encoding="utf-8") == "two"
    assert len(ledger.committed()) == 2


def test_rejected_attempt_is_rolled_back_and_retried(tmp_path: Path) -> None:
    # The first attempt at step 0 is bad, so it is rolled back, then the graph retries it.
    calls = {"n": 0}

    def attempt(index: int) -> bool:
        if index == 0:
            calls["n"] += 1
            return calls["n"] > 1
        return True

    state, ws, ledger = _run(tmp_path, attempt_fn=attempt)
    assert state["committed"] == ["a.txt", "b.txt"]
    assert len(state["rejections"]) == 1
    assert (ws / "a.txt").read_text(encoding="utf-8") == "one"
    statuses = [c.status.value for c in ledger.committed()]
    assert statuses == ["committed", "committed"]


def test_gives_up_after_max_attempts(tmp_path: Path) -> None:
    state, ws, _ = _run(tmp_path, attempt_fn=lambda index: False, thread="t-fail")
    assert state["committed"] == []
    assert len(state["rejections"]) == 3
    assert not (ws / "a.txt").exists()


def test_resumes_from_checkpoint_with_same_thread_id(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = Ledger(tmp_path / "task.db")
    checkpointer = example.open_checkpointer(tmp_path / "graph.db")
    config = {"configurable": {"thread_id": "resume"}}

    # Stop after the first node, as if the process died there, then resume with the same thread id.
    first = example.build_graph(
        ledger=ledger, workspace=ws, actions=ACTIONS,
        verifier=FileExistsVerifier(), checkpointer=checkpointer, interrupt_after_act=True,
    )
    partial = first.invoke({"step": 0, "rejections": [], "committed": []}, config)
    assert partial["committed"] == ["a.txt"]

    resumed = example.build_graph(
        ledger=ledger, workspace=ws, actions=ACTIONS,
        verifier=FileExistsVerifier(), checkpointer=checkpointer,
    )
    final = resumed.invoke(None, config)
    assert final["committed"] == ["a.txt", "b.txt"]
    assert len(ledger.committed()) == 2
