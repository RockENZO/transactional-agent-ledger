# transactional-agent-ledger

Status: early (0.1), the API may still change. It is used as the optional completion gate (`--verify`) of [Odyssey](https://github.com/RockENZO/odyssey).

A tool-using agent should not decide on its own that a task is done. This library keeps task claims in a ledger outside the model's context. Each agent step runs as a transaction on a workspace directory:

1. Snapshot the workspace.
2. Run the step, which returns claims with provenance (for example, the tool call that produced the evidence).
3. A deterministic verifier checks the workspace. It sees the artifacts, not the builder's reasoning.
4. Pass: claims are committed. Fail: the workspace is restored and the claims are rejected with the verifier's reason.

Only committed claims count as done. The ledger is SQLite, so it can be inspected and replayed.

## Usage

```python
from pathlib import Path
from agent_ledger import Ledger, FileExistsVerifier, run_step

ledger = Ledger("task.db")

def step(ws: Path):
    (ws / "report.txt").write_text("done")
    return [("files", ["report.txt"], {"tool_call": "write_1"})]

result = run_step(ledger, Path("workspace"), step, FileExistsVerifier())
print(result.committed, result.verdict.reason)
```

## LangGraph agent with checkpointed ledger steps

`examples/langgraph_agent.py` runs each planned action as a `run_step` transaction inside a LangGraph graph. A failed verification rolls the workspace back and the node retries, up to a limit. LangGraph's SQLite checkpointer saves state after each node, so a run stopped midway resumes with the same `thread_id`. No model is called: the plan is a fixed list, so the example runs without an API key.

```bash
pip install -e ".[dev,langgraph]"
PYTHONPATH=src python -m pytest -q tests/test_langgraph_agent.py
```

## Gating on a recorded tool trajectory

`CassetteVerifier` passes a step only when its tool-call claims match a recorded cassette, in order, by name and arguments. Use a claim with key `tool_call`, value `{"name", "arguments"}`, and optional provenance `call_id`. The cassette is the JSON list written by `trajectory_replay.save`. It is read when the verifier is created, so keep it outside the workspace. Results are not compared.

```python
from agent_ledger import CassetteVerifier
verifier = CassetteVerifier("cassettes/task-01.json")
```

## Gating a coding agent on its tests

```python
import sys
from pathlib import Path
from agent_ledger import AllOf, CommandVerifier, ExpectedStateVerifier, Ledger, run_step

verifier = AllOf(
    CommandVerifier([sys.executable, "-m", "pytest", "-q"], timeout=300),
    ExpectedStateVerifier({"tests/test_calc.py": TEST_FILE_TEXT}),  # the agent may not rewrite the tests
)
result = run_step(Ledger("task.db"), Path("workspace"), step, verifier)
if not result.committed:
    print(result.verdict.reason)  # the command's exit code and the tail of its output
```

A failing command restores the workspace and rejects the step's claims; the reason ends with the
command's output, so it can be given back to the agent. The command is not sandboxed: only
changes inside the workspace are rolled back, not network calls or files elsewhere. A verifier
only checks what its command checks, which is why the example also pins the test file.

## Scope

- Included: ledger, snapshot/restore transactions, file-existence, JSON-field and expected-state verifiers, `CommandVerifier` (run a command such as the test suite), `AllOf`, tests.
- Also included: a 20-task file-operations set with a scripted harness self-test ([benchmarks/file_ops](benchmarks/file_ops/README.md)). It runs no language model, so it shows that the ledger commits correct work and rejects false completions, not how often a model succeeds.
- Not included: model calls, a builder agent, model-based verifiers.
- Snapshots copy the whole workspace. That is fine for small task sets, not for large trees.

## Tests

```
pip install -e ".[dev]"
pytest
```

## License

MIT.
