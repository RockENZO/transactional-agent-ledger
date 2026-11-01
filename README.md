# transactional-agent-ledger

Status: skeleton, pre-release. Part of the planned Design A from the frontier agent architecture proposal.

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

## Scope of this skeleton

- Included: ledger, snapshot/restore transactions, file-existence and JSON-field verifiers, tests.
- Not included: model calls, a builder agent, the 20-task benchmark, model-based verifiers, trajectory-replay integration.
- Snapshots copy the whole workspace. That is fine for small task sets, not for large trees.

## Tests

```
pip install -e ".[dev]"
pytest
```

## License

MIT.
