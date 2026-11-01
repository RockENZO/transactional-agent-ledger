# File-operations task set (harness self-test)

Twenty deterministic tasks: 5 renames, 5 JSON field edits, 5 log appends, 5 file concatenations. Each task has a setup, a goal, and an expected final state.

`run.py` runs three scripted agents through the ledger and through a no-verifier baseline:

- `oracle` does the work correctly.
- `skip` claims completion without touching the workspace.
- `partial` applies every change except the last, then claims completion.

## What this does and does not show

- It checks that the ledger commits correct work and rejects false completions and partial work, with rollback. It runs no language model.
- The `baseline` column only shows that a system with no check accepts every claim. It is a count, not a success rate.
- It says nothing about how often a real local model completes these tasks. That needs a model run, which is not done here.

## Run

```
pip install -e ".[dev]"
pytest
python -m benchmarks.file_ops.run report.json
```
