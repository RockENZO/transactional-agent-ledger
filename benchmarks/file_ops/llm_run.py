"""Run the 20 file-operation tasks with a local model served by Ollama.

The model gets file tools inside one workspace per task. It must call `done`
with the paths it changed. Each run goes through the ledger's run_step, so a
claim is committed only if the expected state holds. Ground truth is recorded
from the workspace before the ledger decides, so the two can be compared.

Uses only the standard library. Talks to Ollama's /api/chat at localhost.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from agent_ledger import ExpectedStateVerifier, Ledger, run_step
from agent_ledger.verifier import resolve_inside

from benchmarks.file_ops.run import materialize
from benchmarks.file_ops.tasks import Task, build_tasks

MAX_TURNS = 12

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List files under a workspace-relative directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a workspace-relative file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write text to a workspace-relative file, creating it if needed.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "delete_file", "description": "Delete a workspace-relative file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "done", "description": "Call once the task is finished. List every path you changed or removed.",
        "parameters": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                       "required": ["paths"]}}},
]


def _call_tool(ws: Path, name: str, args: dict) -> str:
    rel = str(args.get("path", ""))
    if name == "list_files":
        target = resolve_inside(ws, rel or ".")
        if target is None or not target.is_dir():
            return f"error: not a directory inside workspace: {rel}"
        return "\n".join(sorted(str(p.relative_to(ws.resolve())) for p in target.rglob("*") if p.is_file()))
    if name in ("read_file", "write_file", "delete_file"):
        target = resolve_inside(ws, rel)
        if target is None:
            return f"error: path escapes workspace: {rel}"
        if name == "read_file":
            return target.read_text(encoding="utf-8") if target.is_file() else f"error: no such file: {rel}"
        if name == "write_file":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(args.get("content", "")), encoding="utf-8")
            return f"wrote {rel}"
        if target.is_file() or target.is_symlink():
            target.unlink()
            return f"deleted {rel}"
        return f"error: no such file: {rel}"
    return f"error: unknown tool {name}"


def _chat(host: str, model: str, messages: list[dict]) -> dict:
    body = json.dumps({"model": model, "messages": messages, "tools": TOOLS,
                       "stream": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]


def run_agent(host: str, model: str, ws: Path, task: Task) -> list[str] | None:
    """Return the paths the model reported as done, or None if it never called done."""
    messages = [
        {"role": "system", "content": "You edit files in a workspace using the tools. Work only with relative paths. When finished, call done with the paths you changed."},
        {"role": "user", "content": task.goal},
    ]
    for _ in range(MAX_TURNS):
        msg = _chat(host, model, messages)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": "Call a tool. When finished, call done."})
            continue
        for call in calls:
            fn = call["function"]
            name, args = fn["name"], fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name == "done":
                paths = args.get("paths") or []
                return [str(p) for p in paths]
            result = _call_tool(ws, name, args)
            messages.append({"role": "tool", "content": result, "name": name})
    return None


def run_task(host: str, model: str, task: Task, root: Path) -> dict:
    ws = materialize(task, root)
    ledger_dir = Path(tempfile.mkdtemp(prefix="ledger-db-"))
    ledger = Ledger(ledger_dir / "task.db")
    outcome: dict = {"id": task.id, "claimed": False, "state_ok": False, "error": None}

    def step(w: Path):
        claimed = run_agent(host, model, w, task)
        if claimed is None:
            raise RuntimeError("model never called done")
        outcome["claimed"] = True
        outcome["state_ok"] = ExpectedStateVerifier(task.expected).check(w, []).ok
        return [(task.id, claimed, {"model": model})]

    try:
        result = run_step(ledger, ws, step, ExpectedStateVerifier(task.expected))
        outcome["committed"] = result.committed
        outcome["reason"] = result.verdict.reason
    except Exception as exc:  # the step itself failed; nothing was committed
        outcome["committed"] = False
        outcome["error"] = str(exc)
    finally:
        ledger.close()
        shutil.rmtree(ledger_dir, ignore_errors=True)
    return outcome


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    true_success = sum(r["state_ok"] for r in rows)
    claimed = sum(r["claimed"] for r in rows)
    committed = sum(bool(r.get("committed")) for r in rows)
    return {
        "tasks": total,
        "true_success": true_success,
        "claimed_done": claimed,
        "ledger_committed": committed,
        "ledger_rejected": total - committed,
        "baseline_accepted_without_check": claimed,
        "baseline_false_completions": sum(r["claimed"] and not r["state_ok"] for r in rows),
        "ledger_missed_successes": sum(r["state_ok"] and not r.get("committed") for r in rows),
        "ledger_false_commits": sum(bool(r.get("committed")) and not r["state_ok"] for r in rows),
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    root = Path(tempfile.mkdtemp(prefix="file-ops-llm-"))
    rows = []
    try:
        for task in build_tasks():
            row = run_task(args.host, args.model, task, root)
            rows.append(row)
            print(json.dumps(row), file=sys.stderr)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    report = {"model": args.model, "summary": summarize(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
