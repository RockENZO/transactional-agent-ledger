"""A CrewAI crew whose file writes are ledger transactions.

Two agents work in sequence. The writer has a ``write_file`` tool, and the reviewer has a
``read_file`` tool. Each write runs through ``run_step``: the file is committed only if the
verifier passes, and otherwise the workspace is restored and the rejection goes back to the
agent as the tool result, so the agent can try again. A path that escapes the workspace is
rejected this way.

The model is a scripted ``BaseLLM`` that replays a fixed sequence of ReAct responses, so the
example runs offline with no API key. Swap in a real CrewAI LLM to run it against a model.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool

from agent_ledger import AllOf, ExpectedStateVerifier, FileExistsVerifier, Ledger, run_step


class ScriptedLLM(BaseLLM):
    """Returns the next scripted response on each call, in order."""

    responses: list[str] = []

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        if not self.responses:
            raise RuntimeError("scripted LLM ran out of responses")
        return self.responses.pop(0)

    def supports_function_calling(self) -> bool:
        return False


def build_crew(*, ledger: Ledger, workspace: Path, llm: BaseLLM) -> Crew:
    ws = workspace.resolve()
    # CrewAI may run tools on worker threads. One step at a time keeps rollbacks from interleaving.
    lock = threading.Lock()

    @tool("write_file")
    def write_file(path: str, text: str) -> str:
        """Write text to a file in the workspace. Args: path (relative to the workspace), text."""

        def step(w: Path):
            target = (w / path).resolve()
            if not target.is_relative_to(w):
                raise ValueError(f"path escapes workspace: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            return [("file", [path], {"tool": "write_file"})]

        check = AllOf(FileExistsVerifier(), ExpectedStateVerifier({path: text}))
        with lock:
            result = run_step(ledger, ws, step, check)
        if result.committed:
            return f"committed {path}"
        return f"rejected {path}: {result.verdict.reason}"

    @tool("read_file")
    def read_file(path: str) -> str:
        """Read a file in the workspace. Args: path (relative to the workspace)."""
        target = (ws / path).resolve()
        if not target.is_relative_to(ws) or not target.is_file():
            return f"missing {path}"
        return target.read_text(encoding="utf-8")

    writer = Agent(
        role="Writer",
        goal="Save the requested value to the requested file.",
        backstory="You write values to files with write_file.",
        llm=llm,
        tools=[write_file],
        allow_delegation=False,
    )
    reviewer = Agent(
        role="Reviewer",
        goal="Read the saved file and report its value.",
        backstory="You check files with read_file.",
        llm=llm,
        tools=[read_file],
        allow_delegation=False,
    )
    write_task = Task(
        description="Write the price 12.50 to price.txt.",
        expected_output="A confirmation that price.txt was saved.",
        agent=writer,
    )
    review_task = Task(
        description="Read price.txt and report the price in it.",
        expected_output="The price value.",
        agent=reviewer,
        context=[write_task],
    )
    return Crew(
        agents=[writer, reviewer],
        tasks=[write_task, review_task],
        process=Process.sequential,
        verbose=False,
    )


# Writer: first tries an escaping path (rejected), then the right file. Reviewer: reads it back.
SCRIPT = [
    'Thought: save it\nAction: write_file\nAction Input: {"path": "../escape.txt", "text": "12.50"}',
    'Thought: the path was rejected, so use a workspace file\n'
    'Action: write_file\nAction Input: {"path": "price.txt", "text": "12.50"}',
    'Thought: saved\nFinal Answer: saved price.txt',
    'Thought: read it\nAction: read_file\nAction Input: {"path": "price.txt"}',
    'Thought: the file has the price\nFinal Answer: the price is 12.50',
]


def run(workspace: Path, ledger: Ledger | None = None) -> tuple[str, Ledger]:
    ledger = ledger or Ledger()
    llm = ScriptedLLM(model="scripted", responses=list(SCRIPT))
    crew = build_crew(ledger=ledger, workspace=workspace, llm=llm)
    output = crew.kickoff()
    return output.raw, ledger


if __name__ == "__main__":
    import sys
    import tempfile

    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="crew-ws-"))
    ws.mkdir(parents=True, exist_ok=True)
    answer, ledger = run(ws)
    print("answer:", answer)
    print("committed:", [c.key for c in ledger.committed()], "rejected:", ledger.rejected_count())
