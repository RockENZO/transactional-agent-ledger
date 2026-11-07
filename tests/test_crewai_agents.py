import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("crewai")

_SPEC = importlib.util.spec_from_file_location(
    "crewai_agents",
    Path(__file__).resolve().parents[1] / "examples" / "crewai_agents.py",
)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def test_crew_commits_the_good_write_and_rolls_back_the_escaping_one(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    answer, ledger = example.run(ws)

    assert (ws / "price.txt").read_text(encoding="utf-8") == "12.50"
    assert not (tmp_path / "escape.txt").exists()
    # The escaping step raised before producing a claim, so only the good write is in the ledger.
    assert [c.key for c in ledger.committed()] == ["file"]
    assert "12.50" in answer


def test_crew_consumes_the_whole_script(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    llm = example.ScriptedLLM(model="scripted", responses=list(example.SCRIPT))
    from agent_ledger import Ledger

    crew = example.build_crew(ledger=Ledger(), workspace=ws, llm=llm)
    crew.kickoff()
    assert llm.responses == []
