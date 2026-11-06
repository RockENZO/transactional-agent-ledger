import json
from pathlib import Path

from agent_ledger import CassetteVerifier, Ledger, run_step


CASSETTE = [
    {"name": "fetch", "arguments": {"url": "a"}, "result": "PRICE: 12.50", "error": None, "call_id": "c1"},
    {"name": "write", "arguments": {"key": "k", "value": "v"}, "result": "ok", "error": None, "call_id": "c2"},
]


def _cassette(tmp_path: Path, calls=CASSETTE) -> Path:
    path = tmp_path / "cassette.json"
    path.write_text(json.dumps(calls), encoding="utf-8")
    return path


def _claims(items):
    return [("tool_call", {"name": n, "arguments": a}, {"call_id": cid}) for n, a, cid in items]


def _step(items):
    def step(ws: Path):
        return _claims(items)
    return step


GOOD = [("fetch", {"url": "a"}, "c1"), ("write", {"key": "k", "value": "v"}, "c2")]


def test_matching_trajectory_commits(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    result = run_step(Ledger(), ws, _step(GOOD), CassetteVerifier(_cassette(tmp_path)))
    assert result.committed
    assert result.verdict.reason == "2 tool calls match cassette"


def test_wrong_arguments_roll_back_the_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "keep.txt").write_text("before", encoding="utf-8")

    def step(w: Path):
        (w / "keep.txt").write_text("changed", encoding="utf-8")
        (w / "extra.txt").write_text("new", encoding="utf-8")
        return _claims([("fetch", {"url": "WRONG"}, "c1"), ("write", {"key": "k", "value": "v"}, "c2")])

    ledger = Ledger()
    result = run_step(ledger, ws, step, CassetteVerifier(_cassette(tmp_path)))
    assert not result.committed
    assert "call 0" in result.verdict.reason
    assert (ws / "keep.txt").read_text(encoding="utf-8") == "before"
    assert not (ws / "extra.txt").exists()
    assert ledger.committed() == []


def test_out_of_order_calls_fail(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    swapped = [GOOD[1], GOOD[0]]
    result = run_step(Ledger(), ws, _step(swapped), CassetteVerifier(_cassette(tmp_path)))
    assert not result.committed


def test_missing_and_extra_calls_fail(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    cassette = CassetteVerifier(_cassette(tmp_path))
    short = run_step(Ledger(), ws, _step(GOOD[:1]), cassette)
    long = run_step(Ledger(), ws, _step(GOOD + [("write", {"key": "x", "value": "y"}, None)]), cassette)
    assert not short.committed and "expected 2 tool calls, got 1" in short.verdict.reason
    assert not long.committed and "expected 2 tool calls, got 3" in long.verdict.reason


def test_call_id_mismatch_fails_but_missing_call_id_is_not_compared(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    cassette = CassetteVerifier(_cassette(tmp_path))
    bad_id = run_step(Ledger(), ws, _step([("fetch", {"url": "a"}, "other"), GOOD[1]]), cassette)
    no_id = run_step(Ledger(), ws, _step([("fetch", {"url": "a"}, None), GOOD[1]]), cassette)
    assert not bad_id.committed and "call_id" in bad_id.verdict.reason
    assert no_id.committed


def test_other_claim_keys_are_ignored(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()

    def step(w: Path):
        return _claims(GOOD) + [("note", "unrelated", None)]

    result = run_step(Ledger(), ws, step, CassetteVerifier(_cassette(tmp_path)))
    assert result.committed


def test_cassette_must_be_a_list_of_calls(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "fetch"}), encoding="utf-8")
    try:
        CassetteVerifier(bad)
    except ValueError as exc:
        assert "JSON list" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_booleans_and_numbers_are_not_interchangeable(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    cassette = _cassette(tmp_path, [
        {"name": "set", "arguments": {"enabled": True}, "result": "ok", "error": None},
    ])
    as_number = run_step(Ledger(), ws, _step([("set", {"enabled": 1}, None)]), CassetteVerifier(cassette))
    as_zero = run_step(Ledger(), ws, _step([("set", {"enabled": False}, None)]), CassetteVerifier(cassette))
    as_true = run_step(Ledger(), ws, _step([("set", {"enabled": True}, None)]), CassetteVerifier(cassette))
    assert not as_number.committed
    assert not as_zero.committed
    assert as_true.committed
