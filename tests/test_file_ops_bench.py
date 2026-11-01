from benchmarks.file_ops.run import main
from benchmarks.file_ops.tasks import build_tasks


def test_twenty_tasks_have_unique_ids() -> None:
    tasks = build_tasks()
    assert len(tasks) == 20
    assert len({t.id for t in tasks}) == 20


def test_harness_self_test_metrics(tmp_path, capsys) -> None:
    out = tmp_path / "report.json"
    assert main([str(out)]) == 0
    import json
    report = json.loads(out.read_text())

    assert report["oracle"]["ledger"] == {"tasks": 20, "committed": 20, "rejected": 0, "rollbacks": 0}
    assert report["skip"]["ledger"]["committed"] == 0
    assert report["skip"]["ledger"]["rejected"] == 20
    assert report["skip"]["baseline"]["accepted_without_check"] == 20
    # Single-output tasks get no change from the partial agent; rename and concat are cut short.
    assert report["partial"]["ledger"]["committed"] == 0
    assert report["partial"]["ledger"]["rollbacks"] == 20


def test_dangling_symlink_at_deleted_path_is_rejected(tmp_path) -> None:
    import os
    from agent_ledger import ExpectedStateVerifier
    from agent_ledger.ledger import Claim, ClaimStatus
    ws = tmp_path / "ws"
    ws.mkdir()
    os.symlink("missing.txt", ws / "draft.txt")
    (ws / "final.txt").write_text("done")
    verdict = ExpectedStateVerifier({"draft.txt": None, "final.txt": "done"}).check(ws, [])
    assert not verdict.ok
    assert "should be deleted: draft.txt" in verdict.reason
