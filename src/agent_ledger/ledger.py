"""SQLite-backed ledger of task claims. Claims live outside any model context."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ClaimStatus(str, Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Claim:
    id: int
    key: str
    value: Any
    status: ClaimStatus
    provenance: dict[str, Any]
    reason: str | None
    created_at: float


class Ledger:
    """Stores claims about task state. Only COMMITTED claims count as done."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                status TEXT NOT NULL,
                provenance TEXT NOT NULL,
                reason TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def add_claim(self, key: str, value: Any, provenance: dict[str, Any] | None = None) -> int:
        cur = self._db.execute(
            "INSERT INTO claims (key, value, status, provenance, reason, created_at) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (key, json.dumps(value), ClaimStatus.PENDING.value,
             json.dumps(provenance or {}), time.time()),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def set_status(self, claim_id: int, status: ClaimStatus, reason: str | None = None) -> None:
        self._db.execute(
            "UPDATE claims SET status = ?, reason = ? WHERE id = ?",
            (status.value, reason, claim_id),
        )
        self._db.commit()

    def get(self, claim_id: int) -> Claim:
        row = self._db.execute(
            "SELECT id, key, value, status, provenance, reason, created_at "
            "FROM claims WHERE id = ?",
            (claim_id,),
        ).fetchone()
        if row is None:
            raise KeyError(claim_id)
        return _row_to_claim(row)

    def committed(self, key: str | None = None) -> list[Claim]:
        sql = ("SELECT id, key, value, status, provenance, reason, created_at "
               "FROM claims WHERE status = ?")
        args: list[Any] = [ClaimStatus.COMMITTED.value]
        if key is not None:
            sql += " AND key = ?"
            args.append(key)
        sql += " ORDER BY id"
        return [_row_to_claim(r) for r in self._db.execute(sql, args)]

    def is_done(self, key: str) -> bool:
        return bool(self.committed(key))

    def rejected_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM claims WHERE status = ?",
            (ClaimStatus.REJECTED.value,),
        ).fetchone()
        return int(row[0])


def _row_to_claim(row: tuple) -> Claim:
    cid, key, value, status, provenance, reason, created_at = row
    return Claim(
        id=cid,
        key=key,
        value=json.loads(value),
        status=ClaimStatus(status),
        provenance=json.loads(provenance),
        reason=reason,
        created_at=created_at,
    )
