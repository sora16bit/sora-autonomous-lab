"""Persistent seed-range leases and bounded worker bookkeeping.

The scheduler is deliberately independent from candidate scoring.  It assigns
disjoint ranges, records fencing tokens, and lets a coordinator reject late
results from an expired worker.  Candidate files remain outside SQLite.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .store import LabStore


@dataclass(frozen=True)
class SeedLease:
    lease_id: str
    namespace: str
    start: int
    end: int
    owner: str
    fencing_token: int
    leased_until: float

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end))


class SeedRangeScheduler:
    """SQLite-backed at-least-once range queue with idempotent seed results."""

    def __init__(self, store: LabStore, *, namespace: str = "default"):
        if not namespace or len(namespace) > 160:
            raise ValueError("namespace must be a bounded non-empty string")
        self.store = store
        self.namespace = namespace
        self.store.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS seed_ranges (
              lease_id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
              start_seed INTEGER NOT NULL, end_seed INTEGER NOT NULL,
              owner TEXT NOT NULL, fencing_token INTEGER NOT NULL,
              leased_until REAL NOT NULL, status TEXT NOT NULL,
              created_at REAL NOT NULL, completed_at REAL,
              UNIQUE(namespace, start_seed, end_seed, fencing_token)
            );
            CREATE INDEX IF NOT EXISTS seed_ranges_ready_idx
              ON seed_ranges(namespace, status, leased_until);
            CREATE TABLE IF NOT EXISTS seed_results (
              result_id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
              seed INTEGER NOT NULL, config_hash TEXT NOT NULL,
              lease_id TEXT NOT NULL, fencing_token INTEGER NOT NULL,
              result_json TEXT NOT NULL, created_at REAL NOT NULL,
              UNIQUE(namespace, seed, config_hash)
            );
            """
        )

    def _next_key(self) -> str:
        return f"scheduler.next_seed.{self.namespace}"

    def _token_key(self) -> str:
        return f"scheduler.fencing_token.{self.namespace}"

    def reap_expired(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)
        cur = self.store.db.execute(
            "UPDATE seed_ranges SET status='queued' WHERE namespace=? AND status='leased' AND leased_until<?",
            (self.namespace, now),
        )
        return int(cur.rowcount)

    def claim(self, owner: str, *, count: int = 16, lease_seconds: float = 900.0,
              start_hint: int | None = None) -> SeedLease:
        if not owner or count < 1 or count > 100_000 or lease_seconds <= 0:
            raise ValueError("invalid seed lease arguments")
        now = time.time()
        with self.store.transaction():
            self.store.db.execute(
                "UPDATE seed_ranges SET status='queued' WHERE namespace=? AND status='leased' AND leased_until<?",
                (self.namespace, now),
            )
            row = self.store.db.execute(
                "SELECT * FROM seed_ranges WHERE namespace=? AND status='queued' ORDER BY start_seed LIMIT 1",
                (self.namespace,),
            ).fetchone()
            if row:
                start, end = int(row["start_seed"]), int(row["end_seed"])
                token = int(row["fencing_token"]) + 1
                lease_id = str(row["lease_id"])
            else:
                raw_next = self.store.get_meta(self._next_key(), start_hint if start_hint is not None else 0)
                start = int(raw_next)
                end = start + count
                token = int(self.store.get_meta(self._token_key(), 0)) + 1
                lease_id = uuid.uuid4().hex
                self.store.db.execute(
                    "INSERT INTO seed_ranges(lease_id,namespace,start_seed,end_seed,owner,fencing_token,leased_until,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (lease_id, self.namespace, start, end, owner, token, now + lease_seconds, "leased", now),
                )
                self.store.set_meta(self._next_key(), end)
            self.store.db.execute(
                "UPDATE seed_ranges SET owner=?,fencing_token=?,leased_until=?,status='leased' WHERE lease_id=?",
                (owner, token, now + lease_seconds, lease_id),
            )
            self.store.set_meta(self._token_key(), token)
        return SeedLease(lease_id, self.namespace, start, end, owner, token, now + lease_seconds)

    def renew(self, lease: SeedLease, *, lease_seconds: float = 900.0) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        cur = self.store.db.execute(
            "UPDATE seed_ranges SET leased_until=? WHERE lease_id=? AND namespace=? AND owner=? AND fencing_token=? AND status='leased'",
            (time.time() + lease_seconds, lease.lease_id, self.namespace, lease.owner, lease.fencing_token),
        )
        return cur.rowcount == 1

    def record_result(self, lease: SeedLease, seed: int, *, config: Any, result: dict[str, Any]) -> bool:
        if seed < lease.start or seed >= lease.end:
            raise ValueError("seed is outside the claimed range")
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
        try:
            self.store.db.execute(
                "INSERT INTO seed_results(result_id,namespace,seed,config_hash,lease_id,fencing_token,result_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, self.namespace, int(seed), config_hash, lease.lease_id,
                 lease.fencing_token, json.dumps(result, ensure_ascii=False, default=str), time.time()),
            )
            return True
        except sqlite3.IntegrityError:
            # Replays after a worker retry are expected and idempotent.
            return False

    def complete(self, lease: SeedLease) -> bool:
        cur = self.store.db.execute(
            "UPDATE seed_ranges SET status='completed',completed_at=? WHERE lease_id=? AND namespace=? AND owner=? AND fencing_token=? AND status='leased'",
            (time.time(), lease.lease_id, self.namespace, lease.owner, lease.fencing_token),
        )
        return cur.rowcount == 1

    def fail(self, lease: SeedLease, *, requeue: bool = True) -> bool:
        status = "queued" if requeue else "failed"
        cur = self.store.db.execute(
            "UPDATE seed_ranges SET status=? WHERE lease_id=? AND namespace=? AND owner=? AND fencing_token=? AND status='leased'",
            (status, lease.lease_id, self.namespace, lease.owner, lease.fencing_token),
        )
        return cur.rowcount == 1

    def summary(self) -> dict[str, int]:
        rows = self.store.db.execute(
            "SELECT status,COUNT(*) AS n FROM seed_ranges WHERE namespace=? GROUP BY status",
            (self.namespace,),
        )
        return {str(row["status"]): int(row["n"]) for row in rows}
