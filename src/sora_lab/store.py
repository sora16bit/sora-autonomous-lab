from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS lab_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, started_at REAL NOT NULL, stopped_at REAL,
  budget_seconds INTEGER NOT NULL, status TEXT NOT NULL,
  accumulated_seconds REAL NOT NULL DEFAULT 0, last_heartbeat REAL,
  pid INTEGER, stop_requested INTEGER NOT NULL DEFAULT 0,
  deadline_at REAL
);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY, parent_id TEXT, created_at REAL NOT NULL,
  started_at REAL, finished_at REAL, state TEXT NOT NULL, hypothesis TEXT NOT NULL,
  config_json TEXT NOT NULL, result_json TEXT, error TEXT,
  patch_hash TEXT, code_commit TEXT, model_digest TEXT
);
CREATE TABLE IF NOT EXISTS advice (
  advice_id TEXT PRIMARY KEY, window_id TEXT NOT NULL, provider TEXT NOT NULL,
  model TEXT NOT NULL, created_at REAL NOT NULL, input_tokens INTEGER,
  output_tokens INTEGER, status TEXT NOT NULL, request_json TEXT NOT NULL,
  response_json TEXT, UNIQUE(window_id, provider, model)
);
CREATE TABLE IF NOT EXISTS intake_candidates (
  candidate_id TEXT PRIMARY KEY, source_repo TEXT NOT NULL, commit_sha TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE, state TEXT NOT NULL, record_json TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,
  kind TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_leases (
  experiment_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL,
  leased_until REAL NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS experiments_state_idx ON experiments(state);
CREATE INDEX IF NOT EXISTS events_kind_idx ON events(kind);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class LabStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        # Existing development databases predate the process-control columns.
        for sql in (
            "ALTER TABLE sessions ADD COLUMN pid INTEGER",
            "ALTER TABLE sessions ADD COLUMN stop_requested INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE sessions ADD COLUMN deadline_at REAL",
        ):
            try:
                self.db.execute(sql)
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise

    def close(self) -> None:
        self.db.close()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM lab_meta WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def set_meta(self, key: str, value: Any) -> None:
        self.db.execute("INSERT INTO lab_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, _json(value)))

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
        except Exception:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO events(created_at,kind,payload_json) VALUES(?,?,?)",
            (time.time(), kind, _json(payload)),
        )

    def latest_event(self, kind: str) -> dict[str, Any] | None:
        """Return the newest event payload of a kind without exposing raw rows."""
        row = self.db.execute(
            "SELECT payload_json FROM events WHERE kind=? ORDER BY event_id DESC LIMIT 1",
            (kind,),
        ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def start_session(self, budget_seconds: int) -> str:
        if budget_seconds < 0:
            raise ValueError("budget_seconds must be non-negative")
        sid = uuid.uuid4().hex
        now = time.time()
        with self.transaction():
            active = self.db.execute("SELECT * FROM sessions WHERE status='running' LIMIT 1").fetchone()
            if active:
                # A killed coordinator can leave an old row marked running.
                # Recover only when its PID is definitively gone and the
                # heartbeat is stale; EPERM/unknown remains a hard block.
                pid = active["pid"]
                age = now - float(active["last_heartbeat"] or active["started_at"])
                process_gone = pid is None
                if pid is not None:
                    try:
                        os.kill(int(pid), 0)
                        process_gone = False
                    except ProcessLookupError:
                        process_gone = True
                    except PermissionError:
                        process_gone = False
                elif age <= 30.0:
                    process_gone = False
                if process_gone and age > 30.0:
                    self.db.execute("UPDATE sessions SET status='interrupted',stopped_at=?,pid=NULL WHERE session_id=?",
                                    (now, active["session_id"]))
                    self.event("session_orphan_recovered", {"session_id": active["session_id"], "heartbeat_age": age})
                else:
                    raise RuntimeError(f"lab session already running: {active['session_id']}")
            self.db.execute(
                "INSERT INTO sessions(session_id,started_at,budget_seconds,status,last_heartbeat,pid,stop_requested,deadline_at) VALUES(?,?,?,?,?,?,0,?)",
                (sid, now, budget_seconds, "running", now, None,
                 now + budget_seconds if budget_seconds > 0 else None),
            )
        self.event("session_started", {"session_id": sid, "budget_seconds": budget_seconds})
        return sid

    def heartbeat(self, session_id: str) -> float:
        row = self.db.execute("SELECT started_at,accumulated_seconds,budget_seconds,status,last_heartbeat FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            raise KeyError(session_id)
        if row["status"] != "running":
            return float(row["accumulated_seconds"])
        now = time.time()
        previous = float(row["last_heartbeat"] or row["started_at"])
        elapsed = max(0.0, now - previous)
        raw_accumulated = float(row["accumulated_seconds"]) + elapsed
        accumulated = (min(float(row["budget_seconds"]), raw_accumulated)
                       if int(row["budget_seconds"]) > 0 else raw_accumulated)
        self.db.execute("UPDATE sessions SET accumulated_seconds=?,last_heartbeat=? WHERE session_id=?", (accumulated, now, session_id))
        return accumulated

    def set_pid(self, session_id: str, pid: int | None) -> None:
        self.db.execute("UPDATE sessions SET pid=? WHERE session_id=?", (pid, session_id))

    def request_stop(self, session_id: str) -> None:
        self.db.execute("UPDATE sessions SET stop_requested=1 WHERE session_id=? AND status='running'", (session_id,))
        self.event("session_stop_requested", {"session_id": session_id})

    def stop_requested(self, session_id: str) -> bool:
        row = self.db.execute("SELECT stop_requested FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return bool(row and row["stop_requested"])

    def stop_session(self, session_id: str, status: str = "stopped") -> None:
        if status not in {"stopped", "completed", "interrupted"}:
            raise ValueError(status)
        accumulated = self.heartbeat(session_id)
        self.db.execute("UPDATE sessions SET stopped_at=?,status=?,accumulated_seconds=?,pid=NULL WHERE session_id=?", (time.time(), status, accumulated, session_id))
        self.event("session_stopped", {"session_id": session_id, "status": status, "accumulated_seconds": accumulated})

    def session(self, session_id: str | None = None) -> dict[str, Any] | None:
        if session_id is None:
            row = self.db.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        else:
            row = self.db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def add_experiment(self, *, hypothesis: str, config: dict[str, Any], parent_id: str | None = None, patch_hash: str | None = None, code_commit: str | None = None, model_digest: str | None = None) -> str:
        if not hypothesis.strip():
            raise ValueError("hypothesis is required")
        eid = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO experiments(experiment_id,parent_id,created_at,state,hypothesis,config_json,patch_hash,code_commit,model_digest) VALUES(?,?,?,?,?,?,?,?,?)",
            (eid, parent_id, time.time(), "queued", hypothesis, _json(config), patch_hash, code_commit, model_digest),
        )
        self.event("experiment_queued", {"experiment_id": eid, "hypothesis": hypothesis})
        return eid

    def update_experiment(self, experiment_id: str, *, state: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        allowed = {"queued", "running", "generated", "scored", "audited", "retained", "rejected", "failed", "interrupted", "audit_unavailable"}
        if state not in allowed:
            raise ValueError(state)
        now = time.time()
        started = now if state == "running" else None
        finished = now if state in {"retained", "rejected", "failed", "interrupted", "audit_unavailable"} else None
        self.db.execute(
            "UPDATE experiments SET state=?,started_at=COALESCE(started_at,?),finished_at=COALESCE(?,finished_at),result_json=?,error=? WHERE experiment_id=?",
            (state, started, finished, _json(result) if result is not None else None, error, experiment_id),
        )
        self.event("experiment_state", {"experiment_id": experiment_id, "state": state, "error": error})

    def claim_experiment(self, worker_id: str, *, lease_seconds: float = 600.0) -> str | None:
        """Atomically claim one queued or expired experiment for a worker."""
        if not worker_id or lease_seconds <= 0:
            raise ValueError("worker_id and positive lease_seconds are required")
        now = time.time()
        with self.transaction():
            self.db.execute("DELETE FROM experiment_leases WHERE leased_until < ?", (now,))
            row = self.db.execute(
                """SELECT e.experiment_id FROM experiments e
                   LEFT JOIN experiment_leases l ON l.experiment_id=e.experiment_id
                   WHERE e.state='queued' AND l.experiment_id IS NULL
                   ORDER BY e.created_at LIMIT 1"""
            ).fetchone()
            if not row:
                return None
            eid = str(row["experiment_id"])
            self.db.execute("INSERT INTO experiment_leases(experiment_id,worker_id,leased_until,created_at) VALUES(?,?,?,?)",
                            (eid, worker_id, now + lease_seconds, now))
            self.db.execute("UPDATE experiments SET state='running',started_at=COALESCE(started_at,?) WHERE experiment_id=? AND state='queued'",
                            (now, eid))
        self.event("experiment_claimed", {"experiment_id": eid, "worker_id": worker_id, "lease_seconds": lease_seconds})
        return eid

    def renew_experiment_lease(self, experiment_id: str, worker_id: str, *, lease_seconds: float = 600.0) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        cur = self.db.execute("UPDATE experiment_leases SET leased_until=? WHERE experiment_id=? AND worker_id=?",
                              (time.time() + lease_seconds, experiment_id, worker_id))
        return cur.rowcount == 1

    def release_experiment(self, experiment_id: str, worker_id: str) -> bool:
        cur = self.db.execute("DELETE FROM experiment_leases WHERE experiment_id=? AND worker_id=?", (experiment_id, worker_id))
        return cur.rowcount == 1

    def add_advice(self, *, window_id: str, provider: str, model: str, request: dict[str, Any], response: dict[str, Any] | None, status: str, input_tokens: int | None = None, output_tokens: int | None = None) -> str:
        aid = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO advice(advice_id,window_id,provider,model,created_at,input_tokens,output_tokens,status,request_json,response_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (aid, window_id, provider, model, time.time(), input_tokens, output_tokens, status, _json(request), _json(response) if response is not None else None),
        )
        return aid

    def reserve_advice(self, *, window_id: str, provider: str, model: str, request: dict[str, Any]) -> bool:
        """Atomically reserve a consultation window before sending it."""
        try:
            self.db.execute(
                "INSERT INTO advice(advice_id,window_id,provider,model,created_at,status,request_json) VALUES(?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, window_id, provider, model, time.time(), "reserved", _json(request)),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def finish_advice(self, *, window_id: str, provider: str, model: str, response: dict[str, Any] | None,
                      status: str, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        self.db.execute(
            "UPDATE advice SET status=?,response_json=?,input_tokens=?,output_tokens=? WHERE window_id=? AND provider=? AND model=?",
            (status, _json(response) if response is not None else None, input_tokens, output_tokens, window_id, provider, model),
        )

    def advice_exists(self, window_id: str, provider: str, model: str) -> bool:
        return self.db.execute("SELECT 1 FROM advice WHERE window_id=? AND provider=? AND model=?", (window_id, provider, model)).fetchone() is not None

    def add_intake_candidate(self, record: dict[str, Any]) -> bool:
        now = time.time()
        try:
            self.db.execute(
                "INSERT INTO intake_candidates(candidate_id,source_repo,commit_sha,content_hash,state,record_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (record["candidate_id"], record["source_repo"], record["commit"], record["content_hash"], record.get("state", "discovered"), _json(record), now, now),
            )
        except sqlite3.IntegrityError:
            return False
        self.event("intake_candidate", {"candidate_id": record["candidate_id"], "source_repo": record["source_repo"], "commit": record["commit"]})
        return True

    def intake_candidates(self) -> list[dict[str, Any]]:
        return [json.loads(row["record_json"]) for row in self.db.execute("SELECT record_json FROM intake_candidates ORDER BY created_at")]

    def update_intake_candidate(self, candidate_id: str, *, state: str, details: dict[str, Any] | None = None) -> None:
        row = self.db.execute("SELECT record_json FROM intake_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if not row:
            raise KeyError(candidate_id)
        record = json.loads(row["record_json"])
        if details:
            record.update(details)
        record["state"] = state
        self.db.execute("UPDATE intake_candidates SET state=?,record_json=?,updated_at=? WHERE candidate_id=?",
                        (state, _json(record), time.time(), candidate_id))
        self.event("intake_state", {"candidate_id": candidate_id, "state": state})

    def next_seed(self, default: int = 0) -> int:
        maximum = default - 1
        for row in self.db.execute("SELECT config_json FROM experiments"):
            try:
                config = json.loads(row["config_json"])
                seeds = config.get("seeds", [])
                if isinstance(seeds, str):
                    seeds = [int(x.strip()) for x in seeds.split(",") if x.strip()]
                maximum = max(maximum, *(int(x) for x in seeds))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return maximum + 1

    def summary(self) -> dict[str, Any]:
        counts = {row["state"]: row["n"] for row in self.db.execute("SELECT state,COUNT(*) n FROM experiments GROUP BY state")}
        last = self.session()
        intake = {row["state"]: row["n"] for row in self.db.execute("SELECT state,COUNT(*) n FROM intake_candidates GROUP BY state")}
        best = None
        best_path = self.path.parent / "best_candidate.json"
        if best_path.exists():
            try:
                best = json.loads(best_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                best = {"status": "unreadable"}
        seed_ranges: dict[str, int] = {}
        try:
            seed_ranges = {row["status"]: row["n"] for row in self.db.execute(
                "SELECT status,COUNT(*) n FROM seed_ranges GROUP BY status")}
        except sqlite3.OperationalError:
            # Older databases receive the scheduler tables lazily on the next
            # run; status must remain usable before that migration.
            pass
        llm_kinds = {str(row["kind"]): int(row["n"]) for row in self.db.execute(
            "SELECT kind,COUNT(*) n FROM events WHERE kind LIKE 'local_llm_%' GROUP BY kind")}
        role_valid: dict[str, dict[str, int]] = {}
        try:
            for row in self.db.execute(
                "SELECT payload_json FROM events WHERE kind='local_llm_role_finished' ORDER BY event_id DESC LIMIT 10000"):
                payload = json.loads(row["payload_json"])
                role = str(payload.get("role", "unknown"))
                counts_for_role = role_valid.setdefault(role, {"calls": 0, "valid": 0})
                counts_for_role["calls"] += 1
                counts_for_role["valid"] += int(bool(payload.get("valid")))
        except (TypeError, json.JSONDecodeError):
            pass
        legacy_review = self.latest_event("local_llm_review") or {}
        legacy_roles = {str(item.get("role", "unknown")): bool(item.get("valid"))
                        for item in legacy_review.get("replies", []) if isinstance(item, dict)}
        local_llm = {
            "enabled": bool(self.get_meta("local_llm_enabled", False)),
            "pending_attack_proposals": len(self.get_meta("pending_attack_specs", []) or []),
            "pending_trial": bool(self.get_meta("pending_trial")),
            "event_counts": llm_kinds,
            "role_validity_last_10000": role_valid,
            "latest_recorded_review": {"status": legacy_review.get("status"),
                                       "role_validity": legacy_roles,
                                       "accepted_parameter_changes": bool(legacy_review.get("accepted_changes")),
                                       "accepted_attack_proposal": bool(legacy_review.get("accepted_attack_spec"))},
            "latest_health": self.latest_event("local_llm_health"),
            "latest_review": self.latest_event("local_llm_review_finished"),
            "runner_configuration": self.latest_event("runner_configuration"),
        }
        process = {"pid": last.get("pid") if last else None, "alive": None,
                   "heartbeat_age_seconds": None, "health": "not_running"}
        if last and last.get("status") == "running":
            pid = last.get("pid")
            heartbeat = last.get("last_heartbeat") or last.get("started_at")
            process["heartbeat_age_seconds"] = max(0.0, time.time() - float(heartbeat))
            if pid is None:
                process["alive"] = False
            else:
                try:
                    os.kill(int(pid), 0)
                    process["alive"] = True
                except ProcessLookupError:
                    process["alive"] = False
                except PermissionError:
                    process["alive"] = None
            age = process["heartbeat_age_seconds"]
            if process["alive"] is None:
                process["health"] = "fresh_heartbeat_pid_unverified" if age <= 360 else "stale_heartbeat_pid_unverified"
            else:
                process["health"] = ("healthy" if process["alive"] and age <= 360
                                      else "heartbeat_stale" if process["alive"]
                                      else "orphan_suspected" if age > 30
                                      else "starting")
        return {"experiments": counts, "intake": intake, "best_candidate": best,
                "last_session": last, "seed_ranges": seed_ranges,
                "pending_trial": bool(self.get_meta("pending_trial")),
                "local_llm": local_llm,
                "runner_process": process,
                "attack_generation": self.get_meta("attack_generation", 0),
                # Keep the latest audit visible in the lightweight status
                # command.  It is deliberately a compact aggregate; raw
                # attack rows remain in the event log/run root.
                "last_audit": self.get_meta("last_audit"),
                "best_confirmed_proxy": self.get_meta("best_confirmed_proxy")}


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()
