from __future__ import annotations

import json
import csv
import os
import subprocess
import sys
import time
import signal
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import field, replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .store import LabStore
from .intake import changed_paths, diff_excerpt, load_specs, make_candidate_record, update_mirror
from .agents import ask_ollama, parse_attack_proposal, validate_proposal, validate_review
from .advisor import AdvicePolicy, build_packet, consult_command, should_consult
from .jev import JevController


_ACTIVE_PROCESSES: dict[str, subprocess.Popen[str]] = {}


@dataclass(frozen=True)
class RunConfig:
    dist: Path
    kit_dir: Path
    run_root: Path
    seeds: str = ""
    seed_start: int = 0
    batch_size: int = 4
    outcome_draws: int = 100
    outcome_step: float = 0.10
    hours: float = 1.0
    poll_seconds: float = 2.0
    ollama_model: str = "qwen3.6:35b"
    intake_specs: Path | None = None
    mirror_root: Path | None = None
    intake_interval_seconds: float = 3600.0
    local_llm: bool = False
    ollama_host: str = "http://127.0.0.1:11434"
    search_params: dict[str, Any] = field(default_factory=dict)
    cloud_advice: bool = False
    cloud_interval_seconds: float = 5 * 60 * 60
    audit_interval_batches: int = 20
    audit_permutations: int = 200
    attack_evolution_interval_batches: int = 20
    attack_evolution_permutations: int = 10
    attack_evolution_population_size: int = 2
    attack_evolution_calibration_trials: int = 32
    parallel_workers: int = 1
    seed_chunk_size: int = 4
    local_llm_interval_batches: int = 50
    jev: bool = False


def _before_deadline(deadline: float | None) -> bool:
    """Return whether a run may continue; None is the explicit forever mode."""
    return deadline is None or time.time() < deadline


def _ollama_models(host: str) -> list[str]:
    request = urllib.request.Request(host.rstrip("/") + "/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama is unavailable: {type(exc).__name__}: {exc}") from exc
    return [str(item.get("name")) for item in payload.get("models", []) if item.get("name")]


def preflight(config: RunConfig, *, require_ollama: bool = False, ollama_host: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    problems: list[str] = []
    if not config.dist.is_dir():
        problems.append(f"dist directory missing: {config.dist}")
    if not config.kit_dir.is_dir():
        problems.append(f"kit directory missing: {config.kit_dir}")
    required_dist = ("B.csv", "B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json")
    if config.dist.is_dir():
        problems.extend(f"dist file missing: {name}" for name in required_dist if not (config.dist / name).exists())
    if config.kit_dir.is_dir() and not (config.kit_dir / "starter").exists():
        problems.append("kit starter directory missing")
    if config.hours < 0:
        problems.append("hours must be non-negative (0 means forever)")
    if config.batch_size <= 0:
        problems.append("batch_size must be positive")
    if config.audit_interval_batches < 0:
        problems.append("audit_interval_batches must be non-negative")
    if config.audit_permutations <= 0:
        problems.append("audit_permutations must be positive")
    if config.attack_evolution_interval_batches < 0:
        problems.append("attack_evolution_interval_batches must be non-negative")
    if config.attack_evolution_permutations <= 0:
        problems.append("attack_evolution_permutations must be positive")
    if not 1 <= config.attack_evolution_population_size <= 64:
        problems.append("attack_evolution_population_size must be in [1,64]")
    if not 1 <= config.attack_evolution_calibration_trials <= 128:
        problems.append("attack_evolution_calibration_trials must be in [1,128]")
    if config.parallel_workers < 1:
        problems.append("parallel_workers must be positive")
    if config.seed_chunk_size < 1:
        problems.append("seed_chunk_size must be positive")
    if config.local_llm_interval_batches <= 0:
        problems.append("local_llm_interval_batches must be positive")
    if config.intake_specs is not None and not config.intake_specs.exists():
        problems.append(f"intake specs missing: {config.intake_specs}")
    if config.intake_specs is not None and config.mirror_root is None:
        problems.append("mirror_root is required when intake_specs is set")
    models: list[str] = []
    ollama_error = None
    try:
        models = _ollama_models(ollama_host)
    except RuntimeError as exc:
        ollama_error = str(exc)
        if require_ollama:
            problems.append(ollama_error)
    if require_ollama and config.ollama_model not in models:
        problems.append(f"required Ollama model missing: {config.ollama_model}")
    return {"ok": not problems, "problems": problems, "dist": str(config.dist), "kit_dir": str(config.kit_dir), "ollama_models": models, "ollama_error": ollama_error, "python": sys.executable}


def _seed_batches(config: RunConfig):
    explicit = [int(x.strip()) for x in config.seeds.split(",") if x.strip()]
    if explicit:
        first = explicit
        yield first
        next_seed = max(first) + 1
    else:
        next_seed = config.seed_start
    while True:
        batch = list(range(next_seed, next_seed + config.batch_size))
        yield batch
        next_seed += config.batch_size


def _terminate_process_group(proc: subprocess.Popen[str], grace: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=grace)
    except ProcessLookupError:
        pass


def _run_batch(store: LabStore, session_id: str, command: list[str], output_dir: Path,
               deadline: float | None, poll_seconds: float, *, process_key: str | None = None,
               set_session_pid: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path = output_dir / "stdout.log", output_dir / "stderr.log"
    started = time.monotonic()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(command, cwd=str(Path(__file__).parents[2]), stdout=stdout, stderr=stderr,
                                text=True, env=env, start_new_session=True)
        key = process_key or session_id
        _ACTIVE_PROCESSES[key] = proc
        if set_session_pid:
            store.set_pid(session_id, proc.pid)
        timed_out = False
        stop = False
        while proc.poll() is None:
            store.heartbeat(session_id)
            stop = store.stop_requested(session_id)
            if stop or not _before_deadline(deadline):
                timed_out = not stop and deadline is not None
                _terminate_process_group(proc)
                break
            remaining = max(0.1, deadline - time.time()) if deadline is not None else max(0.1, poll_seconds)
            time.sleep(min(max(0.1, poll_seconds), remaining))
        returncode = proc.wait()
    if set_session_pid:
        store.set_pid(session_id, None)
    _ACTIVE_PROCESSES.pop(process_key or session_id, None)
    return {"returncode": returncode, "elapsed_seconds": time.monotonic() - started,
            "timeout": timed_out, "stop_requested": stop,
            "stdout": str(stdout_path), "stderr": str(stderr_path), "command": command}


def _merge_parallel_scores(output_dir: Path, worker_dirs: list[Path]) -> None:
    """Create one deterministic score table while retaining worker artifacts."""
    target = output_dir / "candidates"
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for worker_dir in worker_dirs:
        score_file = worker_dir / "candidates" / "candidate_scores.csv"
        if not score_file.exists():
            continue
        with score_file.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(dict.fromkeys([*fieldnames, *(reader.fieldnames or [])]))
            rows.extend(dict(row) for row in reader)
    if not fieldnames:
        return
    temporary = target / "candidate_scores.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target / "candidate_scores.csv")


def _run_parallel_seed_batch(store: LabStore, session_id: str, command_prefix: list[str],
                             seeds: list[int], output_dir: Path, *, workers: int,
                             chunk_size: int, deadline: float | None, poll_seconds: float) -> dict[str, Any]:
    """Run independent seed chunks concurrently and merge only their summaries."""
    import time as _time
    chunks = [seeds[i:i + chunk_size] for i in range(0, len(seeds), chunk_size)]
    started = _time.monotonic()
    worker_dirs = [output_dir / f"worker_{index:03d}" for index in range(len(chunks))]
    runs: list[dict[str, Any]] = []
    store.set_pid(session_id, os.getpid())
    with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as executor:
        futures = {}
        for index, (chunk, worker_dir) in enumerate(zip(chunks, worker_dirs)):
            command = [*command_prefix, "--out-dir", str(worker_dir / "candidates"),
                       "--seeds", ",".join(map(str, chunk))]
            def run_worker(index=index, command=command, worker_dir=worker_dir):
                # sqlite connections are thread-affine; each subprocess monitor
                # gets its own connection while WAL handles the short writes.
                local_store = LabStore(store.path)
                try:
                    return _run_batch(local_store, session_id, command, worker_dir,
                                      deadline, poll_seconds,
                                      process_key=f"{session_id}:{index}",
                                      set_session_pid=False)
                finally:
                    local_store.close()
            futures[executor.submit(run_worker)] = index
        for future in as_completed(futures):
            index = futures[future]
            try:
                runs.append({"worker": index, **future.result()})
            except Exception as exc:  # noqa: BLE001
                runs.append({"worker": index, "returncode": -1,
                             "error": f"{type(exc).__name__}: {exc}"})
    _merge_parallel_scores(output_dir, worker_dirs)
    return {"returncode": 0 if runs and all(item.get("returncode") == 0 for item in runs) else 1,
            "elapsed_seconds": _time.monotonic() - started,
            "timeout": any(item.get("timeout", False) for item in runs),
            "stop_requested": any(item.get("stop_requested", False) for item in runs),
            "workers": runs, "output_dir": str(output_dir)}


def _local_review(store: LabStore, *, batch: int, seeds: list[int], model: str, host: str,
                  output_dir: Path, current_params: dict[str, Any],
                  session_id: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": 0, "best_utility": None, "median_utility": None}
    score_file = output_dir / "candidates" / "candidate_scores.csv"
    if score_file.exists():
        try:
            with score_file.open(newline="", encoding="utf-8") as fh:
                values = [float(row["utility"]) for row in csv.DictReader(fh)
                          if row.get("utility") not in (None, "") and row["utility"] != "nan"]
            if values:
                summary = {"count": len(values), "best_utility": max(values),
                           "median_utility": float(np.median(values))}
        except (OSError, ValueError, KeyError):
            summary["read_error"] = True
    last_audit = store.get_meta("last_audit")
    active_attacks = [
        {"attack_id": str(attack_id), "family": str(attack_id).split("_", 2)[1] if "_" in str(attack_id) else "unknown"}
        for attack_id in (last_audit.get("attacks", {}) if isinstance(last_audit, dict) else {})
        if attack_id != "_aia"
    ]
    # Prefer the family recorded in attack specs where available; ids alone are
    # retained as the constrained parent-choice set for the LLM proposal.
    family_by_prefix = {"mia_distance": "distance", "mia_blocked": "blocked_distance",
                        "mia_marginal": "marginal", "mia_shadow": "shadow"}
    for item in active_attacks:
        item["family"] = next((family for prefix, family in family_by_prefix.items()
                               if prefix in item["attack_id"]), item["family"])
    archive_path = output_dir.parent.parent / "attack-archive.json"
    if archive_path.exists():
        try:
            from .attack_archive import AttackArchive
            active_attacks = [{"attack_id": spec.attack_id, "family": spec.family}
                              for spec in AttackArchive.load(archive_path).active_specs()]
        except Exception:  # noqa: BLE001 - the audit result remains the fallback source
            pass
    report = {"batch": batch, "seeds": seeds, "score_summary": summary,
              "current_params": current_params, "last_audit": last_audit,
              "active_attacks": active_attacks,
              "instruction": "候補集計と直近の自己攻撃監査を見て次の許可済み実験を1つ提案。"
                             "監査リスクを下げる変更を優先し、秘密データ・行値・shellは要求しない。"}
    replies = []
    context = report
    for role in ("improver", "attacker", "reviewer"):
        role_started = time.monotonic()
        store.event("local_llm_role_started", {"batch": batch, "role": role, "model": model})
        if session_id:
            store.heartbeat(session_id)
        try:
            reply = ask_ollama(role=role, report=context, model=model, host=host, timeout=180)
            if role == "improver":
                valid, reasons = validate_proposal(reply)
                parsed_attack = None
            elif role == "attacker":
                parent_ids = {item["attack_id"] for item in active_attacks if item.get("family") == (reply.parsed or {}).get("family")}
                parsed_attack, reasons = parse_attack_proposal(reply, allowed_parent_ids=parent_ids)
                valid = parsed_attack is not None
            else:
                valid, reasons = validate_review(reply)
                parsed_attack = None
            replies.append({"role": role, "model": reply.model, "parsed": reply.parsed,
                            "valid": valid, "reasons": reasons,
                            "latency_ms": round((time.monotonic() - role_started) * 1000),
                            **({"attack_spec": parsed_attack} if parsed_attack else {})})
            if role == "improver":
                context = {**report, "improver_proposal": reply.parsed}
            elif role == "attacker":
                context = {**report, "improver_proposal": replies[0].get("parsed"),
                           "attacker_proposal": parsed_attack or {"invalid": reasons}}
        except Exception as exc:  # noqa: BLE001
            replies.append({"role": role, "valid": False,
                            "latency_ms": round((time.monotonic() - role_started) * 1000),
                            "reasons": [f"unavailable:{type(exc).__name__}"]})
        finally:
            store.event("local_llm_role_finished", {"batch": batch, "role": role,
                                                     "valid": bool(replies[-1].get("valid")),
                                                     "latency_ms": replies[-1].get("latency_ms"),
                                                     "reasons": replies[-1].get("reasons", [])})
            if session_id:
                store.heartbeat(session_id)
    changes: dict[str, Any] = {}
    reviewer = replies[-1] if replies else {}
    improver = replies[0] if replies else {}
    decision = str((reviewer.get("parsed") or {}).get("decision", "")).lower()
    reviewer_accepts = reviewer.get("valid") and decision in {"accept", "accepted", "approve", "approved"}
    if improver.get("valid") and reviewer_accepts:
        raw = (improver.get("parsed") or {}).get("changes", {})
        if isinstance(raw, dict) and raw:
            changes = dict(raw)
    attacker = next((item for item in replies if item.get("role") == "attacker"), {})
    accepted_attack = attacker.get("attack_spec") if attacker.get("valid") and reviewer_accepts else None
    pending_attacks = store.get_meta("pending_attack_specs", []) or []
    if accepted_attack:
        known = {item.get("semantic_hash") for item in pending_attacks}
        from .attack_registry import AttackSpec
        proposed = AttackSpec.from_dict(accepted_attack)
        active_hashes = set()
        archive = output_dir.parent.parent / "attack-archive.json"
        if archive.exists():
            try:
                from .attack_archive import AttackArchive
                active_hashes = {item.semantic_hash for item in AttackArchive.load(archive).active_specs()}
            except Exception:  # noqa: BLE001
                pass
        if proposed.semantic_hash not in known and proposed.semantic_hash not in active_hashes:
            accepted_attack["semantic_hash"] = proposed.semantic_hash
            pending_attacks.append(accepted_attack)
            store.set_meta("pending_attack_specs", pending_attacks)
            store.event("local_llm_attack_queued", {"batch": batch, "attack_id": proposed.attack_id,
                                                     "family": proposed.family, "parent": proposed.parents[0]})
        else:
            accepted_attack = None
            store.event("local_llm_attack_rejected", {"batch": batch, "reason": "duplicate_semantics"})
    payload = {"batch": batch, "replies": replies, "accepted_changes": changes,
               "accepted_attack_spec": accepted_attack,
               "status": "completed" if any(item.get("valid") for item in replies) else "no_valid_role_reply"}
    store.event("local_llm_review", payload)
    return payload


def _apply_changes(params: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    next_params = dict(params)
    for key, value in changes.items():
        if key not in next_params:
            continue
        numeric = float(value)
        next_params[key] = int(numeric) if key == "outcome_draws" else numeric
    return next_params


def _best_score(output_dir: Path) -> float | None:
    path = output_dir / "candidates" / "candidate_scores.csv"
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            values = [float(row["utility"]) for row in csv.DictReader(fh)
                      if row.get("validation_ok", "").lower() == "true" and row.get("utility") not in (None, "", "nan")]
        return max(values) if values else None
    except (OSError, ValueError, KeyError):
        return None


def _score_summary(output_dir: Path) -> dict[str, float | int | None]:
    path = output_dir / "candidates" / "candidate_scores.csv"
    values: list[float] = []
    if path.exists():
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("validation_ok", "").lower() != "true":
                        continue
                    raw = row.get("utility")
                    if raw in (None, "", "nan"):
                        continue
                    value = float(raw)
                    if np.isfinite(value):
                        values.append(value)
        except (OSError, ValueError, KeyError, csv.Error):
            return {"count": 0, "best": None, "median": None}
    if not values:
        return {"count": 0, "best": None, "median": None}
    values.sort()
    return {"count": len(values), "best": values[-1], "median": float(np.median(values))}


TRIAL_MEDIAN_TOLERANCE = 0.002


def _trial_gates(*, baseline: dict[str, Any], trial: dict[str, Any],
                 baseline_anon: float | None, trial_anon: float | None,
                 repair: bool) -> dict[str, bool]:
    baseline_best, baseline_median = baseline.get("best"), baseline.get("median")
    trial_best, trial_median = trial.get("best"), trial.get("median")
    utility_noninferior = all(value is not None for value in
                              (baseline_best, baseline_median, trial_best, trial_median))
    if utility_noninferior:
        utility_noninferior = (
            float(trial_best) >= float(baseline_best) - TRIAL_MEDIAN_TOLERANCE
            and float(trial_median) >= float(baseline_median) - TRIAL_MEDIAN_TOLERANCE
        )
    privacy_gate = baseline_anon is not None and trial_anon is not None
    if privacy_gate:
        privacy_gate = (float(trial_anon) >= float(baseline_anon) + 0.005 if repair
                        else float(trial_anon) >= float(baseline_anon) - 0.005)
    return {"utility_noninferior": bool(utility_noninferior),
            "privacy_gate": bool(privacy_gate),
            "promoted": bool(utility_noninferior and privacy_gate)}


def _update_global_best(run_root: Path, output_dir: Path) -> dict[str, Any] | None:
    """Promote the best validated candidate across all batches atomically."""
    scores = output_dir / "candidates" / "candidate_scores.csv"
    if not scores.exists():
        return None
    candidates: list[dict[str, Any]] = []
    try:
        with scores.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("validation_ok", "").lower() != "true":
                    continue
                try:
                    utility = float(row["utility"])
                except (KeyError, TypeError, ValueError):
                    continue
                path = Path(row.get("c_path", ""))
                if not path.is_absolute():
                    path = output_dir / "candidates" / path.name
                if path.exists():
                    candidates.append({"candidate_id": row.get("candidate_id"), "seed": row.get("seed"),
                                       "utility": utility, "c_path": str(path),
                                       "selection_status": "utility_only_audit_pending",
                                       "facets": {k: row.get(k) for k in ("facet_U_gen", "facet_U_spec", "facet_U_rare", "facet_U_valid")}})
    except (OSError, csv.Error):
        return None
    if not candidates:
        return None
    candidate = max(candidates, key=lambda item: item["utility"])
    manifest_path = run_root / "best_candidate.json"
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    if float(previous.get("utility", float("-inf"))) >= candidate["utility"]:
        return previous or candidate
    destination = run_root / "best_candidate.csv"
    shutil.copy2(candidate["c_path"], destination)
    saved = {**candidate, "c_path": str(destination), "source_batch": str(output_dir)}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return saved


def _audit_anon_proxy(audit: dict[str, Any] | None) -> float | None:
    """Return a bounded local Anon proxy, never treating missing attacks as zero."""
    if not isinstance(audit, dict) or audit.get("status") != "proxy_complete":
        return None
    values = (audit.get("mia_risk"), audit.get("aia_risk"))
    try:
        if any(value is None or not np.isfinite(float(value)) for value in values):
            return None
        return float(max(0.0, 1.0 - max(float(values[0]), float(values[1]))))
    except (TypeError, ValueError):
        return None


def run_search(store: LabStore, config: RunConfig, *, require_ollama: bool = False) -> dict[str, Any]:
    check = preflight(config, require_ollama=require_ollama, ollama_host=config.ollama_host)
    if not check["ok"]:
        raise RuntimeError("preflight failed: " + "; ".join(check["problems"]))
    config.run_root.mkdir(parents=True, exist_ok=True)
    session_id = store.start_session(round(config.hours * 3600) if config.hours > 0 else 0)
    # The coordinator PID, not a short-lived scoring child, owns the durable
    # session lease.  Otherwise PID=None between batches can be mistaken for a
    # dead runner and permit a second coordinator to start against this DB.
    store.set_pid(session_id, os.getpid())
    if config.local_llm:
        store.set_meta("local_llm_enabled", True)
    store.event("runner_configuration", {"session_id": session_id,
                                         "local_llm_enabled": config.local_llm or bool(store.get_meta("local_llm_enabled", False)),
                                         "ollama_model": config.ollama_model,
                                         "ollama_models_at_start": check.get("ollama_models", []),
                                         "local_llm_interval_batches": config.local_llm_interval_batches,
                                         "jev_shadow_enabled": config.jev})
    # When no fixed seed list is supplied, seed allocation is persisted in the
    # lab database.  A crash can therefore requeue an unfinished range rather
    # than silently restarting from seed_start and evaluating duplicates.
    seed_scheduler = None
    if not config.seeds:
        from .scheduler import SeedRangeScheduler
        namespace = f"search:{config.dist.resolve()}:{config.kit_dir.resolve()}"
        seed_scheduler = SeedRangeScheduler(store, namespace=namespace)
    started = time.monotonic()
    session_row = store.session(session_id) or {}
    # deadline_at is wall-clock and persisted so sleep cannot silently extend
    # a bounded run.  Monotonic is retained only for local duration metrics.
    deadline_value = session_row.get("deadline_at")
    deadline = float(deadline_value) if deadline_value is not None else None
    result: dict[str, Any] = {"session_id": session_id, "preflight": check, "batches": []}
    batch_number = 0
    last_ollama_check_batch = -config.local_llm_interval_batches
    ollama_models = list(check.get("ollama_models", []))
    next_intake = 0.0
    hard_failure = False
    defaults = {"tau_general": 100.0, "tau_rare": 100.0, "rare_multiplier": 1.0,
                "jitter_general": 0.02, "jitter_rare": 0.05, "penalizer": 0.0,
                "outcome_draws": config.outcome_draws, "outcome_step": config.outcome_step,
                "rng_mode": "random"}
    search_params: dict[str, Any] = {**defaults, **(store.get_meta("search_params", {}) or {}), **config.search_params}
    # LLM proposals become controlled comparison trials. They are not promoted
    # merely because a reviewer returned "accept".
    pending_trial = store.get_meta("pending_trial", {}) or {}
    trial_params: dict[str, Any] | None = (None if pending_trial.get("awaiting_baseline_audit")
                                           else pending_trial.get("params"))
    trial_baseline: dict[str, float | int | None] | None = pending_trial.get("baseline")
    trial_seeds: list[int] | None = pending_trial.get("seeds")
    trial_repair_id: str | None = pending_trial.get("repair_id")
    jev_controller = JevController(store) if config.jev else None
    jev_disabled_recorded = False
    cloud_disabled_recorded = False
    if trial_seeds and not config.seeds:
        # Rebuild the seed iterator at the persisted comparison point. Without
        # this, a restart would consume and skip a fresh batch before replaying
        # the pending trial.
        config = replace(config, seed_start=min(trial_seeds))
    previous_handlers: dict[int, Any] = {}

    def handle_parent_signal(signum: int, _frame: Any) -> None:
        # A terminal close normally sends SIGHUP to the Codex/CLI process.  The
        # worker is a separate process group, so explicitly reap it before the
        # parent exits instead of leaving a five-hour search orphaned.
        store.request_stop(session_id)
        for key, proc in list(_ACTIVE_PROCESSES.items()):
            if key == session_id or key.startswith(session_id + ":"):
                _terminate_process_group(proc, grace=2.0)

    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_parent_signal)
    try:
        for generated_seeds in _seed_batches(config):
            if not _before_deadline(deadline) or store.stop_requested(session_id):
                break
            seed_lease = None
            if seed_scheduler is not None and trial_seeds is None:
                seed_lease = seed_scheduler.claim(
                    f"{session_id}:coordinator", count=config.batch_size,
                    start_hint=max(config.seed_start, store.next_seed()),
                )
                seeds = list(seed_lease.seeds)
            else:
                seeds = list(trial_seeds or generated_seeds)
            pending_state = store.get_meta("pending_trial", {}) or {}
            awaiting_baseline_audit = bool(pending_state.get("awaiting_baseline_audit"))
            if awaiting_baseline_audit:
                trial_params = None
                trial_baseline = pending_state.get("baseline")
                trial_seeds = list(pending_state.get("seeds") or seeds)
                trial_repair_id = pending_state.get("repair_id")
            is_trial = trial_params is not None
            if is_trial:
                trial_seeds = None
            if config.intake_specs is not None and time.monotonic() >= next_intake:
                try:
                    intake_updates = []
                    for spec in load_specs(config.intake_specs):
                        update = update_mirror(spec, config.mirror_root)
                        paths = changed_paths(update["mirror"], update["previous_sha"], update["sha"])
                        diff = diff_excerpt(update["mirror"], update["previous_sha"], update["sha"])
                        record = make_candidate_record(update, paths, diff)
                        added = store.add_intake_candidate(record)
                        intake_updates.append({"repo": update["name"], "sha": update["sha"], "changed": update["changed"], "new_candidate": added})
                    result.setdefault("intake", []).extend(intake_updates)
                except Exception as exc:  # noqa: BLE001
                    store.event("intake_failed", {"error": f"{type(exc).__name__}: {exc}"})
                    result.setdefault("intake_errors", []).append(str(exc))
                next_intake = time.monotonic() + max(60.0, config.intake_interval_seconds)
            batch_number += 1
            batch_params = dict(trial_params or search_params)
            eid = store.add_experiment(
                hypothesis="continuous sora_synth search batch",
                config={"dist": str(config.dist), "kit_dir": str(config.kit_dir), "seeds": seeds,
                        "search_params": batch_params, "trial": trial_params is not None, "batch": batch_number,
                        "parallel_workers": config.parallel_workers,
                        "seed_chunk_size": config.seed_chunk_size,
                        "seed_lease_id": seed_lease.lease_id if seed_lease is not None else None},
            )
            store.update_experiment(eid, state="running")
            output_dir = config.run_root / f"batch_{batch_number:05d}_{eid}"
            command_prefix = [sys.executable, "-m", "sora_synth.cli", "search", "--dist", str(config.dist),
                       "--kit-dir", str(config.kit_dir), "--outcome-draws", str(batch_params["outcome_draws"]),
                       "--outcome-step", str(batch_params["outcome_step"]),
                       "--tau-general", str(batch_params["tau_general"]), "--tau-rare", str(batch_params["tau_rare"]),
                       "--rare-multiplier", str(batch_params["rare_multiplier"]),
                       "--jitter-general", str(batch_params["jitter_general"]), "--jitter-rare", str(batch_params["jitter_rare"]),
                       "--penalizer", str(batch_params["penalizer"]),
                       "--rng-mode", str(batch_params.get("rng_mode", "random"))]
            if config.parallel_workers > 1 and len(seeds) > 1:
                batch_result = _run_parallel_seed_batch(
                    store, session_id, command_prefix, seeds, output_dir,
                    workers=config.parallel_workers, chunk_size=config.seed_chunk_size,
                    deadline=deadline, poll_seconds=config.poll_seconds)
            else:
                command = [*command_prefix, "--out-dir", str(output_dir / "candidates"),
                           "--seeds", ",".join(map(str, seeds))]
                batch_result = _run_batch(store, session_id, command, output_dir, deadline, config.poll_seconds)
            batch_result.update({"experiment_id": eid, "seeds": seeds})
            batch_result["best_utility"] = _best_score(output_dir)
            batch_result["score_summary"] = _score_summary(output_dir)
            batch_result["global_best"] = _update_global_best(config.run_root, output_dir)
            result["batches"].append(batch_result)
            utilities = [x.get("best_utility") for x in result["batches"] if x.get("best_utility") is not None]
            result["best_utility"] = max(utilities) if utilities else None
            ok = batch_result["returncode"] == 0
            state = "scored" if ok else ("interrupted" if batch_result["stop_requested"] or batch_result["timeout"] else "failed")
            hard_failure = hard_failure or state == "failed"
            if not ok and trial_params is not None:
                # A failed comparison must be retried with the same seeds and
                # parameters after a transient worker or power interruption.
                trial_seeds = list(seeds)
                store.set_meta("pending_trial", {"params": trial_params,
                                                  "baseline": trial_baseline,
                                                  "seeds": trial_seeds,
                                                  "repair_id": trial_repair_id})
            local_enabled = config.local_llm or bool(store.get_meta("local_llm_enabled", False))
            local_llm_due = local_enabled and batch_number % config.local_llm_interval_batches == 0
            if local_llm_due and _before_deadline(deadline):
                if batch_number - last_ollama_check_batch >= min(10, config.local_llm_interval_batches):
                    try:
                        ollama_models = _ollama_models(config.ollama_host)
                        store.event("local_llm_health", {"status": "available", "models": ollama_models,
                                                         "requested_model": config.ollama_model,
                                                         "batch": batch_number})
                    except RuntimeError as exc:
                        ollama_models = []
                        store.event("local_llm_health", {"status": "unavailable",
                                                         "error": f"{type(exc).__name__}: {exc}",
                                                         "batch": batch_number})
                    last_ollama_check_batch = batch_number
                if config.ollama_model in ollama_models:
                    store.event("local_llm_review_started", {"batch": batch_number,
                                                              "model": config.ollama_model,
                                                              "session_id": session_id})
                    batch_result["local_llm"] = _local_review(store, batch=batch_number, seeds=seeds,
                                                               model=config.ollama_model, host=config.ollama_host,
                                                               output_dir=output_dir, current_params=search_params,
                                                               session_id=session_id)
                    store.event("local_llm_review_finished", {"batch": batch_number,
                                                               "status": batch_result["local_llm"]["status"],
                                                               "accepted_changes": bool(batch_result["local_llm"].get("accepted_changes")),
                                                               "accepted_attack": bool(batch_result["local_llm"].get("accepted_attack_spec"))})
                    accepted = batch_result["local_llm"].get("accepted_changes", {})
                    if accepted and trial_params is None:
                        trial_params = _apply_changes(search_params, accepted)
                        trial_baseline = batch_result["score_summary"]
                        # Re-run the same seeds so the proposal is compared
                        # against a controlled baseline rather than a new
                        # random draw. The generator still advances past them
                        # exactly once when the outer iterator continues.
                        trial_seeds = list(seeds)
                        batch_result["next_search_params"] = trial_params
                        store.set_meta("pending_trial", {"params": trial_params,
                                                          "baseline": trial_baseline,
                                                          "seeds": trial_seeds,
                                                          "awaiting_baseline_audit": True,
                                                          "audit_seed": batch_number})
                        store.event("parameter_trial_queued", {"experiment_id": eid, "changes": accepted,
                                                                 "baseline": trial_baseline, "params": trial_params})
                else:
                    batch_result["local_llm"] = {"status": "skipped_model_unavailable",
                                                  "requested_model": config.ollama_model,
                                                  "available_models": ollama_models}
                    store.event("local_llm_review_skipped", {"batch": batch_number,
                                                              **batch_result["local_llm"]})
            # Automatic Astra/cloud advice is intentionally disabled in the
            # continuous runner. The standalone `consult` command remains
            # available for an explicit human-triggered review.
            if config.cloud_advice and not cloud_disabled_recorded:
                store.event("cloud_advice_disabled", {"session_id": session_id, "experiment_id": eid,
                                                        "reason": "automatic Astra calls disabled by continuous policy"})
                cloud_disabled_recorded = True
            if ok and _before_deadline(deadline) and (
                is_trial or awaiting_baseline_audit or local_llm_due
                or (config.audit_interval_batches and batch_number % config.audit_interval_batches == 0)
            ):
                try:
                    audit_params = {key: batch_params[key] for key in (
                        "tau_general", "tau_rare", "rare_multiplier",
                        "jitter_general", "jitter_rare", "penalizer", "rng_mode",
                    ) if key in batch_params}
                    # The next audit uses only the last persisted active suite.
                    # Challengers remain calibration evidence until they pass
                    # the archive gate; this prevents an unconfirmed mutation
                    # from changing the comparison world mid-campaign.
                    audit_attack_specs = None
                    archive_path = config.run_root / "attack-archive.json"
                    if archive_path.exists():
                        try:
                            from .attack_archive import AttackArchive
                            archive = AttackArchive.load(archive_path)
                            active_specs = archive.active_specs()
                            if active_specs:
                                audit_attack_specs = [spec.to_dict() for spec in active_specs]
                        except Exception as exc:  # noqa: BLE001
                            store.event("attack_archive_unavailable", {"error": f"{type(exc).__name__}: {exc}"})
                    pending_before_audit = store.get_meta("pending_trial", {}) or {}
                    if is_trial and "audit_attack_specs" in (trial_baseline or {}):
                        audit_attack_specs = (trial_baseline or {}).get("audit_attack_specs")
                    audit_seed = int((trial_baseline or {}).get("audit_seed", batch_number)) if is_trial else batch_number
                    audit = member_holdout_audit(config.dist, config.kit_dir, seed=audit_seed,
                                                 permutations=config.audit_permutations,
                                                 model_params=audit_params,
                                                 attack_specs=audit_attack_specs)
                    batch_result["audit"] = audit
                    # Keep only a bounded, non-row-level packet for the next
                    # local review.  The complete audit remains in the event
                    # log; this summary is the feedback edge of the loop.
                    attack_summary = {
                        key: {name: value.get(name) for name in ("a_mia", "tpr_rare", "beta_raw")}
                        for key, value in audit.get("results", {}).items()
                        if isinstance(value, dict) and "a_mia" in value
                    }
                    attack_summary["_aia"] = {"aia_risk": audit.get("aia_risk")}
                    aia_methods = (audit.get("aia", {}) or {}).get("methods", {})
                    for attack_id, row in aia_methods.items():
                        if isinstance(row, dict) and row.get("R") is not None:
                            attack_summary[f"aia_{attack_id}"] = {"aia_risk": row.get("R")}
                    store.set_meta("last_audit", {
                        "status": audit.get("status"),
                        "mia_risk": audit.get("mia_risk"),
                        "aia_risk": audit.get("aia_risk"),
                        "rare_kind": audit.get("rare_kind"),
                        "attacks": attack_summary,
                        "batch": batch_number,
                    })
                    store.event("audit_proxy_complete", {"experiment_id": eid, "audit": audit})
                    if not is_trial and pending_before_audit.get("awaiting_baseline_audit"):
                        baseline_with_audit = dict(pending_before_audit.get("baseline") or {})
                        baseline_with_audit.update({"anon_proxy": _audit_anon_proxy(audit),
                                                    "audit_seed": audit_seed,
                                                    "audit_attack_specs": audit_attack_specs,
                                                    "audit_status": audit.get("status")})
                        pending_before_audit.update({"baseline": baseline_with_audit,
                                                     "awaiting_baseline_audit": False})
                        store.set_meta("pending_trial", pending_before_audit)
                        trial_baseline = baseline_with_audit
                        store.event("parameter_trial_baseline_audited", {"experiment_id": eid,
                                                                          "audit_seed": audit_seed,
                                                                          "anon_proxy": baseline_with_audit["anon_proxy"],
                                                                          "status": baseline_with_audit["audit_status"]})
                        trial_params = pending_before_audit.get("params")
                        trial_seeds = list(pending_before_audit.get("seeds") or seeds)
                        trial_repair_id = pending_before_audit.get("repair_id")
                except Exception as exc:  # noqa: BLE001
                    batch_result["audit"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
                    store.event("audit_proxy_unavailable", {"experiment_id": eid, "error": batch_result["audit"]["error"]})
                    if not is_trial and pending_before_audit.get("awaiting_baseline_audit"):
                        retries = int(pending_before_audit.get("audit_retries", 0) or 0) + 1
                        if retries <= 3:
                            pending_before_audit.update({"audit_retries": retries,
                                                         "seeds": list(seeds)})
                            store.set_meta("pending_trial", pending_before_audit)
                            trial_params = None
                            trial_seeds = list(seeds)
                            store.event("parameter_trial_baseline_deferred", {"experiment_id": eid,
                                                                              "reason": "audit_incomplete",
                                                                              "retries": retries})
                        else:
                            store.set_meta("pending_trial", None)
                            trial_params = None
                            trial_baseline = None
                            trial_seeds = None
                            store.event("parameter_trial", {"experiment_id": eid, "promoted": False,
                                                             "reason": "baseline_audit_incomplete_after_retries"})
            if config.jev:
                if jev_controller is None or not jev_controller.available:
                    if not jev_disabled_recorded:
                        store.event("jev_disabled", {"reason": "TYPESAFE_API_KEY or JEV_API_KEY is not set"})
                        jev_disabled_recorded = True
                elif ok and isinstance(batch_result.get("audit"), dict):
                    audit_snapshot = batch_result.get("audit") or {}
                    score_summary = batch_result.get("score_summary") or {}
                    snapshot = {
                        "session": {"batch": batch_number, "scored": len(result["batches"]),
                                     "attack_generation": int(store.get_meta("attack_generation", 0) or 0)},
                        "utility": {"best": result.get("best_utility"),
                                    "batch_best": score_summary.get("best"),
                                    "batch_median": score_summary.get("median")},
                        "audit": {"status": audit_snapshot.get("status"),
                                  "mia_risk": audit_snapshot.get("mia_risk"),
                                  "aia_risk": audit_snapshot.get("aia_risk"),
                                  "rare_kind": audit_snapshot.get("rare_kind")},
                        "search": {"trial": bool(is_trial), "repair_pending": bool(trial_repair_id),
                                   "candidate_seeds": len(seeds)},
                    }
                    decision = jev_controller.decide(snapshot)
                    if decision is not None:
                        store.event("jev_decision_shadow", {
                            "experiment_id": eid,
                            "request_hash": decision.request_hash,
                            "input_tokens": decision.input_tokens,
                            "estimated_cost_usd": decision.input_tokens * 0.042 / 1_000_000,
                            "latency_ms": decision.latency_ms,
                            "response": decision.response,
                            "mode": "shadow",
                        })

            # Repair trials are promoted only after their forced independent
            # audit.  The previous ordering inspected batch_result["audit"]
            # before that audit was created, which made every repair look like
            # an inconclusive U-only trial.
            if ok and is_trial and trial_params is not None:
                audit = batch_result.get("audit")
                anon = _audit_anon_proxy(audit)
                baseline_anon = (trial_baseline or {}).get("anon_proxy")
                if anon is None or baseline_anon is None:
                    pending = store.get_meta("pending_trial", {}) or {}
                    retries = int(pending.get("audit_retries", 0) or 0) + 1
                    if retries <= 3:
                        trial_seeds = list(seeds)
                        store.set_meta("pending_trial", {"params": trial_params,
                                                          "baseline": trial_baseline,
                                                          "seeds": trial_seeds,
                                                          "repair_id": trial_repair_id,
                                                          "audit_retries": retries})
                        store.event("parameter_trial_deferred", {"experiment_id": eid,
                                                                  "repair_id": trial_repair_id,
                                                                  "reason": "audit_incomplete",
                                                                  "retries": retries})
                    else:
                        store.event("parameter_trial", {"experiment_id": eid, "promoted": False,
                                                         "reason": "audit_incomplete_after_retries",
                                                         "repair_id": trial_repair_id})
                        trial_params = None
                        trial_baseline = None
                        trial_seeds = None
                        trial_repair_id = None
                        store.set_meta("pending_trial", None)
                else:
                    trial_best = batch_result["score_summary"].get("best")
                    trial_median = batch_result["score_summary"].get("median")
                    baseline_best = (trial_baseline or {}).get("best")
                    baseline_median = (trial_baseline or {}).get("median")
                    gates = _trial_gates(baseline=trial_baseline or {},
                                         trial=batch_result["score_summary"],
                                         baseline_anon=baseline_anon,
                                         trial_anon=anon,
                                         repair=bool(trial_repair_id))
                    utility_noninferior = gates["utility_noninferior"]
                    privacy_gate = gates["privacy_gate"]
                    promoted = gates["promoted"]
                    if promoted:
                        search_params = dict(trial_params)
                        store.set_meta("search_params", search_params)
                    store.event("parameter_trial", {"experiment_id": eid, "promoted": promoted,
                                                     "baseline": trial_baseline,
                                                     "trial": batch_result["score_summary"],
                                                     "trial_anon_proxy": anon,
                                                     "utility_noninferior": utility_noninferior,
                                                     "privacy_gate": privacy_gate,
                                                     "params": trial_params,
                                                     "repair_id": trial_repair_id})
                    batch_result["parameter_trial"] = {"promoted": promoted,
                                                        "baseline": trial_baseline,
                                                        "trial_anon_proxy": anon}
                    trial_params = None
                    trial_baseline = None
                    trial_seeds = None
                    trial_repair_id = None
                    store.set_meta("pending_trial", None)
            if ok and isinstance(batch_result.get("audit"), dict) and batch_result["audit"].get("status") == "proxy_complete" and trial_params is None and "parameter_trial" not in batch_result:
                from .coevolution import propose_repairs, repair_for_case
                repair_summary = dict(batch_result["audit"].get("results", {}))
                repair_summary["_aia"] = {"aia_risk": batch_result["audit"].get("aia_risk")}
                for attack_id, row in ((batch_result["audit"].get("aia", {}) or {}).get("methods", {}) or {}).items():
                    if isinstance(row, dict) and row.get("R") is not None:
                        repair_summary[f"aia_{attack_id}"] = {"aia_risk": row.get("R")}
                cases = propose_repairs(repair_summary,
                                        batch=batch_number, current_params=batch_params)
                if cases:
                    repair = repair_for_case(cases[0], batch_params)
                    trial_params = _apply_changes(search_params, dict(repair.changes))
                    trial_baseline = {**batch_result["score_summary"],
                                      "anon_proxy": _audit_anon_proxy(batch_result["audit"]),
                                      "audit_seed": int(batch_result["audit"].get("seed", batch_number)),
                                      "audit_attack_specs": audit_attack_specs}
                    trial_seeds = list(seeds)
                    trial_repair_id = repair.repair_id
                    batch_result["defense_repair"] = {
                        "case_id": cases[0].case_id, "attack_id": cases[0].attack_id,
                        "repair_id": repair.repair_id, "changes": dict(repair.changes),
                        "paired_trial": True,
                    }
                    store.set_meta("pending_trial", {"params": trial_params,
                                                      "baseline": trial_baseline,
                                                      "seeds": trial_seeds,
                                                      "repair_id": repair.repair_id})
                    store.event("defense_repair_queued", {"experiment_id": eid,
                                                           "case_id": cases[0].case_id,
                                                           "repair_id": repair.repair_id,
                                                           "changes": dict(repair.changes)})
            if ok and _before_deadline(deadline) and config.attack_evolution_interval_batches and batch_number % config.attack_evolution_interval_batches == 0:
                try:
                    generation = int(store.get_meta("attack_generation", 0) or 0) + 1
                    archive_path = config.run_root / "attack-archive.json"
                    evolution = evolve_member_holdout_attacks(
                        config.dist, config.kit_dir, seed=batch_number,
                        train_fraction=0.8, permutations=config.attack_evolution_permutations,
                        population_size=config.attack_evolution_population_size,
                        generation=generation, calibration_trials=config.attack_evolution_calibration_trials,
                        archive_path=archive_path,
                        proposed_specs=store.get_meta("pending_attack_specs", []) or [],
                    )
                    if store.get_meta("pending_attack_specs", []):
                        store.set_meta("pending_attack_specs", [])
                    store.set_meta("attack_generation", generation)
                    batch_result["attack_evolution"] = {
                        "status": evolution.get("status"), "generation": generation,
                        "active": len(evolution.get("active", [])),
                        "challengers": len(evolution.get("challengers", [])),
                        "archive": str(archive_path),
                    }
                    store.event("attack_evolution_complete", {"experiment_id": eid,
                                                                 "summary": batch_result["attack_evolution"]})
                except Exception as exc:  # noqa: BLE001
                    batch_result["attack_evolution"] = {
                        "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
                    store.event("attack_evolution_unavailable", {"experiment_id": eid,
                                                                    "error": batch_result["attack_evolution"]["error"]})
            store.update_experiment(eid, state=state, result=batch_result,
                                    error=None if ok else f"search batch exited {batch_result['returncode']}")
            if seed_lease is not None:
                if ok:
                    seed_scheduler.complete(seed_lease)
                else:
                    # Keep the range available for the next resume.  The
                    # experiment row and worker logs still retain the failed
                    # attempt for diagnosis.
                    seed_scheduler.fail(seed_lease, requeue=True)
            store.heartbeat(session_id)
            if not ok and state == "failed":
                # A broken batch should be visible, but a single model failure must not
                # silently terminate an otherwise resumable session.
                continue
        result.update({"returncode": 2 if hard_failure else 0, "elapsed_seconds": time.monotonic() - started,
                       "stopped": store.stop_requested(session_id), "batches_completed": batch_number})
        return result
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        store.stop_session(session_id, status="stopped" if store.stop_requested(session_id) else "completed")


def _member_holdout_context(dist: Path, kit_dir: Path, *, seed: int, train_fraction: float,
                            model_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one valid audit world; all evaluators in a generation share it."""
    from sora_synth.data import load_dataset
    from sora_synth.generator import fit_model

    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    dataset = load_dataset(dist, kit_dir)
    rng = np.random.default_rng(seed)
    rare = dataset.rare_truth
    fit_idx: list[int] = []
    holdout_idx: list[int] = []
    for value in (False, True):
        indices = np.flatnonzero(rare == value)
        shuffled = rng.permutation(indices)
        if len(shuffled) < 2:
            continue
        cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
        fit_idx.extend(shuffled[:cut].tolist())
        holdout_idx.extend(shuffled[cut:].tolist())
    fit_idx = np.array(sorted(fit_idx))
    if len(fit_idx) == 0 or len(holdout_idx) == 0:
        raise ValueError("member split too small")
    a_order = rng.permutation(len(dataset.a_bg))
    ref_n = min(len(fit_idx), max(2, len(a_order) // 5))
    prior_idx, reference_idx = a_order[ref_n:], a_order[:ref_n]
    if len(prior_idx) < 2 or len(reference_idx) < 2:
        raise ValueError("independent reference too small")
    member_dataset = replace(
        dataset,
        b=dataset.b.iloc[fit_idx].reset_index(drop=True),
        b_self=dataset.b_self.iloc[fit_idx].reset_index(drop=True),
        a_bg=dataset.a_bg.iloc[prior_idx].reset_index(drop=True),
    )
    model = fit_model(member_dataset, **(model_params or {}))
    candidate = model.sample(seed + 100_000)
    member = dataset.b.iloc[fit_idx].reset_index(drop=True).drop(columns=["record_id"], errors="ignore")
    nonmember = dataset.b.iloc[holdout_idx].reset_index(drop=True).drop(columns=["record_id"], errors="ignore")
    reference = dataset.a_bg.iloc[reference_idx].reset_index(drop=True).drop(columns=["record_id"], errors="ignore")
    # AIA proxy: use an equal-sized holdout target and an independently drawn
    # public-background control. Their truth columns stay on the audit side;
    # the attacker receives only C and the declared QI columns.
    control_idx = rng.choice(len(dataset.a_bg), size=len(holdout_idx), replace=len(dataset.a_bg) < len(holdout_idx))
    aia_target = dataset.b.iloc[holdout_idx].reset_index(drop=True).drop(columns=["record_id"], errors="ignore")
    aia_control = dataset.a_bg.iloc[control_idx].reset_index(drop=True).drop(columns=["record_id"], errors="ignore")
    return {"candidate": candidate, "member": member, "nonmember": nonmember,
            "reference": reference, "rare_labels": np.r_[rare[fit_idx], rare[holdout_idx]],
            "aia_target": aia_target, "aia_control": aia_control,
            "utility_ref": dataset.utility_ref,
            "model_params": dict(model_params or {}), "fit_member_n": len(member),
            "holdout_nonmember_n": len(nonmember), "reference_n": len(reference),
            "a_bg_partition": {"prior_n": len(prior_idx), "reference_n": len(reference_idx)}}


def member_holdout_audit(dist: Path, kit_dir: Path, *, seed: int = 0, train_fraction: float = 0.8,
                         permutations: int = 200, model_params: dict[str, Any] | None = None,
                         candidate: Any | None = None,
                         attack_specs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run a genuine member/nonmember proxy: fit only on member rows.

    This deliberately does not call the official U scorer because the generated
    cohort is the member split size.  It is an audit of leakage, not official
    utility or official Anon.
    """
    from sora_synth.audit import evaluate_aia_suite, evaluate_mia
    if candidate is not None:
        raise ValueError("member_holdout_audit cannot evaluate a candidate fit on all B; use a saved-C distribution diagnostic")
    context = _member_holdout_context(dist, kit_dir, seed=seed, train_fraction=train_fraction,
                                      model_params=model_params)
    candidate = context["candidate"]
    member = context["member"]
    nonmember = context["nonmember"]
    reference = context["reference"]
    rare_labels = context["rare_labels"]
    candidate_source = "member_fit_generated"
    mia_attack_specs = ([spec for spec in attack_specs if not str(spec.get("family", "")).startswith("aia_")]
                        if attack_specs is not None else None)
    aia_attack_specs = ([spec for spec in attack_specs if str(spec.get("family", "")).startswith("aia_")]
                        if attack_specs is not None else None)
    needs_shadow = mia_attack_specs is None or any(str(spec.get("family", "")) == "shadow" for spec in mia_attack_specs)
    shadow_contexts = [
        _member_holdout_context(dist, kit_dir, seed=seed + 10_000 + index,
                                 train_fraction=train_fraction, model_params=model_params)
        for index in range(2)
    ] if needs_shadow else []
    results = evaluate_mia(candidate, member, nonmember, reference=reference,
                            rare_ref=context["utility_ref"], rare_labels=rare_labels,
                            attack_specs=mia_attack_specs or None,
                            shadow_members=[item["member"] for item in shadow_contexts] if needs_shadow else None,
                            shadow_nonmembers=[item["nonmember"] for item in shadow_contexts] if needs_shadow else None,
                            n_perm=permutations, seed=seed + 200_000)
    aia = evaluate_aia_suite(candidate, context["aia_target"], context["aia_control"], seed=seed + 300_000,
                             attack_specs=aia_attack_specs or None)
    numeric_attacks = [value for key, value in results.items()
                       if key != "_meta" and isinstance(value, dict) and np.isfinite(value.get("a_mia", np.nan))]
    mia_risk = max((float(value["a_mia"]) for value in numeric_attacks), default=float("nan"))
    aia_risk = float(aia.get("conservative_R", float("nan")))
    audit_status = "proxy_complete" if np.isfinite(mia_risk) and np.isfinite(aia_risk) else "proxy_incomplete"
    return {
        "kind": "member_nonmember_proxy",
        "status": audit_status,
        "seed": seed,
        "train_fraction": train_fraction,
        "fit_member_n": context["fit_member_n"],
        "holdout_nonmember_n": context["holdout_nonmember_n"],
        "reference_n": context["reference_n"],
        "model_params": context["model_params"],
        "candidate_source": candidate_source,
        "results": results,
        "mia_risk": mia_risk,
        "aia_risk": aia_risk,
        "aia": aia,
        "audit_status": audit_status,
        "rare_kind": "member_holdout_truth",
        "provenance": {"generator_fit": "B_fit + A_bg_prior_slice", "member_labels": "B_fit",
                       "nonmember_labels": "B_holdout", "reference": "A_bg_independent_slice",
                       "candidate": candidate_source,
                       "shadow_attack": "independent_world" if needs_shadow else "not_requested",
                       "aia_control": "A_bg_equal_sized_proxy",
                       "a_bg_partition": context["a_bg_partition"]},
    }


def evolve_member_holdout_attacks(dist: Path, kit_dir: Path, *, seed: int = 0,
                                  train_fraction: float = 0.8, permutations: int = 100,
                                  model_params: dict[str, Any] | None = None,
                                  population_size: int = 16, generation: int = 1,
                                  archive_path: Path | None = None,
                                  calibration_trials: int = 32,
                                  proposed_specs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run one bounded attack-spec generation on a valid member-fit world.

    Each calibration trial creates a fixed member-fit pseudo-world; all specs
    use the same world and permutation stream for that trial.  This makes
    paired promotion order-independent while retaining a bounded development
    proxy; it does not claim to reproduce the hidden competition attack pool.
    """
    import time as _time

    from .attack_archive import AttackArchive, AttackEvaluation
    from .attack_evolution import run_attack_generation
    from .attack_registry import AttackSpec, default_attack_specs
    from sora_synth.audit import evaluate_mia

    if calibration_trials < 1 or calibration_trials > 128:
        raise ValueError("calibration_trials must be in [1,128]")
    context = _member_holdout_context(dist, kit_dir, seed=seed, train_fraction=train_fraction,
                                      model_params=model_params)
    archive = None
    if archive_path is not None and archive_path.exists():
        archive = AttackArchive.load(archive_path)
    defaults = list(default_attack_specs())
    parents = archive.active_specs() if archive and archive.active_specs() else []
    present = {spec.semantic_hash for spec in parents}
    # Upgrade older MIA-only archives in place while retaining every existing
    # active attack.  Regression AIA attacks must not disappear just because
    # the archive predates AIA co-evolution.
    parents.extend(spec for spec in defaults if spec.semantic_hash not in present)
    if not parents:
        parents = defaults
    parent_ids = {item.attack_id: item for item in parents}
    llm_specs = []
    for raw in proposed_specs or []:
        spec = AttackSpec.from_dict(raw)
        parent = parent_ids.get(spec.parents[0]) if spec.parents else None
        if parent is None or parent.family != spec.family:
            continue
        llm_specs.append(spec)
    regression_ids = [spec.attack_id for spec in default_attack_specs()]
    evaluations: list[AttackEvaluation] = []
    seen_families: dict[str, int] = {}
    trial_cache: dict[str, list[float]] = {}
    context_cache: dict[int, dict[str, Any]] = {0: context}
    shadow_cache: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    parents_by_id = {spec.attack_id: spec for spec in parents}

    def trial_context(trial: int) -> dict[str, Any]:
        if trial not in context_cache:
            context_cache[trial] = _member_holdout_context(
                dist, kit_dir, seed=seed + generation * 100_000 + trial,
                train_fraction=train_fraction, model_params=model_params)
        return context_cache[trial]

    def trial_values(spec: AttackSpec) -> list[float]:
        key = spec.semantic_hash
        if key in trial_cache:
            return trial_cache[key]
        values: list[float] = []
        for trial in range(calibration_trials):
            trial_world = trial_context(trial)
            if spec.family.startswith("aia_"):
                from sora_synth.audit import evaluate_aia_suite
                measured_aia = evaluate_aia_suite(
                    trial_world["candidate"], trial_world["aia_target"], trial_world["aia_control"],
                    seed=seed + generation * 100_000 + trial, attack_specs=[spec.to_dict()])
                row = measured_aia.get("methods", {}).get(spec.attack_id, {})
                if not isinstance(row, dict) or "R" not in row:
                    raise ValueError(f"AIA attack unavailable: {spec.attack_id}")
                values.append(float(row["R"]))
                continue
            if spec.family == "shadow":
                if trial not in shadow_cache:
                    shadow_cache[trial] = ([], [])
                    members, nonmembers = shadow_cache[trial]
                    for index in range(2):
                        shadow = _member_holdout_context(
                            dist, kit_dir,
                            seed=seed + generation * 100_000 + trial + (index + 1) * 10_000,
                            train_fraction=train_fraction, model_params=model_params)
                        members.append(shadow)
                        nonmembers.append(shadow)
                shadow_members = [item["member"] for item in shadow_cache[trial][0]]
                shadow_nonmembers = [item["nonmember"] for item in shadow_cache[trial][1]]
            else:
                shadow_members = shadow_nonmembers = None
            measured = evaluate_mia(
                trial_world["candidate"], trial_world["member"], trial_world["nonmember"],
                reference=trial_world["reference"], rare_labels=trial_world["rare_labels"],
                attack_specs=[spec.to_dict()], n_perm=permutations,
                shadow_members=shadow_members, shadow_nonmembers=shadow_nonmembers,
                # All specs share the same trial world and permutation stream.
                seed=seed + generation * 100_000 + trial)
            values.append(float(measured[spec.attack_id]["a_mia"]))
        trial_cache[key] = values
        return values

    def evaluate(spec: AttackSpec) -> AttackEvaluation:
        started = _time.perf_counter()
        values = trial_values(spec)
        strength = float(sum(values) / len(values))
        family_count = seen_families.get(spec.family, 0)
        seen_families[spec.family] = family_count + 1
        novelty = 1.0 / (1.0 + family_count)
        evaluation = AttackEvaluation(
            spec=spec, strength=strength, novelty=novelty,
            cost_seconds=_time.perf_counter() - started,
            split_id=f"member_holdout:{seed}:{train_fraction}",
            evidence={"trial_strengths": values, "calibration_trials": calibration_trials,
                      "world_plan": "member_fit_world_v1"},
        )
        if spec.parents and calibration_trials >= 32:
            parent = parents_by_id.get(spec.parents[0])
            if parent is not None:
                from .calibration import assess_promotion
                parent_values = trial_values(parent)
                deltas = [child - base for child, base in zip(values, parent_values)]
                decision = assess_promotion(values, parent_values,
                                            family_deltas={spec.family: deltas},
                                            min_trials=32, max_trials=128, seed=seed)
                evaluation = AttackEvaluation(
                    spec=spec, strength=strength, novelty=novelty,
                    cost_seconds=_time.perf_counter() - started,
                    status="complete" if decision.status == "confirmed" else "rejected",
                    split_id=f"member_holdout:{seed}:{train_fraction}",
                    evidence={"trial_strengths": values, "calibration_trials": calibration_trials,
                              "promotion": decision.__dict__, "world_plan": "member_fit_world_v1"},
                )
        evaluations.append(evaluation)
        return evaluation

    result = run_attack_generation(parents, evaluate, generation=generation, seed=seed,
                                   population_size=population_size, active_limit=16,
                                   challenger_limit=8,
                                   regression_ids=regression_ids, archive=archive,
                                   additional_specs=llm_specs)
    archive = archive or AttackArchive(active_limit=16, challenger_limit=8,
                                       regression_ids=regression_ids)
    archive.regression_ids.update(regression_ids)
    archive.promote(result.evaluations, pool=result.evaluations)
    if archive_path is not None:
        archive.save(archive_path)
    return {
        "kind": "attack_coevolution_member_holdout",
        "status": "proxy_complete",
        "generation": generation,
        "seed": seed,
        "population_size": population_size,
        "calibration_trials": calibration_trials,
        "llm_proposals_evaluated": [item.attack_id for item in llm_specs],
        "evaluations": [item.to_dict() for item in result.evaluations],
        "active": [item.to_dict() for item in result.active],
        "challengers": [item.to_dict() for item in result.challengers],
        "provenance": {"generator_fit": "B_fit", "nonmember_labels": "B_holdout",
                       "reference": "A_bg_independent_slice", "candidate": "member_fit_generated",
                       "warning": "development proxy; hidden competition attacks are not reproduced"},
    }
