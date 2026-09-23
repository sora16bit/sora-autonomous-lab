from __future__ import annotations

import tempfile
import unittest
import subprocess
import time
import json
from unittest.mock import patch
from pathlib import Path

from sora_lab.advisor import AdvicePolicy, build_packet, consult_command, should_consult
from sora_lab.adapters import AdapterSpec, validate_adapter
from sora_lab.agents import AgentReply, _extract_json, parse_attack_proposal, validate_proposal
from sora_lab.attack_archive import AttackArchive, AttackEvaluation
from sora_lab.attack_evolution import evolve_attack_specs, run_attack_generation
from sora_lab.attack_registry import AttackSpec, default_attack_specs
from sora_lab.calibration import assess_promotion, paired_bootstrap_lower, required_trials
from sora_lab.resources import QueueStat, allocate_resources
from sora_lab.worlds import WorldPlan
from sora_lab.scheduler import SeedRangeScheduler
from sora_lab.coevolution import propose_repairs, repair_for_case
from sora_lab.intake import load_specs, make_candidate_record
from sora_lab.policy import choose_verified, pareto_front, proxy_score
from sora_lab.runner import RunConfig, _apply_changes, _local_review, _score_summary, _trial_gates, preflight, run_search
from sora_lab.store import LabStore
from sora_lab.jev import build_report


class LabTests(unittest.TestCase):
    def test_jev_report_is_credential_free_and_measures_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            store.event("jev_decision_shadow", {
                "request_hash": "abc", "input_tokens": 320,
                "estimated_cost_usd": 0.00001344, "latency_ms": 90,
                "response": {"answers": {"next_focus": {"choice": "evolve_attack"}}},
            })
            store.event("jev_fallback", {"reason": "timeout", "request_hash": "def"})
            report = build_report(store)
            self.assertEqual(report["attempts"], 2)
            self.assertEqual(report["counts"]["jev_decision_shadow"], 1)
            self.assertAlmostEqual(report["valid_response_rate"], 0.5)
            self.assertEqual(report["next_focus_distribution"], {"evolve_attack": 1})
            self.assertNotIn("abc", json.dumps(report))
            store.close()

    def test_session_accepts_forever_budget_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            sid = store.start_session(0)
            self.assertIsNone(store.session(sid)["deadline_at"])
            with self.assertRaises(RuntimeError):
                store.start_session(60)
            store.stop_session(sid)
            self.assertEqual(store.session(sid)["status"], "stopped")
            store.close()

    def test_start_session_recovers_definitively_orphaned_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            sid = store.start_session(60)
            store.db.execute("UPDATE sessions SET pid=?,last_heartbeat=? WHERE session_id=?", (99999999, time.time() - 120, sid))
            recovered = store.start_session(60)
            self.assertNotEqual(sid, recovered)
            self.assertEqual(store.session(sid)["status"], "interrupted")
            store.stop_session(recovered)
            store.close()

    def test_pareto_keeps_utility_risk_tradeoff(self):
        rows = [
            {"id": "a", "utility": 0.99, "r_MIA": 0.4, "r_AIA": 0.4},
            {"id": "b", "utility": 0.95, "r_MIA": 0.1, "r_AIA": 0.1},
            {"id": "c", "utility": 0.90, "r_MIA": 0.5, "r_AIA": 0.5},
        ]
        ids = {row["id"] for row in pareto_front(rows)}
        self.assertEqual(ids, {"a", "b"})
        self.assertAlmostEqual(proxy_score({**rows[0], "U_gen": .99, "U_spec": .99, "U_rare": .99, "U_valid": .99}), 0.60)

    def test_verified_selection_requires_complete_low_risk_audit(self):
        rows = [
            {"utility": .95, "U_gen": .95, "U_spec": .95, "U_rare": .95, "U_valid": 1.0,
             "r_MIA": .04, "r_AIA": .06, "validation_ok": True, "audit_status": "proxy_complete"},
            {"utility": .99, "U_gen": .99, "U_spec": .99, "U_rare": .99, "U_valid": 1.0,
             "r_MIA": .2, "r_AIA": .01, "validation_ok": True, "audit_status": "proxy_complete"},
        ]
        self.assertAlmostEqual(choose_verified(rows)["utility"], .95)

    def test_advice_packet_is_bounded_and_window_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            policy = AdvicePolicy()
            self.assertTrue(should_consult(store, accumulated_seconds=5 * 3600, policy=policy, needs_decision=True))
            packet = build_packet({"very_long": "x" * 10000}, max_chars=100)
            self.assertTrue(packet["summary"]["truncated"])
            store.add_advice(window_id="1", provider=policy.provider, model=policy.model, request=packet, response=None, status="manual_required")
            self.assertFalse(should_consult(store, accumulated_seconds=5 * 3600, policy=policy, needs_decision=True))
            store.close()

    def test_incident_advice_is_one_shot_without_periodic_cloud_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            policy = AdvicePolicy()
            fake = lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="{}", stderr="")
            result = consult_command(store, accumulated_seconds=10, policy=policy, packet={"incident": "failed"}, incident_id="incident-e1", command=["fake"], runner=fake)
            self.assertEqual(result["status"], "completed")
            second = consult_command(store, accumulated_seconds=10, policy=policy, packet={"incident": "failed"}, incident_id="incident-e1", command=["fake"], runner=fake)
            self.assertEqual(second["status"], "already_reserved")
            store.close()

    def test_preflight_reports_missing_inputs_without_network_requirement(self):
        result = preflight(RunConfig(Path("/missing/dist"), Path("/missing/kit"), Path("/tmp/run")))
        self.assertFalse(result["ok"])
        self.assertTrue(any("dist directory missing" in item for item in result["problems"]))

    def test_preflight_rejects_unbounded_local_review_interval(self):
        result = preflight(RunConfig(Path("/missing/dist"), Path("/missing/kit"), Path("/tmp/run"), local_llm_interval_batches=0))
        self.assertTrue(any("local_llm_interval_batches" in item for item in result["problems"]))

    def test_session_seed_resume_and_advice_reservation_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            store.add_experiment(hypothesis="seed batch", config={"seeds": [10, 11]})
            self.assertEqual(store.next_seed(), 12)
            store.set_meta("search_params", {"tau_rare": 400})
            self.assertEqual(store.get_meta("search_params")["tau_rare"], 400)
            self.assertTrue(store.reserve_advice(window_id="0", provider="astra", model="astra-low", request={"x": 1}))
            self.assertFalse(store.reserve_advice(window_id="0", provider="astra", model="astra-low", request={"x": 1}))

    def test_experiment_lease_is_single_worker_and_releasable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            eid = store.add_experiment(hypothesis="queued", config={})
            self.assertEqual(store.claim_experiment("worker-a"), eid)
            self.assertIsNone(store.claim_experiment("worker-b"))
            self.assertTrue(store.renew_experiment_lease(eid, "worker-a"))
            self.assertTrue(store.release_experiment(eid, "worker-a"))
            store.close()

    def test_seed_ranges_are_disjoint_fenced_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            scheduler = SeedRangeScheduler(store, namespace="test")
            first = scheduler.claim("worker-a", count=10, start_hint=0)
            second = scheduler.claim("worker-b", count=10)
            self.assertEqual(first.seeds, tuple(range(0, 10)))
            self.assertEqual(second.seeds, tuple(range(10, 20)))
            self.assertNotEqual(first.fencing_token, second.fencing_token)
            self.assertTrue(scheduler.record_result(first, 3, config={"v": 1}, result={"u": .9}))
            self.assertFalse(scheduler.record_result(first, 3, config={"v": 1}, result={"u": .9}))
            self.assertTrue(scheduler.complete(first))
            self.assertTrue(scheduler.complete(second))
            self.assertEqual(scheduler.summary().get("completed"), 2)
            store.close()

    def test_runner_uses_persistent_seed_lease_when_seed_list_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist, kit = root / "dist", root / "kit"
            dist.mkdir()
            (kit / "starter").mkdir(parents=True)
            for name in ("B.csv", "B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json"):
                (dist / name).write_text("{}", encoding="utf-8")
            store = LabStore(root / "lab.sqlite3")

            def fake_batch(local_store, session_id, command, output_dir, deadline, poll_seconds, *, process_key=None):
                output_dir.mkdir(parents=True, exist_ok=True)
                local_store.request_stop(session_id)
                return {"returncode": 0, "timeout": False, "stop_requested": True}

            config = RunConfig(dist, kit, root / "run", hours=.01, batch_size=2,
                               audit_interval_batches=0, attack_evolution_interval_batches=0)
            with patch("sora_lab.runner._run_batch", side_effect=fake_batch):
                result = run_search(store, config)
            self.assertEqual(result["batches_completed"], 1)
            row = store.db.execute("SELECT status,start_seed,end_seed FROM seed_ranges").fetchone()
            self.assertEqual((row["status"], row["start_seed"], row["end_seed"]), ("completed", 0, 2))
            store.close()

    def test_attack_breakthrough_produces_bounded_paired_repair(self):
        cases = propose_repairs({"marginal_support": {"a_mia": .2, "tpr_rare": .3}},
                                batch=4, current_params={"tau_rare": 100, "jitter_rare": .05})
        self.assertEqual(len(cases), 1)
        repair = repair_for_case(cases[0], cases[0].defense_params)
        self.assertTrue(repair.requires_paired_trial)
        self.assertLessEqual(repair.changes["tau_rare"], 400)

    def test_parallel_batch_merges_disjoint_worker_scores(self):
        from unittest.mock import patch
        from sora_lab.runner import _run_parallel_seed_batch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LabStore(root / "lab.sqlite3")
            sid = store.start_session(60)
            def fake_run(local_store, session_id, command, output_dir, deadline, poll_seconds, *, process_key=None, set_session_pid=True):
                target = output_dir / "candidates"
                target.mkdir(parents=True)
                seed = command[command.index("--seeds") + 1]
                (target / "candidate_scores.csv").write_text(
                    "candidate_id,seed,validation_ok,utility,c_path\n"
                    f"seed_{seed},{seed},True,0.9,{target / ('C_seed' + seed + '.csv')}\n", encoding="utf-8")
                return {"returncode": 0, "timeout": False, "stop_requested": False}
            with patch("sora_lab.runner._run_batch", side_effect=fake_run):
                result = _run_parallel_seed_batch(store, sid, ["search"], [1, 2], root / "batch",
                                                   workers=2, chunk_size=1, deadline=9999999999, poll_seconds=.01)
            self.assertEqual(result["returncode"], 0)
            merged = (root / "batch" / "candidates" / "candidate_scores.csv").read_text(encoding="utf-8")
            self.assertEqual(merged.count("True"), 2)
            store.stop_session(sid)
            store.close()

    def test_latest_event_returns_only_newest_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            store.event("audit", {"batch": 1})
            store.event("audit", {"batch": 2})
            self.assertEqual(store.latest_event("audit"), {"batch": 2})
            self.assertIsNone(store.latest_event("missing"))
            store.close()

    def test_attack_specs_are_allowlisted_and_generation_is_bounded(self):
        with self.assertRaises(ValueError):
            AttackSpec("bad", "distance", {"shell": "rm -rf"}).validate()
        specs = evolve_attack_specs(default_attack_specs(), generation=2, seed=11, population_size=16)
        self.assertEqual(len(specs), 16)
        self.assertTrue(all(spec.generation == 2 for spec in specs))
        self.assertTrue(all("shell" not in spec.to_dict() for spec in specs))

    def test_attack_generation_keeps_regression_attacks_and_archive_roundtrips(self):
        parents = list(default_attack_specs())
        def evaluate(spec):
            return AttackEvaluation(spec, 0.5 if spec.family == "distance" else 0.1, 1.0, 0.01)
        result = run_attack_generation(parents, evaluate, generation=1, seed=2, population_size=4,
                                       active_limit=12, regression_ids=[spec.attack_id for spec in parents])
        active_ids = {spec.attack_id for spec in result.active}
        self.assertTrue({spec.attack_id for spec in parents} <= active_ids)
        with tempfile.TemporaryDirectory() as tmp:
            archive = AttackArchive(active_limit=12, regression_ids=[spec.attack_id for spec in parents])
            archive.promote(result.evaluations)
            path = Path(tmp) / "attacks.json"
            archive.save(path)
            loaded = AttackArchive.load(path)
            self.assertEqual(set(loaded.active), set(archive.active))

    def test_llm_attack_proposal_enters_measured_generation_as_challenger(self):
        parents = list(default_attack_specs())
        proposed = AttackSpec("llm_test", "distance", {"k": 3}, parents=("mia_distance_k1",))
        measured = []
        def evaluate(spec):
            measured.append(spec.attack_id)
            return AttackEvaluation(spec, .4, 1.0, .01)
        result = run_attack_generation(parents, evaluate, generation=2, seed=4,
                                       population_size=2, additional_specs=[proposed])
        self.assertIn("llm_test", measured)
        self.assertTrue(any(item.spec.attack_id == "llm_test" for item in result.evaluations))

    def test_attack_semantic_identity_excludes_lineage(self):
        left = AttackSpec("a", "distance", {"k": 3}, parents=("p",), generation=1)
        right = AttackSpec("b", "distance", {"k": 3}, parents=("q",), generation=9)
        self.assertEqual(left.semantic_hash, right.semantic_hash)
        self.assertNotEqual(left.lineage_id, right.lineage_id)
        archive = AttackArchive()
        archive.add(AttackEvaluation(left, .2, .1, .1, split_id="world-1"))
        archive.add(AttackEvaluation(right, .3, .1, .1, split_id="world-2"))
        self.assertEqual(len(archive.history), 2)

    def test_promotion_calibration_requires_repeated_paired_gain(self):
        decision = assess_promotion([.05] * 32, [0.0] * 32, family_deltas={"distance": [.05] * 32})
        self.assertEqual(decision.status, "confirmed")
        self.assertEqual(assess_promotion([.05] * 8, [0.0] * 8).status, "inconclusive_min_trials")
        self.assertEqual(assess_promotion([.05] * 32, [0.0] * 32, failures=1).status, "rejected")
        self.assertGreaterEqual(required_trials(.1), 32)

    def test_world_plan_checks_shared_members_and_rare_structure(self):
        membership = __import__("numpy").array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]], dtype=bool)
        plan = WorldPlan("w0", membership, __import__("numpy").array([1, 0, 1, 0], dtype=bool))
        plan.validate(expected_cohort_sizes=(2, 2, 2))
        self.assertEqual(plan.pairwise_shared[0, 1], 1)
        self.assertEqual(plan.shared_rare(0, 2), 1)
        self.assertEqual(plan.triple_shared(0, 1, 2), 0)

    def test_resource_policy_has_floors_and_audit_backlog_override(self):
        stats = {name: QueueStat(completed=10, reward_sum=(10.0 if name == "attack" else 1.0), worker_seconds=10.0)
                 for name in ("attack", "audit", "seed", "generator")}
        allocation = allocate_resources(stats, previous={"attack": .30, "audit": .30, "seed": .25, "generator": .15})
        self.assertAlmostEqual(sum(allocation.values()), 1.0)
        self.assertGreaterEqual(allocation["audit"], .25)
        self.assertTrue(all(abs(allocation[name] - {"attack": .30, "audit": .30, "seed": .25, "generator": .15}[name]) <= .05 + 1e-9
                            for name in allocation))
        backlog = allocate_resources(stats, audit_backlog_age_minutes=90)
        self.assertGreaterEqual(backlog["audit"], .40)

    def test_llm_proposal_is_data_and_bounded(self):
        good = AgentReply("improver", "qwen", "", {"decision": "test", "reason": "ok", "changes": {"tau_rare": 400}, "next_experiment": "paired trial", "uncertainty": "U may fall"})
        self.assertEqual(validate_proposal(good), (True, []))
        bad = AgentReply("improver", "qwen", "", {"changes": {"shell": "rm -rf"}})
        self.assertFalse(validate_proposal(bad)[0])
        numeric_bad = AgentReply("improver", "qwen", "", {"changes": {"shell": 1}})
        self.assertFalse(validate_proposal(numeric_bad)[0])
        normalized = _extract_json('{"仮説":"rare shrink","変更対象":{"tau_rare":400},"期待する指標":"paired trial","反証条件":"median falls"}')
        self.assertTrue(validate_proposal(AgentReply("improver", "qwen", "", normalized))[0])

    def test_llm_attack_proposal_is_allowlisted_and_parented(self):
        valid = AgentReply("attacker", "qwen", "", {
            "reason": "rare nearest-neighbor variant", "parent_attack_id": "mia_distance_k1",
            "family": "distance", "params": {"k": 3},
        })
        parsed, reasons = parse_attack_proposal(valid, allowed_parent_ids={"mia_distance_k1"})
        self.assertEqual(reasons, [])
        self.assertEqual(parsed["parents"], ["mia_distance_k1"])
        alias = AgentReply("attacker", "qwen", "", {
            "reason": "distance variant", "parent_attack_id": "mia_distance_k1",
            "family": "distance", "params": {"distance_k": 3},
        })
        parsed_alias, alias_reasons = parse_attack_proposal(alias, allowed_parent_ids={"mia_distance_k1"})
        self.assertEqual(alias_reasons, [])
        self.assertEqual(parsed_alias["params"], {"k": 3})
        aia = AgentReply("attacker", "qwen", "", {
            "reason": "AIA neighborhood variant", "parent_attack_id": "aia_knn1",
            "family": "aia_knn", "params": {"k": 4},
        })
        parsed_aia, aia_reasons = parse_attack_proposal(aia, allowed_parent_ids={"aia_knn1"})
        self.assertEqual(aia_reasons, [])
        self.assertEqual(parsed_aia["family"], "aia_knn")
        invalid = AgentReply("attacker", "qwen", "", {
            "reason": "bad", "parent_attack_id": "mia_distance_k1", "family": "distance",
            "params": {"k": 3, "shell": "rm -rf"},
        })
        parsed, reasons = parse_attack_proposal(invalid, allowed_parent_ids={"mia_distance_k1"})
        self.assertIsNone(parsed)
        self.assertTrue(reasons)

    def test_parameter_trial_requires_utility_and_privacy_gates(self):
        baseline = {"best": .95, "median": .94}
        self.assertTrue(_trial_gates(baseline=baseline, trial={"best": .951, "median": .939},
                                     baseline_anon=.70, trial_anon=.70, repair=False)["promoted"])
        self.assertFalse(_trial_gates(baseline=baseline, trial={"best": .98, "median": .96},
                                      baseline_anon=.70, trial_anon=.69, repair=False)["promoted"])
        self.assertFalse(_trial_gates(baseline=baseline, trial={"best": .951, "median": .939},
                                      baseline_anon=.70, trial_anon=.701, repair=True)["promoted"])

    def test_intake_record_is_content_addressed(self):
        update = {"name": "team-a", "sha": "abc123"}
        paths = [{"status": "M", "path": "generator.py"}]
        record = make_candidate_record(update, paths)
        self.assertEqual(record["state"], "discovered")
        self.assertEqual(len(record["content_hash"]), 64)
        with tempfile.TemporaryDirectory() as tmp:
            store = LabStore(Path(tmp) / "lab.sqlite3")
            self.assertTrue(store.add_intake_candidate(record))
            store.update_intake_candidate(record["candidate_id"], state="triaged", details={"triage": {"valid": True}})
            self.assertEqual(store.intake_candidates()[0]["state"], "triaged")

    def test_intake_rejects_path_traversal_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = Path(tmp) / "specs.json"
            specs.write_text('[{"name":"../escape","url":"https://example.invalid/repo.git"}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_specs(specs)

    def test_adapter_boundary_rejects_shell_and_outside_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mirror"
            root.mkdir()
            entry = root / "run.py"
            entry.write_text("print('ok')", encoding="utf-8")
            shell = validate_adapter(AdapterSpec("x", root, entry, ("sh", "run.py")))
            outside = validate_adapter(AdapterSpec("x", root, Path(tmp) / "run.py", ("python", "run.py")))
            self.assertIn("shell_command_forbidden", shell)
            self.assertIn("entrypoint_outside_mirror", outside)

    def test_only_allowlisted_numeric_changes_reach_next_batch(self):
        current = {"tau_rare": 100.0, "outcome_draws": 100, "jitter_rare": 0.05}
        updated = _apply_changes(current, {"tau_rare": 400, "outcome_draws": 300, "shell": "bad"})
        self.assertEqual(updated, {"tau_rare": 400.0, "outcome_draws": 300, "jitter_rare": 0.05})

    def test_trial_summary_uses_statistical_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch" / "candidates"
            root.mkdir(parents=True)
            (root / "candidate_scores.csv").write_text(
                "validation_ok,utility\nTrue,0.80\nTrue,0.90\nTrue,0.99\nTrue,1.00\n",
                encoding="utf-8",
            )
            summary = _score_summary(root.parent)
            self.assertEqual(summary["count"], 4)
            self.assertAlmostEqual(float(summary["median"]), 0.945)

    def test_local_roles_are_sequential_and_reviewer_gates_changes(self):
        replies = [
            AgentReply("improver", "qwen", "", {"reason": "rare shrink", "changes": {"tau_rare": 400}, "next_experiment": "paired trial", "uncertainty": "U may fall"}),
            AgentReply("attacker", "qwen", "", {"reason": "check nn", "parent_attack_id": "mia_distance_k1", "family": "distance", "params": {"k": 3}}),
            AgentReply("reviewer", "qwen", "", {"decision": "accept", "reason": "bounded"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidates").mkdir()
            (root / "candidates" / "candidate_scores.csv").write_text("validation_ok,utility\nTrue,0.9\n", encoding="utf-8")
            store = LabStore(root / "lab.sqlite3")
            store.set_meta("last_audit", {"attacks": {"mia_distance_k1": {"a_mia": .1}}})
            with patch("sora_lab.runner.ask_ollama", side_effect=replies) as mocked:
                result = _local_review(store, batch=2, seeds=[1], model="qwen", host="http://local", output_dir=root, current_params={"tau_rare": 100})
            self.assertEqual(mocked.call_count, 3)
            self.assertEqual(result["accepted_changes"], {"tau_rare": 400})
            self.assertEqual(result["accepted_attack_spec"]["family"], "distance")
            self.assertIn("improver_proposal", mocked.call_args_list[1].kwargs["report"])
            self.assertIn("attacker_proposal", mocked.call_args_list[2].kwargs["report"])
            store.close()
