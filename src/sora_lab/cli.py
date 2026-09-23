from __future__ import annotations

import argparse
import json
from pathlib import Path

from .advisor import AdvicePolicy, build_packet, consult_command, should_consult
from .intake import changed_paths, diff_excerpt, load_specs, make_candidate_record, update_mirror
from .agents import ask_ollama, validate_proposal
from .policy import pareto_front
from .runner import RunConfig, _ollama_models, evolve_member_holdout_attacks, member_holdout_audit, preflight, run_search
from .store import LabStore
from .jev import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sora-lab")
    parser.add_argument("--db", default="lab.sqlite3", help="外部run_root内の台帳SQLite")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    p_start = sub.add_parser("start")
    p_start.add_argument("--hours", type=float, required=True)
    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--session", default=None)
    p_llm = sub.add_parser("local-llm", help="実行中セッションのローカルQwenレビューを切り替える")
    llm_group = p_llm.add_mutually_exclusive_group(required=True)
    llm_group.add_argument("--enable", action="store_true")
    llm_group.add_argument("--disable", action="store_true")
    p_llm.add_argument("--model", default="qwen3.6:35b")
    p_llm.add_argument("--host", default="http://127.0.0.1:11434")
    p_resume = sub.add_parser("resume")
    p_resume.add_argument("--dist", required=True)
    p_resume.add_argument("--kit-dir", required=True)
    p_resume.add_argument("--run-root", required=True)
    p_resume.add_argument("--hours", type=float, default=None,
                          help="新しい時間枠。省略時は直前セッションの残り予算を引き継ぐ")
    p_resume.add_argument("--forever", action="store_true", help="停止要求まで無期限で再開する")
    p_resume.add_argument("--seed-start", type=int, default=None)
    p_resume.add_argument("--batch-size", type=int, default=4)
    p_resume.add_argument("--intake-specs")
    p_resume.add_argument("--mirror-root")
    p_resume.add_argument("--local-llm", action="store_true")
    p_resume.add_argument("--cloud-advice", action="store_true")
    p_resume.add_argument("--jev", action="store_true", help="Jev shadow判断を有効化（実行権限なし）")
    p_resume.add_argument("--audit-interval-batches", type=int, default=20)
    p_resume.add_argument("--audit-permutations", type=int, default=200)
    p_resume.add_argument("--attack-evolution-interval-batches", type=int, default=20)
    p_resume.add_argument("--attack-evolution-permutations", type=int, default=10)
    p_resume.add_argument("--attack-evolution-population-size", type=int, default=2)
    p_resume.add_argument("--attack-evolution-calibration-trials", type=int, default=32)
    p_resume.add_argument("--parallel-workers", type=int, default=1)
    p_resume.add_argument("--seed-chunk-size", type=int, default=4)
    p_resume.add_argument("--local-llm-interval-batches", type=int, default=50)
    p_exp = sub.add_parser("experiment")
    p_exp.add_argument("hypothesis")
    p_exp.add_argument("--config-json", default="{}")
    p_packet = sub.add_parser("advice-packet")
    p_packet.add_argument("--summary-json", required=True)
    p_consult = sub.add_parser("consult")
    p_consult.add_argument("--summary-json", required=True)
    p_consult.add_argument("--elapsed-hours", type=float, required=True)
    p_consult.add_argument("--needs-decision", action="store_true")
    p_preflight = sub.add_parser("preflight")
    p_preflight.add_argument("--dist", required=True)
    p_preflight.add_argument("--kit-dir", required=True)
    p_preflight.add_argument("--require-ollama", action="store_true")
    p_run = sub.add_parser("run")
    p_run.add_argument("--dist", required=True)
    p_run.add_argument("--kit-dir", required=True)
    p_run.add_argument("--run-root", required=True)
    p_run.add_argument("--hours", type=float, default=1.0)
    p_run.add_argument("--forever", action="store_true", help="停止要求まで無期限で実行する（--hours=0相当）")
    p_run.add_argument("--seeds", default="",
                       help="固定seed列（省略時はSQLite永続範囲リースで重複なく割当）")
    p_run.add_argument("--seed-start", type=int, default=0)
    p_run.add_argument("--batch-size", type=int, default=4)
    p_run.add_argument("--intake-specs", help="チームリポジトリspec JSON。指定時は外部mirrorを定期更新")
    p_run.add_argument("--mirror-root", help="チームリポジトリの外部mirror保存先")
    p_run.add_argument("--intake-interval-minutes", type=float, default=60.0)
    p_run.add_argument("--local-llm", action="store_true", help="許可した間隔でOllamaの3役レビューを行う")
    p_run.add_argument("--cloud-advice", action="store_true", help="互換引数。自動Astra呼出しは行わない")
    p_run.add_argument("--jev", action="store_true", help="Jev shadow判断を有効化（実行権限なし）")
    p_run.add_argument("--outcome-draws", type=int, default=100)
    p_run.add_argument("--outcome-step", type=float, default=0.10)
    p_run.add_argument("--require-ollama", action="store_true")
    p_run.add_argument("--audit-interval-batches", type=int, default=20, help="proxy MIAを実行する間隔。0で無効")
    p_run.add_argument("--audit-permutations", type=int, default=200)
    p_run.add_argument("--attack-evolution-interval-batches", type=int, default=20, help="攻撃進化を実行する間隔。0で無効")
    p_run.add_argument("--attack-evolution-permutations", type=int, default=10)
    p_run.add_argument("--attack-evolution-population-size", type=int, default=2)
    p_run.add_argument("--attack-evolution-calibration-trials", type=int, default=32)
    p_run.add_argument("--parallel-workers", type=int, default=1)
    p_run.add_argument("--seed-chunk-size", type=int, default=4)
    p_run.add_argument("--local-llm-interval-batches", type=int, default=50)
    p_audit = sub.add_parser("audit-split")
    p_audit.add_argument("--dist", required=True)
    p_audit.add_argument("--kit-dir", required=True)
    p_audit.add_argument("--seed", type=int, default=0)
    p_audit.add_argument("--train-fraction", type=float, default=0.8)
    p_audit.add_argument("--permutations", type=int, default=200)
    p_attack = sub.add_parser("attack-evolve", help="固定したmember-fit世界で攻撃仕様を1世代進化させる")
    p_attack.add_argument("--dist", required=True)
    p_attack.add_argument("--kit-dir", required=True)
    p_attack.add_argument("--seed", type=int, default=0)
    p_attack.add_argument("--train-fraction", type=float, default=0.8)
    p_attack.add_argument("--permutations", type=int, default=100)
    p_attack.add_argument("--population-size", type=int, default=16)
    p_attack.add_argument("--generation", type=int, default=1)
    p_attack.add_argument("--archive")
    p_attack.add_argument("--calibration-trials", type=int, default=32,
                          help="同一world設計でpaired昇格判定に使う試行数 (1-128)")
    p_intake = sub.add_parser("intake", help="チームリポジトリを外部mirrorへ取得し差分を台帳化")
    p_intake.add_argument("--specs", required=True, help="[{name,url,branch}] JSON")
    p_intake.add_argument("--mirror-root", required=True)
    p_intake.add_argument("--local-llm", action="store_true", help="差分をローカルreviewerでトリアージ")
    p_intake.add_argument("--model", default="qwen3.6:35b")
    p_jev_report = sub.add_parser("jev-report", help="Jev shadow呼出しの接続・費用・遅延・判断分布を表示")
    p_jev_report.add_argument("--since-hours", type=float, default=None)

    args = parser.parse_args(argv)
    store = LabStore(Path(args.db))
    try:
        if args.command == "status":
            print(json.dumps(store.summary(), ensure_ascii=False, indent=2, default=str))
            return 0
        if args.command == "start":
            sid = store.start_session(round(args.hours * 3600))
            print(json.dumps({"session_id": sid, "status": "running"}, ensure_ascii=False))
            return 0
        if args.command == "stop":
            session = store.session(args.session)
            if not session:
                raise SystemExit("session not found")
            store.request_stop(session["session_id"])
            print(json.dumps({**store.session(session["session_id"]), "stop_requested": True}, ensure_ascii=False, default=str))
            return 0
        if args.command == "local-llm":
            enabled = bool(args.enable)
            store.set_meta("local_llm_enabled", enabled)
            models, error = [], None
            try:
                models = _ollama_models(args.host)
            except RuntimeError as exc:
                error = str(exc)
            ready = not enabled or args.model in models
            store.event("local_llm_toggle", {"enabled": enabled, "requested_model": args.model,
                                              "ready": ready, "available_models": models,
                                              "error": error})
            print(json.dumps({"local_llm_enabled": enabled, "requested_model": args.model,
                              "ready": ready, "available_models": models, "error": error}, ensure_ascii=False))
            return 0
        if args.command == "experiment":
            eid = store.add_experiment(hypothesis=args.hypothesis, config=json.loads(args.config_json))
            print(json.dumps({"experiment_id": eid, "state": "queued"}, ensure_ascii=False))
            return 0
        if args.command == "preflight":
            config = RunConfig(dist=Path(args.dist), kit_dir=Path(args.kit_dir), run_root=Path("."))
            result = preflight(config, require_ollama=args.require_ollama)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 2
        if args.command == "run":
            hours = 0.0 if args.forever else args.hours
            config = RunConfig(dist=Path(args.dist), kit_dir=Path(args.kit_dir), run_root=Path(args.run_root), hours=hours, seeds=args.seeds, seed_start=args.seed_start, batch_size=args.batch_size, outcome_draws=args.outcome_draws, outcome_step=args.outcome_step, intake_specs=Path(args.intake_specs) if args.intake_specs else None, mirror_root=Path(args.mirror_root) if args.mirror_root else None, intake_interval_seconds=args.intake_interval_minutes * 60, local_llm=args.local_llm, cloud_advice=args.cloud_advice, jev=args.jev, audit_interval_batches=args.audit_interval_batches, audit_permutations=args.audit_permutations, attack_evolution_interval_batches=args.attack_evolution_interval_batches, attack_evolution_permutations=args.attack_evolution_permutations, attack_evolution_population_size=args.attack_evolution_population_size, attack_evolution_calibration_trials=args.attack_evolution_calibration_trials, parallel_workers=args.parallel_workers, seed_chunk_size=args.seed_chunk_size, local_llm_interval_batches=args.local_llm_interval_batches)
            result = run_search(store, config, require_ollama=args.require_ollama)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0 if result.get("returncode") == 0 else 2
        if args.command == "resume":
            start = args.seed_start if args.seed_start is not None else store.next_seed()
            if args.forever:
                hours = 0.0
            elif args.hours is None:
                previous = store.session()
                remaining = (float(previous["budget_seconds"]) - float(previous["accumulated_seconds"])) / 3600.0 if previous else 0.0
                if remaining <= 0:
                    raise SystemExit("直前セッションに残り予算がありません。--hoursで新しい枠を指定してください")
                hours = remaining
            else:
                hours = args.hours
            config = RunConfig(dist=Path(args.dist), kit_dir=Path(args.kit_dir), run_root=Path(args.run_root), hours=hours, seed_start=start, batch_size=args.batch_size, intake_specs=Path(args.intake_specs) if args.intake_specs else None, mirror_root=Path(args.mirror_root) if args.mirror_root else None, local_llm=args.local_llm, cloud_advice=args.cloud_advice, jev=args.jev, audit_interval_batches=args.audit_interval_batches, audit_permutations=args.audit_permutations, attack_evolution_interval_batches=args.attack_evolution_interval_batches, attack_evolution_permutations=args.attack_evolution_permutations, attack_evolution_population_size=args.attack_evolution_population_size, attack_evolution_calibration_trials=args.attack_evolution_calibration_trials, parallel_workers=args.parallel_workers, seed_chunk_size=args.seed_chunk_size, local_llm_interval_batches=args.local_llm_interval_batches)
            result = run_search(store, config)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0 if result.get("returncode") == 0 else 2
        if args.command == "audit-split":
            result = member_holdout_audit(Path(args.dist), Path(args.kit_dir), seed=args.seed, train_fraction=args.train_fraction, permutations=args.permutations)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
            return 0
        if args.command == "attack-evolve":
            result = evolve_member_holdout_attacks(
                Path(args.dist), Path(args.kit_dir), seed=args.seed,
                train_fraction=args.train_fraction, permutations=args.permutations,
                population_size=args.population_size, generation=args.generation,
                calibration_trials=args.calibration_trials,
                archive_path=Path(args.archive) if args.archive else None,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
            return 0
        if args.command == "intake":
            updates = []
            for spec in load_specs(args.specs):
                update = update_mirror(spec, args.mirror_root)
                paths = changed_paths(update["mirror"], update["previous_sha"], update["sha"])
                diff = diff_excerpt(update["mirror"], update["previous_sha"], update["sha"])
                record = make_candidate_record(update, paths, diff)
                added = store.add_intake_candidate(record)
                triage = None
                if args.local_llm and added:
                    try:
                        reply = ask_ollama(role="reviewer", model=args.model, host="http://127.0.0.1:11434",
                                           report={"source_repo": update["name"], "commit": update["sha"],
                                                   "changed_paths": paths, "diff_excerpt": diff[:6000],
                                                   "instruction": "候補の比較実験の価値と前提差だけを評価。コード実行や秘密値を要求しない。"})
                        valid, reasons = validate_proposal(reply)
                        triage = {"valid": valid, "reasons": reasons, "parsed": reply.parsed}
                        store.update_intake_candidate(record["candidate_id"], state="triaged" if valid else "needs_adapter", details={"triage": triage})
                    except Exception as exc:  # noqa: BLE001
                        triage = {"valid": False, "reasons": [f"unavailable:{type(exc).__name__}"]}
                        store.update_intake_candidate(record["candidate_id"], state="discovered", details={"triage": triage})
                updates.append({**update, "candidate_id": record["candidate_id"], "new_candidate": added, "paths": paths, "triage": triage})
            print(json.dumps({"updates": updates}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "jev-report":
            print(json.dumps(build_report(store, since_hours=args.since_hours), ensure_ascii=False, indent=2))
            return 0
        summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
        packet = build_packet(summary)
        if args.command == "advice-packet":
            print(json.dumps(packet, ensure_ascii=False, indent=2))
            return 0
        policy = AdvicePolicy()
        elapsed = args.elapsed_hours * 3600
        if not args.needs_decision or not should_consult(store, accumulated_seconds=elapsed, policy=policy, needs_decision=True):
            print(json.dumps({"status": "skipped", "reason": "not_due_or_no_decision_needed"}, ensure_ascii=False))
            return 0
        print(json.dumps(consult_command(store, accumulated_seconds=elapsed, policy=policy, packet=packet), ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
