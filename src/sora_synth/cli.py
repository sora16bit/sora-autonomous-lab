from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .artifacts import write_candidate, write_manifest
from .audit import audit_c_to_abg, evaluate_aia, evaluate_aia_suite, evaluate_mia
from .data import load_dataset, validate_frame
from .generator import fit_model
from .official import official_score, validate_with_kit
from .search import choose_by_utility, generate_candidates


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dist", required=True, help="本戦配布物のローカルフォルダ")
    parser.add_argument("--kit-dir", required=True, help="main-process-20260912キットのルート")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sora-synth")
    sub = ap.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect")
    _common(p_inspect)

    p_gen = sub.add_parser("generate")
    _common(p_gen)
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--tau-general", type=float, default=100.0)
    p_gen.add_argument("--tau-rare", type=float, default=100.0)
    p_gen.add_argument("--rare-multiplier", type=float, default=1.0)
    p_gen.add_argument("--jitter-general", type=float, default=0.02)
    p_gen.add_argument("--jitter-rare", type=float, default=0.05)
    p_gen.add_argument("--penalizer", type=float, default=0.0)
    p_gen.add_argument("--rng-mode", choices=("random", "sobol"), default="random")

    p_score = sub.add_parser("score")
    _common(p_score)
    p_score.add_argument("--csv", required=True)

    p_audit = sub.add_parser("audit-mia", help="member/nonmemberを分けてMIA自己監査")
    _common(p_audit)
    p_audit.add_argument("--csv", required=True)
    p_audit.add_argument("--member", required=True, help="C生成に使ったmember側CSV")
    p_audit.add_argument("--nonmember", required=True, help="C生成から除外したnonmember側CSV")
    p_audit.add_argument("--reference", required=True, help="評価対象と重複しない攻撃reference CSV")
    p_audit.add_argument("--permutations", type=int, default=500)
    p_audit.add_argument("--seed", type=int, default=0)

    p_aia = sub.add_parser("audit-aia", help="AIAのtarget/control自己監査")
    _common(p_aia)
    p_aia.add_argument("--csv", required=True, help="提出候補C")
    p_aia.add_argument("--target", required=True, help="正解target行（timeを含む）")
    p_aia.add_argument("--control", required=True, help="正解control行（timeを含む）")
    p_aia.add_argument("--tau", type=float, default=0.5)

    p_diag = sub.add_parser("diagnose", help="C対A_bgの補助診断。公式Anonとは別物")
    _common(p_diag)
    p_diag.add_argument("--csv", required=True)

    p_search = sub.add_parser("search")
    _common(p_search)
    p_search.add_argument("--out-dir", required=True)
    p_search.add_argument("--seeds", default="0,1,2,3,4,5")
    p_search.add_argument("--tau-general", type=float, default=100.0)
    p_search.add_argument("--tau-rare", type=float, default=100.0)
    p_search.add_argument("--rare-multiplier", type=float, default=1.0)
    p_search.add_argument("--jitter-general", type=float, default=0.02)
    p_search.add_argument("--jitter-rare", type=float, default=0.05)
    p_search.add_argument("--penalizer", type=float, default=0.0)
    p_search.add_argument("--outcome-draws", type=int, default=100, help="転帰の局所探索回数。固定共変量上で公式U_specを高速評価する")
    p_search.add_argument("--outcome-step", type=float, default=0.10)
    p_search.add_argument("--rng-mode", choices=("random", "sobol"), default="random")

    args = ap.parse_args(argv)
    dataset = load_dataset(args.dist, args.kit_dir)
    if args.command == "inspect":
        print(json.dumps({"n_rows": len(dataset.b), "horizon": dataset.horizon, "input_hashes": dataset.hash_inputs(), "rare_truth": int(dataset.rare_truth.sum()), "b_columns": list(dataset.b.columns)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "score":
        c = pd.read_csv(args.csv, float_precision="round_trip")
        errors = validate_with_kit(c, dataset, args.kit_dir)
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(official_score(c, dataset, args.kit_dir), ensure_ascii=False, indent=2, default=float))
        return 0
    if args.command == "audit-mia":
        c = pd.read_csv(args.csv, float_precision="round_trip")
        member = pd.read_csv(args.member, float_precision="round_trip")
        nonmember = pd.read_csv(args.nonmember, float_precision="round_trip")
        reference = pd.read_csv(args.reference, float_precision="round_trip")
        print(json.dumps({"kind": "member_nonmember_proxy", "results": evaluate_mia(c, member, nonmember, reference=reference, rare_ref=dataset.utility_ref, n_perm=args.permutations, seed=args.seed)}, ensure_ascii=False, indent=2, default=float))
        return 0
    if args.command == "audit-aia":
        c = pd.read_csv(args.csv, float_precision="round_trip")
        target = pd.read_csv(args.target, float_precision="round_trip")
        control = pd.read_csv(args.control, float_precision="round_trip")
        print(json.dumps({"kind": "aia_proxy", "results": evaluate_aia_suite(c, target, control, tau=args.tau, seed=0)}, ensure_ascii=False, indent=2, default=float))
        return 0
    if args.command == "diagnose":
        c = pd.read_csv(args.csv, float_precision="round_trip")
        result = audit_c_to_abg(c, dataset.a_bg, seed=0)
        print(json.dumps({"kind": result.status.get("kind"), "values": result.values, "status": result.status}, ensure_ascii=False, indent=2, default=float))
        return 0
    params = dict(tau_general=args.tau_general, tau_rare=args.tau_rare, rare_multiplier=args.rare_multiplier, jitter_general=args.jitter_general, jitter_rare=args.jitter_rare, penalizer=args.penalizer, rng_mode=args.rng_mode)
    model = fit_model(dataset, **params)
    if args.command == "generate":
        c = model.sample(args.seed)
        errors = validate_with_kit(c, dataset, args.kit_dir)
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 2
        c_hash = write_candidate(c, args.out)
        score = official_score(c, dataset, args.kit_dir)
        write_manifest(Path(args.out).with_suffix(".manifest.json"), dataset=dataset, kit_dir=args.kit_dir, c=c, seed=args.seed, model_params=params, official_score=score)
        print(json.dumps({"ok": True, "csv": str(args.out), "sha256": c_hash, "score": score}, ensure_ascii=False, indent=2, default=float))
        return 0
    if args.command == "search":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        scores = generate_candidates(model, dataset, args.kit_dir, args.out_dir, seeds, outcome_draws=args.outcome_draws, outcome_step=args.outcome_step)
        print(scores.to_string(index=False))
        if scores["validation_ok"].any():
            pick = choose_by_utility(scores)
            print(f"recommended={pick['candidate_id']} U={pick['utility']:.6f}")
            return 0
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
