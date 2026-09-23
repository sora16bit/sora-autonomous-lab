from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_candidate
from .audit import audit_c_to_abg
from .data import Dataset
from .generator import SoraModel
from .official import official_score, validate_with_kit


def _spec_score(c: pd.DataFrame, dataset: Dataset, kit_dir: str | Path) -> float:
    """公式U_specだけを高速評価する。転帰局所探索の内側ループ用。"""
    import sys

    src = str(Path(kit_dir) / "starter" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pwscup2026.scoring.utility import _score_u_spec

    return float(_score_u_spec(c, dataset.b)[0])


@dataclass
class CandidateResult:
    candidate_id: str
    seed: int
    utility: float
    facets: dict[str, float]
    validation_errors: list[str]
    audit: dict[str, float]
    c_path: str


def _outcome_local_search(model: SoraModel, covars: pd.DataFrame, dataset: Dataset, kit_dir: str | Path, *, seed: int, draws: int, step: float) -> tuple[pd.DataFrame, dict]:
    """固定共変量上で転帰を引き直し、公式U_specを天井まで局所探索する。

    共変量は固定されるため、各試行で重いU全体を再計算しない。最後にだけ公式全facetを
    採点し、U_rare/U_validの悪化を確認する。
    """
    # The initial uniforms use the same seed as model.sample(seed), so enabling
    # local search with zero draws is a reproducible no-op.
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random(len(covars)), rng.random(len(covars))
    best = model.sample_with_uniforms(covars, u1, u2)
    ceiling_score = official_score(best, dataset, kit_dir)
    ceiling = min(ceiling_score["facets"][k] for k in ("U_gen", "U_rare", "U_valid"))
    best_spec = _spec_score(best, dataset, kit_dir)
    accepted = 0
    used = 0
    while used < max(0, draws) and best_spec < ceiling:
        used += 1
        mask = rng.random(len(covars)) < step
        trial_u1, trial_u2 = u1.copy(), u2.copy()
        trial_u1[mask], trial_u2[mask] = rng.random(int(mask.sum())), rng.random(int(mask.sum()))
        trial = model.sample_with_uniforms(covars, trial_u1, trial_u2)
        try:
            spec = _spec_score(trial, dataset, kit_dir)
        except Exception:  # noqa: BLE001
            continue
        if spec > best_spec:
            best, best_spec, u1, u2 = trial, spec, trial_u1, trial_u2
            accepted += 1
    errors = validate_with_kit(best, dataset, kit_dir)
    if errors:
        raise ValueError("転帰探索後の候補が公式validatorに失敗しました: " + "; ".join(errors))
    best_score = official_score(best, dataset, kit_dir)
    return best, {
        "outcome_draws": used,
        "outcome_budget": draws,
        "outcome_step": step,
        "outcome_accepted": accepted,
        "outcome_ceiling": ceiling,
        "outcome_spec_search": best_spec,
        "official_score": best_score,
    }


def generate_candidates(model: SoraModel, dataset: Dataset, kit_dir: str | Path, out_dir: str | Path, seeds: list[int], *, outcome_draws: int = 0, outcome_step: float = 0.10) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        covars = model.sample_covariates(seed)
        c = model.sample(seed) if outcome_draws <= 0 else None
        outcome_meta = {}
        if outcome_draws > 0:
            c, outcome_meta = _outcome_local_search(model, covars, dataset, kit_dir, seed=seed, draws=outcome_draws, step=outcome_step)
        errors = validate_with_kit(c, dataset, kit_dir)
        row = {"candidate_id": f"seed_{seed}", "seed": seed, "validation_ok": not errors, "validation_errors": " | ".join(errors), **{k: v for k, v in outcome_meta.items() if k != "official_score"}}
        if not errors:
            score = outcome_meta.get("official_score") or official_score(c, dataset, kit_dir)
            row.update({"utility": score["utility"], **{f"facet_{k}": v for k, v in score["facets"].items()}})
            diag = audit_c_to_abg(c, dataset.a_bg, seed=seed)
            row.update({f"diag_{k}": v for k, v in diag.values.items()})
            path = out / f"C_seed{seed}.csv"
            write_candidate(c, path)
            row["c_path"] = str(path)
        else:
            row["utility"] = np.nan
            row["c_path"] = ""
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["validation_ok", "utility"], ascending=[False, False])
    result.to_csv(out / "candidate_scores.csv", index=False)
    return result


def choose_by_utility(scores: pd.DataFrame, *, min_utility: float | None = None) -> pd.Series:
    valid = scores.loc[scores["validation_ok"].astype(bool)].copy()
    if min_utility is not None:
        valid = valid[valid["utility"] >= min_utility]
    if valid.empty:
        raise ValueError("公式検証に通った採用可能候補がありません")
    return valid.sort_values("utility", ascending=False).iloc[0]
