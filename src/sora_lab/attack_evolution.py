"""Deterministic, bounded attack-spec mutation and survivor selection."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable

from .attack_archive import AttackArchive, AttackEvaluation
from .attack_registry import AttackSpec, default_attack_specs


@dataclass(frozen=True)
class EvolutionResult:
    generation: int
    evaluations: tuple[AttackEvaluation, ...]
    active: tuple[AttackSpec, ...]
    challengers: tuple[AttackSpec, ...]


def _mutate(spec: AttackSpec, rng: random.Random, generation: int, index: int) -> AttackSpec:
    params = dict(spec.params)
    if spec.family in {"distance", "blocked_distance", "aia_knn"}:
        params["k"] = max(1, min(25, int(params.get("k", 1)) + rng.choice((-2, -1, 1, 2))))
    elif spec.family == "shadow":
        params["max_leaf_nodes"] = max(3, min(64, int(params.get("max_leaf_nodes", 15)) + rng.choice((-4, -2, 2, 4))))
        params["l2_regularization"] = max(0.0, min(100.0, float(params.get("l2_regularization", 1.0)) * rng.choice((0.5, 1.0, 2.0))))
    elif spec.family == "marginal":
        columns = list(params.get("columns", ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")))
        if len(columns) > 3 and rng.random() < 0.7:
            columns.pop(rng.randrange(len(columns)))
        params["columns"] = columns
    elif spec.family == "aia_tree":
        params["max_iter"] = max(20, min(500, int(params.get("max_iter", 120)) + rng.choice((-40, -20, 20, 40))))
        params["max_leaf_nodes"] = max(3, min(64, int(params.get("max_leaf_nodes", 15)) + rng.choice((-4, -2, 2, 4))))
        params["l2_regularization"] = max(0.0, min(100.0, float(params.get("l2_regularization", 1.0)) * rng.choice((0.5, 1.0, 2.0))))
    return AttackSpec(f"g{generation}_{spec.attack_id}_m{index}", spec.family, params,
                      parents=(spec.attack_id,), generation=generation)


def _crossover(left: AttackSpec, right: AttackSpec, generation: int, index: int) -> AttackSpec | None:
    if left.family != right.family:
        return None
    params = dict(left.params)
    for key, value in right.params.items():
        if index % 2 == 0 or key not in params:
            params[key] = value
    return AttackSpec(f"g{generation}_{left.attack_id}_x{right.attack_id}_{index}", left.family, params,
                      parents=(left.attack_id, right.attack_id), generation=generation)


def evolve_attack_specs(parents: Iterable[AttackSpec] | None = None, *, generation: int = 1,
                        seed: int = 0, population_size: int = 16) -> list[AttackSpec]:
    """Create one bounded generation; no arbitrary code is generated."""
    if population_size < 1 or population_size > 64:
        raise ValueError("population_size must be in [1,64]")
    base = list(parents or default_attack_specs())
    for spec in base:
        spec.validate()
    rng = random.Random(seed)
    children: list[AttackSpec] = []
    index = 0
    while len(children) < population_size:
        parent = base[index % len(base)]
        children.append(_mutate(parent, rng, generation, index))
        index += 1
        if len(base) > 1 and len(children) < population_size and rng.random() < 0.5:
            other = base[rng.randrange(len(base))]
            child = _crossover(parent, other, generation, index)
            if child is not None:
                children.append(child)
                index += 1
    return children[:population_size]


def run_attack_generation(parents: Iterable[AttackSpec], evaluator: Callable[[AttackSpec], AttackEvaluation],
                          *, generation: int, seed: int, population_size: int = 16,
                          active_limit: int = 16, challenger_limit: int = 8,
                          regression_ids: Iterable[str] = (), archive: AttackArchive | None = None,
                          additional_specs: Iterable[AttackSpec] = ()) -> EvolutionResult:
    candidates = list(parents) + evolve_attack_specs(list(parents), generation=generation,
                                                     seed=seed, population_size=population_size)
    seen = {item.semantic_hash for item in candidates}
    for spec in additional_specs:
        spec.validate()
        if spec.semantic_hash not in seen:
            candidates.append(spec)
            seen.add(spec.semantic_hash)
    evaluations_list: list[AttackEvaluation] = []
    for spec in candidates:
        try:
            evaluations_list.append(evaluator(spec))
        except Exception as exc:  # noqa: BLE001
            evaluations_list.append(AttackEvaluation(
                spec=spec, strength=0.0, novelty=0.0, cost_seconds=0.0,
                status="failed", evidence={"error": f"{type(exc).__name__}: {exc}"},
            ))
    evaluations = tuple(evaluations_list)
    archive = archive or AttackArchive(active_limit=active_limit, challenger_limit=challenger_limit,
                                       regression_ids=regression_ids)
    if regression_ids:
        archive.regression_ids.update(regression_ids)
    active = archive.promote(evaluations, pool=evaluations)
    active_ids = {item.evaluation_id for item in active}
    challengers = [item for item in evaluations if item.evaluation_id not in active_ids and item.status == "complete"]
    challengers.sort(key=lambda item: (item.strength, item.novelty, -item.cost_seconds), reverse=True)
    return EvolutionResult(generation, evaluations, tuple(item.spec for item in active),
                           tuple(item.spec for item in challengers[:challenger_limit]))
