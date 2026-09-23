"""Small, deterministic bridge from attack findings to defense trials.

The bridge emits bounded RepairSpecs.  It never edits generator code or applies
an untested change; the runner places one proposal into its existing paired
comparison queue.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BreakthroughCase:
    case_id: str
    attack_id: str
    family: str
    batch: int
    strength: float
    rare_strength: float | None
    defense_params: Mapping[str, Any]
    status: str = "observed"


@dataclass(frozen=True)
class RepairSpec:
    repair_id: str
    hypothesis: str
    changes: Mapping[str, float | int]
    parent_case: str
    requires_paired_trial: bool = True


def _bounded(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def propose_repairs(attack_summary: Mapping[str, Mapping[str, Any]], *, batch: int,
                    current_params: Mapping[str, Any]) -> tuple[BreakthroughCase, ...]:
    """Convert a measured proxy breakthrough into at most two trial proposals."""
    cases: list[BreakthroughCase] = []
    for attack_id, row in attack_summary.items():
        if not isinstance(row, Mapping):
            continue
        aia_strength = float(row.get("aia_risk", 0.0) or 0.0)
        strength = float(row.get("a_mia", 0.0) or 0.0)
        rare = row.get("tpr_rare")
        rare_value = float(rare) if rare is not None else None
        if strength <= 0.05 and (rare_value is None or rare_value <= 0.05) and aia_strength <= 0.05:
            continue
        if attack_id == "_aia" or "aia" in attack_id.lower():
            family = "aia"
        else:
            family = "marginal" if "marginal" in attack_id else ("shadow" if "shadow" in attack_id else "distance")
        raw = f"{batch}:{attack_id}:{strength:.8f}:{aia_strength:.8f}:{rare_value}"
        case_id = "breakthrough_" + hashlib.sha256(raw.encode()).hexdigest()[:16]
        cases.append(BreakthroughCase(case_id, attack_id, family, batch, max(strength, aia_strength), rare_value,
                                      dict(current_params)))
    return tuple(cases[:2])


def repair_for_case(case: BreakthroughCase, current_params: Mapping[str, Any]) -> RepairSpec:
    """Return one allow-listed, conservative repair hypothesis."""
    changes: dict[str, float | int] = {}
    if case.family in {"distance", "marginal"}:
        changes["tau_rare"] = _bounded(float(current_params.get("tau_rare", 100.0)) * 1.5, 25.0, 400.0)
        changes["jitter_rare"] = _bounded(float(current_params.get("jitter_rare", 0.05)) * 1.25, 0.01, 0.15)
    elif case.family == "aia":
        changes["penalizer"] = _bounded(float(current_params.get("penalizer", 0.0)) + 0.005, 0.0, 0.02)
        changes["outcome_draws"] = int(_bounded(float(current_params.get("outcome_draws", 100.0)) * 0.8, 30.0, 1000.0))
        changes["jitter_rare"] = _bounded(float(current_params.get("jitter_rare", 0.05)) * 1.10, 0.01, 0.15)
    else:
        changes["penalizer"] = _bounded(float(current_params.get("penalizer", 0.0)) + 0.005, 0.0, 0.02)
        changes["jitter_rare"] = _bounded(float(current_params.get("jitter_rare", 0.05)) * 1.15, 0.01, 0.15)
    repair_id = "repair_" + hashlib.sha256((case.case_id + repr(sorted(changes.items()))).encode()).hexdigest()[:16]
    return RepairSpec(repair_id, f"reduce {case.family} leakage detected by {case.attack_id}", changes, case.case_id)
