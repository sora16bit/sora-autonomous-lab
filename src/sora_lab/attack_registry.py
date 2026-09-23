"""Allow-listed attack specifications used by the attack co-evolution loop.

An AttackSpec is data only.  It cannot contain Python, shell, paths, or an
arbitrary callable.  The evaluator maps its family and parameters to a known
implementation in ``sora_synth.audit``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


NUMERIC_LIMITS = {
    "k": (1, 25),
    "max_iter": (20, 500),
    "max_leaf_nodes": (3, 64),
    "l2_regularization": (0.0, 100.0),
    "temperature": (0.0, 2.0),
}
ALLOWED_FAMILIES = {"distance", "blocked_distance", "marginal", "shadow", "ensemble", "aia_knn", "aia_tree"}
ALLOWED_COLUMNS = {
    "age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "onset", "death",
    "sex", "prefecture", "smoking",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class AttackSpec:
    attack_id: str
    family: str
    params: Mapping[str, Any] = field(default_factory=dict)
    parents: tuple[str, ...] = ()
    generation: int = 0

    def validate(self) -> None:
        if not self.attack_id or len(self.attack_id) > 120:
            raise ValueError("attack_id must be non-empty and bounded")
        if self.family not in ALLOWED_FAMILIES:
            raise ValueError(f"unsupported attack family: {self.family}")
        for key, value in self.params.items():
            if key in {"columns", "children"}:
                if not isinstance(value, (list, tuple)) or not value:
                    raise ValueError(f"{key} must be a non-empty list")
                if key == "columns" and any(str(column) not in ALLOWED_COLUMNS for column in value):
                    raise ValueError("attack columns contain an unallowlisted feature")
                if key == "children" and any(not isinstance(child, str) for child in value):
                    raise ValueError("ensemble children must be attack ids")
                continue
            if key not in NUMERIC_LIMITS:
                raise ValueError(f"unsupported attack parameter: {key}")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"attack parameter {key} must be numeric") from exc
            lo, hi = NUMERIC_LIMITS[key]
            if not lo <= numeric <= hi:
                raise ValueError(f"attack parameter {key} out of range")
            if key in {"k", "max_iter", "max_leaf_nodes"} and int(numeric) != numeric:
                raise ValueError(f"attack parameter {key} must be integral")
        if len(self.parents) > 4 or any(not isinstance(parent, str) for parent in self.parents):
            raise ValueError("attack parents are bounded strings")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"attack_id": self.attack_id, "family": self.family,
                "params": dict(self.params), "parents": list(self.parents),
                "generation": int(self.generation)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttackSpec":
        spec = cls(str(value["attack_id"]), str(value["family"]),
                   dict(value.get("params", {})), tuple(value.get("parents", ())),
                   int(value.get("generation", 0)))
        spec.validate()
        return spec

    @property
    def semantic_hash(self) -> str:
        """Hash the executable attack meaning, excluding lineage and labels."""
        params = dict(self.params)
        if isinstance(params.get("columns"), (list, tuple)):
            params["columns"] = sorted(str(column) for column in params["columns"])
        payload = {"family": self.family, "params": params}
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    @property
    def lineage_id(self) -> str:
        payload = {"semantic_hash": self.semantic_hash, "parents": sorted(self.parents),
                   "generation": int(self.generation)}
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    @property
    def digest(self) -> str:
        """Backward-compatible alias for the semantic attack identity."""
        return self.semantic_hash


def default_attack_specs() -> tuple[AttackSpec, ...]:
    """Regression attacks that must remain in every active suite."""
    return (
        AttackSpec("mia_distance_k1", "distance", {"k": 1}),
        AttackSpec("mia_distance_k5", "distance", {"k": 5}),
        AttackSpec("mia_blocked_k1", "blocked_distance", {"k": 1}),
        AttackSpec("mia_marginal_all", "marginal", {"columns": ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]}),
        AttackSpec("mia_shadow_tree", "shadow", {"max_iter": 160, "max_leaf_nodes": 15, "l2_regularization": 1.0}),
        AttackSpec("aia_knn1", "aia_knn", {"k": 1}),
        AttackSpec("aia_knn3", "aia_knn", {"k": 3}),
        AttackSpec("aia_knn5", "aia_knn", {"k": 5}),
        AttackSpec("aia_regularized_tree", "aia_tree", {"max_iter": 120, "max_leaf_nodes": 15, "l2_regularization": 1.0}),
    )
