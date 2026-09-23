"""Bounded active set and durable metadata for evolving attacks."""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .attack_registry import AttackSpec


@dataclass(frozen=True)
class AttackEvaluation:
    spec: AttackSpec
    strength: float
    novelty: float
    cost_seconds: float
    status: str = "complete"
    split_id: str = ""
    evidence: dict[str, Any] | None = None

    @property
    def evaluation_id(self) -> str:
        payload = {"semantic_hash": self.spec.semantic_hash, "split_id": self.split_id,
                   "status": self.status, "evidence": self.evidence or {}}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), default=str).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(), "strength": self.strength,
                "novelty": self.novelty, "cost_seconds": self.cost_seconds,
                "status": self.status, "split_id": self.split_id,
                "evidence": self.evidence or {}}


class AttackArchive:
    """Keep all metadata while bounding the attacks run each cycle."""

    def __init__(self, *, active_limit: int = 16, challenger_limit: int = 8,
                 regression_ids: Iterable[str] = ()):
        if active_limit < 1 or challenger_limit < 0:
            raise ValueError("attack archive limits must be positive")
        self.active_limit = active_limit
        self.challenger_limit = challenger_limit
        self.regression_ids = set(regression_ids)
        self.history: dict[str, AttackEvaluation] = {}
        self.active: list[str] = []
        self.challengers: list[str] = []

    def add(self, evaluation: AttackEvaluation) -> None:
        if not 0.0 <= evaluation.strength <= 1.0:
            raise ValueError("attack strength must be in [0,1]")
        if evaluation.cost_seconds < 0:
            raise ValueError("attack cost must be non-negative")
        self.history[evaluation.evaluation_id] = evaluation

    def promote(self, evaluations: Iterable[AttackEvaluation], *,
                pool: Iterable[AttackEvaluation] | None = None) -> list[AttackEvaluation]:
        for evaluation in evaluations:
            self.add(evaluation)
        values = list(pool) if pool is not None else list(self.history.values())
        values = [value for value in values if value.status == "complete"]
        values.sort(key=lambda x: (x.strength, x.novelty, -x.cost_seconds), reverse=True)
        selected: list[AttackEvaluation] = []
        for value in values:
            if value.spec.attack_id in self.regression_ids:
                selected.append(value)
        for value in values:
            if value in selected:
                continue
            selected.append(value)
            if len(selected) >= self.active_limit:
                break
        selected = selected[:self.active_limit]
        self.active = [value.evaluation_id for value in selected]
        active_ids = set(self.active)
        self.challengers = [value.evaluation_id for value in values
                            if value.evaluation_id not in active_ids][:self.challenger_limit]
        return selected

    def active_specs(self) -> list[AttackSpec]:
        return [self.history[evaluation_id].spec for evaluation_id in self.active
                if evaluation_id in self.history]

    def save(self, path: str | Path) -> None:
        payload = {"active_limit": self.active_limit, "challenger_limit": self.challenger_limit,
                   "regression_ids": sorted(self.regression_ids), "active": self.active,
                   "challengers": self.challengers,
                   "history": [value.to_dict() for value in self.history.values()]}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "AttackArchive":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        archive = cls(active_limit=int(payload["active_limit"]),
                      challenger_limit=int(payload["challenger_limit"]),
                      regression_ids=payload.get("regression_ids", ()))
        for raw in payload.get("history", ()):
            spec = AttackSpec.from_dict(raw["spec"])
            archive.add(AttackEvaluation(spec, float(raw["strength"]), float(raw["novelty"]),
                                         float(raw["cost_seconds"]), str(raw.get("status", "complete")),
                                         str(raw.get("split_id", "")), dict(raw.get("evidence", {}))))
        saved_active = list(payload.get("active", ()))
        saved_challengers = list(payload.get("challengers", ()))
        # Archives written before evaluation_id existed stored semantic digests.
        digest_to_ids = {}
        for evaluation_id, evaluation in archive.history.items():
            digest_to_ids.setdefault(evaluation.spec.digest, []).append(evaluation_id)
        archive.active = [item if item in archive.history else digest_to_ids[item][-1]
                          for item in saved_active if item in archive.history or item in digest_to_ids]
        archive.challengers = [item if item in archive.history else digest_to_ids[item][-1]
                               for item in saved_challengers if item in archive.history or item in digest_to_ids]
        return archive
