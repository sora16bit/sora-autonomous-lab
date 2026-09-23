"""Adaptive worker allocation for attack, audit, seed, and generator queues."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


QUEUES = ("attack", "audit", "seed", "generator")
INITIAL_TARGETS = {"attack": 0.30, "audit": 0.30, "seed": 0.25, "generator": 0.15}
FLOORS = {"attack": 0.20, "audit": 0.25, "seed": 0.10, "generator": 0.10}
CEILINGS = {"attack": 0.45, "audit": 0.50, "seed": 0.45, "generator": 0.30}


@dataclass(frozen=True)
class QueueStat:
    completed: int = 0
    reward_sum: float = 0.0
    worker_seconds: float = 0.0


def allocate_resources(stats: Mapping[str, QueueStat], *, previous: Mapping[str, float] | None = None,
                       audit_backlog_age_minutes: float = 0.0, pending_campaigns: int = 0) -> dict[str, float]:
    if any(name not in stats for name in QUEUES):
        raise ValueError("stats must include every worker queue")
    if all(stats[name].completed < 5 for name in QUEUES):
        result = dict(INITIAL_TARGETS)
    else:
        rates = {name: max(0.0, stats[name].reward_sum) / max(stats[name].worker_seconds, 1e-9)
                 if stats[name].completed >= 5 else 0.0 for name in QUEUES}
        base_floor = dict(FLOORS)
        if audit_backlog_age_minutes > 60 or pending_campaigns >= 2:
            base_floor["audit"] = 0.40
        residual = max(0.0, 1.0 - sum(base_floor.values()))
        total_rate = sum(rates.values())
        result = dict(base_floor)
        if total_rate:
            for name in QUEUES:
                result[name] += residual * rates[name] / total_rate
        else:
            for name in QUEUES:
                result[name] += residual * INITIAL_TARGETS[name] / sum(INITIAL_TARGETS.values())
    for name in QUEUES:
        result[name] = max(FLOORS[name], min(CEILINGS[name], result[name]))
    if previous is not None:
        lower = {name: max(FLOORS[name], previous.get(name, result[name]) - 0.05) for name in QUEUES}
        upper = {name: min(CEILINGS[name], previous.get(name, result[name]) + 0.05) for name in QUEUES}
        result = {name: max(lower[name], min(upper[name], result[name])) for name in QUEUES}
        # Redistribute the normalization residual inside the same per-queue bounds.
        for _ in range(8):
            residual = 1.0 - sum(result.values())
            if abs(residual) < 1e-12:
                break
            room = ({name: upper[name] - result[name] for name in QUEUES} if residual > 0
                    else {name: result[name] - lower[name] for name in QUEUES})
            total_room = sum(max(0.0, value) for value in room.values())
            if total_room <= 0:
                break
            for name in QUEUES:
                result[name] += residual * max(0.0, room[name]) / total_room
    total = sum(result.values())
    if total <= 0:
        return dict(INITIAL_TARGETS)
    return {name: result[name] / total for name in QUEUES}
