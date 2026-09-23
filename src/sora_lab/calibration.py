"""Statistical gates that distinguish repeatable attack gains from lucky runs."""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    reason: str
    n: int
    mean_delta: float
    lower_bound: float
    family_lower_bounds: Mapping[str, float] = field(default_factory=dict)
    failures: int = 0
    alpha: float = 0.05
    required_n: int = 32


def required_trials(std: float, *, delta: float = 0.02, alpha: float = 0.05,
                    power: float = 0.80, comparisons: int = 1,
                    minimum: int = 32, maximum: int = 128) -> int:
    """Conservative paired-test sample size with a bounded campaign budget."""
    if delta <= 0 or not 0 < alpha < 1 or not 0 < power < 1 or comparisons < 1:
        raise ValueError("invalid calibration parameters")
    if not math.isfinite(std) or std <= 0:
        return maximum
    alpha_star = alpha / comparisons
    z_alpha = NormalDist().inv_cdf(1.0 - alpha_star)
    z_power = NormalDist().inv_cdf(power)
    estimate = math.ceil(((z_alpha + z_power) * max(std, 0.02) / delta) ** 2)
    return max(minimum, min(maximum, estimate))


def paired_bootstrap_lower(values: Sequence[float], *, alpha: float = 0.05,
                           draws: int = 20_000, seed: int = 0) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired deltas must be a non-empty vector")
    if not 0 < alpha < 1 or draws < 100:
        raise ValueError("invalid bootstrap parameters")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, alpha))


def assess_promotion(child: Sequence[float], parent: Sequence[float], *,
                     family_deltas: Mapping[str, Sequence[float]] | None = None,
                     failures: int = 0, min_trials: int = 32, max_trials: int = 128,
                     delta: float = 0.01, noninferiority: float = -0.02,
                     alpha: float = 0.05, comparisons: int = 1, seed: int = 0) -> PromotionDecision:
    child_arr, parent_arr = np.asarray(child, dtype=float), np.asarray(parent, dtype=float)
    if child_arr.shape != parent_arr.shape or child_arr.ndim != 1:
        raise ValueError("child and parent outcomes must have the same one-dimensional shape")
    n = int(child_arr.size)
    if min_trials < 1 or max_trials < min_trials:
        raise ValueError("invalid trial bounds")
    if failures < 0:
        raise ValueError("failures must be non-negative")
    alpha_star = alpha / comparisons
    required = required_trials(float(np.std(child_arr - parent_arr, ddof=1)) if n > 1 else float("nan"),
                               delta=max(delta, 0.001), alpha=alpha, comparisons=comparisons,
                               minimum=min_trials, maximum=max_trials)
    if n < min_trials:
        return PromotionDecision("inconclusive_min_trials", "minimum paired trials not reached", n,
                                 float(np.mean(child_arr - parent_arr)) if n else 0.0, float("nan"),
                                 {}, failures, alpha_star, required)
    if n > max_trials:
        return PromotionDecision("inconclusive_budget", "campaign exceeded maximum paired trials", n,
                                 float(np.mean(child_arr - parent_arr)), float("nan"), {}, failures,
                                 alpha_star, required)
    deltas = child_arr - parent_arr
    mean_delta = float(np.mean(deltas))
    lower = paired_bootstrap_lower(deltas, alpha=alpha_star, seed=seed)
    family_lower: dict[str, float] = {}
    for family, values in (family_deltas or {}).items():
        family_arr = np.asarray(values, dtype=float)
        if family_arr.ndim != 1 or family_arr.size != n:
            raise ValueError(f"family deltas for {family!r} must have length {n}")
        stable_offset = int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:6], 16) % 10000
        family_lower[family] = paired_bootstrap_lower(family_arr, alpha=alpha_star, seed=seed + stable_offset)
    if failures:
        status, reason = "rejected", "one or more evaluation failures"
    elif mean_delta < delta:
        status, reason = "rejected", "mean paired gain below promotion threshold"
    elif lower <= 0:
        status, reason = "rejected", "one-sided confidence bound does not exceed zero"
    elif any(value < noninferiority for value in family_lower.values()):
        status, reason = "rejected", "a protected attack family regressed beyond the bound"
    else:
        status, reason = "confirmed", "paired gain passed all promotion gates"
    return PromotionDecision(status, reason, n, mean_delta, lower, family_lower, failures,
                             alpha_star, required)
