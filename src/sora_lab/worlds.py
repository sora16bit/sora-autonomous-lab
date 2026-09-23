"""Validation of pseudo-worlds used for leakage and attack calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WorldPlan:
    world_id: str
    membership: np.ndarray
    rare_labels: np.ndarray
    version: str = "v1"
    aia_target_n: int = 163
    reference_boundary: str = "fit/reference disjoint"

    def validate(self, *, expected_cohort_sizes: Sequence[int] | None = None,
                 expected_pairwise: np.ndarray | None = None) -> None:
        membership = np.asarray(self.membership, dtype=bool)
        rare = np.asarray(self.rare_labels)
        if membership.ndim != 2 or membership.shape[0] < 2:
            raise ValueError("membership must be a 2-D matrix with at least two cohorts")
        if rare.ndim != 1 or rare.size != membership.shape[1]:
            raise ValueError("rare labels must have one value per person")
        if expected_cohort_sizes is not None and tuple(membership.sum(axis=1)) != tuple(expected_cohort_sizes):
            raise ValueError("cohort sizes do not match the calibration world contract")
        if expected_pairwise is not None:
            actual = membership.astype(int) @ membership.astype(int).T
            expected = np.asarray(expected_pairwise)
            if expected.shape != actual.shape or not np.array_equal(actual, expected):
                raise ValueError("shared-member pairwise matrix does not match the contract")
        if self.aia_target_n < 1:
            raise ValueError("aia_target_n must be positive")

    @property
    def cohort_sizes(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.asarray(self.membership, dtype=bool).sum(axis=1))

    @property
    def pairwise_shared(self) -> np.ndarray:
        matrix = np.asarray(self.membership, dtype=bool).astype(int)
        return matrix @ matrix.T

    def shared_rare(self, cohort_a: int, cohort_b: int) -> int:
        membership = np.asarray(self.membership, dtype=bool)
        rare = np.asarray(self.rare_labels, dtype=bool)
        return int(np.count_nonzero(membership[cohort_a] & membership[cohort_b] & rare))

    def triple_shared(self, cohort_a: int, cohort_b: int, cohort_c: int) -> int:
        membership = np.asarray(self.membership, dtype=bool)
        return int(np.count_nonzero(membership[cohort_a] & membership[cohort_b] & membership[cohort_c]))
