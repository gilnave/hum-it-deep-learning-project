# This file gives small and limited points to chorus matches.
"""Bounded additive points for repeated chorus-melody agreement.

The legacy product rule multiplied a standardized chorus score by four.  That
could create very large positive or negative terms from tiny score differences.
This module instead awards a small, explicit number of non-negative points.

Location alone never earns a point.  A candidate must first have one of the
strongest acoustic chorus matches in the catalog.  It can then earn additional
fixed points for full-window coverage and for independently matching the same
hum in multiple documented chorus occurrences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AdditiveChorusConfig:
    """Human-readable point schedule with a strict finite maximum."""

    top_one_points: float = 6.0
    top_three_points: float = 4.0
    top_ten_points: float = 3.0
    top_twenty_points: float = 1.5
    full_coverage_threshold: float = 0.95
    partial_coverage_threshold: float = 0.75
    full_coverage_points: float = 2.5
    partial_coverage_points: float = 1.25
    support_quantile: float = 0.35
    support_margin_standard_deviations: float = 1.5
    repeated_occurrence_points: float = 1.5
    maximum_repeated_occurrences: int = 2
    all_occurrences_points: float = 1.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> AdditiveChorusConfig:
        values = values or {}
        defaults = cls()
        result = cls(
            **{
                name: type(default)(values.get(name, default))
                for name, default in asdict(defaults).items()
            }
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def maximum_points(self) -> float:
        return (
            self.top_one_points
            + self.full_coverage_points
            + self.repeated_occurrence_points * self.maximum_repeated_occurrences
            + self.all_occurrences_points
        )

    def validate(self) -> None:
        values = (
            self.top_one_points,
            self.top_three_points,
            self.top_ten_points,
            self.top_twenty_points,
            self.full_coverage_points,
            self.partial_coverage_points,
            self.repeated_occurrence_points,
            self.all_occurrences_points,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("chorus points cannot be negative")
        if not (
            self.top_one_points
            >= self.top_three_points
            >= self.top_ten_points
            >= self.top_twenty_points
        ):
            raise ValueError("stronger chorus ranks cannot receive fewer points")
        if not 0.0 <= self.partial_coverage_threshold <= self.full_coverage_threshold <= 1.0:
            raise ValueError("chorus coverage thresholds are invalid")
        if not 0.0 <= self.support_quantile <= 1.0:
            raise ValueError("chorus support quantile is invalid")
        if self.support_margin_standard_deviations < 0.0:
            raise ValueError("chorus support margin cannot be negative")
        if self.maximum_repeated_occurrences < 0:
            raise ValueError("maximum repeated occurrences cannot be negative")


@dataclass(frozen=True)
class ChorusPointBreakdown:
    """Explanation for one candidate's exact additive point award."""

    total_points: float
    strength_points: float
    coverage_points: float
    repetition_points: float
    all_occurrences_points: float
    best_rank: int
    matched_occurrences: int
    total_occurrences: int
    best_occurrence: int


def _strength_points(rank: int, config: AdditiveChorusConfig) -> float:
    if rank <= 1:
        return config.top_one_points
    if rank <= 3:
        return config.top_three_points
    if rank <= 10:
        return config.top_ten_points
    if rank <= 20:
        return config.top_twenty_points
    return 0.0


def award_additive_chorus_points(
    occurrence_scores: Sequence[Sequence[float]],
    occurrence_coverages: Sequence[Sequence[float]],
    config: AdditiveChorusConfig | None = None,
) -> tuple[NDArray[np.float64], tuple[ChorusPointBreakdown, ...]]:
    """Award bounded points from acoustic rank, coverage, and repetition.

    ``occurrence_scores[candidate][occurrence]`` is the best acoustic melody
    score among windows belonging to that distinct chorus occurrence.  The
    matching coverage vector describes those same selected windows.
    """

    settings = config or AdditiveChorusConfig()
    settings.validate()
    if not occurrence_scores or len(occurrence_scores) != len(occurrence_coverages):
        raise ValueError("chorus occurrence scores and coverages must align")
    scores = [np.asarray(values, dtype=np.float64) for values in occurrence_scores]
    coverages = [np.asarray(values, dtype=np.float64) for values in occurrence_coverages]
    for candidate_scores, candidate_coverages in zip(scores, coverages, strict=True):
        if (
            candidate_scores.ndim != 1
            or not len(candidate_scores)
            or candidate_scores.shape != candidate_coverages.shape
            or not np.isfinite(candidate_scores).all()
            or not np.isfinite(candidate_coverages).all()
            or np.any((candidate_coverages < 0.0) | (candidate_coverages > 1.0))
        ):
            raise ValueError("each candidate needs finite aligned chorus occurrences")

    best_scores = np.asarray([values.max() for values in scores], dtype=np.float64)
    best_ranks = 1 + (best_scores[None, :] > best_scores[:, None]).sum(axis=1)
    population_spread = max(float(best_scores.std()), 1e-8)
    population_floor = float(np.quantile(best_scores, settings.support_quantile))
    points = np.zeros(len(scores), dtype=np.float64)
    breakdowns: list[ChorusPointBreakdown] = []
    for candidate, (candidate_scores, candidate_coverages) in enumerate(
        zip(scores, coverages, strict=True)
    ):
        best_occurrence = int(np.argmax(candidate_scores))
        rank = int(best_ranks[candidate])
        strength = _strength_points(rank, settings)
        # A high rank caused only by a large tie at the population floor is
        # not acoustic evidence. This is the explicit location-only guard.
        if float(candidate_scores[best_occurrence]) <= population_floor + 1e-8:
            strength = 0.0
        coverage = float(candidate_coverages[best_occurrence])
        if strength <= 0.0:
            coverage_points = 0.0
        elif coverage >= settings.full_coverage_threshold:
            coverage_points = settings.full_coverage_points
        elif coverage >= settings.partial_coverage_threshold:
            coverage_points = settings.partial_coverage_points
        else:
            coverage_points = 0.0

        support_floor = max(
            population_floor,
            float(candidate_scores[best_occurrence])
            - settings.support_margin_standard_deviations * population_spread,
        )
        matched = int(np.sum(candidate_scores >= support_floor)) if strength > 0.0 else 0
        repeated = min(
            max(0, matched - 1), settings.maximum_repeated_occurrences
        )
        repetition_points = repeated * settings.repeated_occurrence_points
        all_points = (
            settings.all_occurrences_points
            if strength > 0.0 and len(candidate_scores) >= 2 and matched == len(candidate_scores)
            else 0.0
        )
        total = strength + coverage_points + repetition_points + all_points
        if not 0.0 <= total <= settings.maximum_points + 1e-8:
            raise RuntimeError("bounded chorus point invariant failed")
        points[candidate] = total
        breakdowns.append(
            ChorusPointBreakdown(
                total_points=float(total),
                strength_points=float(strength),
                coverage_points=float(coverage_points),
                repetition_points=float(repetition_points),
                all_occurrences_points=float(all_points),
                best_rank=rank,
                matched_occurrences=matched,
                total_occurrences=len(candidate_scores),
                best_occurrence=best_occurrence,
            )
        )
    return points, tuple(breakdowns)


__all__ = [
    "AdditiveChorusConfig",
    "ChorusPointBreakdown",
    "award_additive_chorus_points",
]
