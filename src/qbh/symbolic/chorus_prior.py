# This file checks if a strong match also appears in a chorus.
"""Validated chorus metadata and bounded chorus-aware score adjustment.

The retrieval models work with windows measured from the beginning of a
silence-trimmed vocal stem.  Chorus annotations, however, are useful to people
only when they use the original song clock.  This module owns that conversion
and keeps the prior deliberately small and auditable: it can reorder a close
decision, but it never changes either model's acoustic features.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ChorusMetadataError(ValueError):
    """Raised when chorus metadata is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ChorusEvidence:
    """Evidence that a method's best local match belongs to a chorus."""

    evidence: float
    coverage: float
    supporting_windows: int
    source_window_start_seconds: float
    source_window_end_seconds: float
    chorus_start_seconds: float | None
    chorus_end_seconds: float | None


def load_chorus_metadata(
    path: str | Path,
    *,
    expected_song_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load a complete, source-clock chorus database for one fixed catalog."""

    metadata_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChorusMetadataError(f"could not read chorus metadata: {metadata_path}") from error
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("format_version", -1)) != 1
        or payload.get("time_reference") != "original_song_seconds"
        or not isinstance(payload.get("songs"), list)
    ):
        raise ChorusMetadataError("unsupported chorus metadata format")

    by_id: dict[str, dict[str, Any]] = {}
    for raw_song in payload["songs"]:
        if not isinstance(raw_song, Mapping):
            raise ChorusMetadataError("chorus song entries must be objects")
        song = dict(raw_song)
        song_id = str(song.get("song_id", ""))
        if not song_id or song_id in by_id:
            raise ChorusMetadataError("chorus metadata has a missing or duplicate song ID")
        duration = float(song.get("duration_seconds", 0.0))
        trim_start = float(song.get("vocal_trim_start_seconds", -1.0))
        trim_end = float(song.get("vocal_trim_end_seconds", -1.0))
        if duration <= 0.0 or not 0.0 <= trim_start < trim_end <= duration + 0.25:
            raise ChorusMetadataError(f"invalid duration/trim clock for {song_id}")
        previous_end = -1.0
        choruses = song.get("choruses")
        if not isinstance(choruses, list):
            raise ChorusMetadataError(f"choruses must be a list for {song_id}")
        normalized: list[dict[str, float]] = []
        for raw_interval in choruses:
            if not isinstance(raw_interval, Mapping):
                raise ChorusMetadataError(f"invalid chorus interval for {song_id}")
            start = float(raw_interval.get("start_seconds", -1.0))
            end = float(raw_interval.get("end_seconds", -1.0))
            if not 0.0 <= start < end <= duration + 0.25 or start < previous_end:
                raise ChorusMetadataError(f"invalid/overlapping chorus intervals for {song_id}")
            normalized.append({"start_seconds": start, "end_seconds": end})
            previous_end = end
        song["choruses"] = normalized
        by_id[song_id] = song

    expected = set(expected_song_ids)
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        raise ChorusMetadataError(
            f"chorus catalog mismatch (missing={missing[:3]}, extra={extra[:3]})"
        )
    return by_id


def standardize_candidate_scores(values: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Put one query's candidate scores on a stable standard-deviation scale."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("candidate scores must be one finite vector")
    deviation = float(scores.std())
    if deviation < 1e-8:
        return np.zeros_like(scores)
    return (scores - float(scores.mean())) / deviation


def _coverage(
    window_start: float,
    window_end: float,
    chorus_start: float,
    chorus_end: float,
    padding_seconds: float,
) -> float:
    padded_start = max(0.0, chorus_start - padding_seconds)
    padded_end = chorus_end + padding_seconds
    intersection = max(0.0, min(window_end, padded_end) - max(window_start, padded_start))
    return intersection / max(window_end - window_start, 1e-8)


def chorus_evidence_for_candidate(
    *,
    song_metadata: Mapping[str, Any],
    reference_windows: Sequence[Mapping[str, Any]],
    best_local_window: int,
    padding_seconds: float = 2.0,
    minimum_coverage: float = 0.5,
    agreement_increment: float = 0.125,
    maximum_evidence: float = 1.25,
) -> ChorusEvidence:
    """Measure chorus overlap, anchored to a method's actual best window.

    The primary evidence is the fraction of the best matched 12-second window
    inside one chorus (with the agreed two-second boundary tolerance).  Other
    retained windows can add a small agreement bonus only when they independently
    overlap that *same* chorus.  This helps long hums spanning adjacent windows
    without letting an unrelated fourth-best match manufacture a chorus reward.
    """

    if not 0 <= best_local_window < len(reference_windows):
        raise IndexError("best local window is outside the retained window bag")
    if padding_seconds < 0.0 or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("invalid chorus overlap settings")
    trim_start = float(song_metadata["vocal_trim_start_seconds"])
    source_windows = [
        (
            trim_start + float(window["start_seconds"]),
            trim_start + float(window["end_seconds"]),
        )
        for window in reference_windows
    ]
    anchor_start, anchor_end = source_windows[best_local_window]
    best_coverage = 0.0
    best_chorus: Mapping[str, Any] | None = None
    for chorus in song_metadata["choruses"]:
        coverage = _coverage(
            anchor_start,
            anchor_end,
            float(chorus["start_seconds"]),
            float(chorus["end_seconds"]),
            padding_seconds,
        )
        if coverage > best_coverage:
            best_coverage = coverage
            best_chorus = chorus
    if best_chorus is None or best_coverage < minimum_coverage:
        return ChorusEvidence(0.0, best_coverage, 0, anchor_start, anchor_end, None, None)

    chorus_start = float(best_chorus["start_seconds"])
    chorus_end = float(best_chorus["end_seconds"])
    supporting = sum(
        _coverage(start, end, chorus_start, chorus_end, padding_seconds)
        >= minimum_coverage
        for start, end in source_windows
    )
    confidence = float(np.clip(float(song_metadata.get("confidence", 0.0)), 0.0, 1.0))
    evidence = confidence * min(
        maximum_evidence,
        best_coverage + agreement_increment * max(0, supporting - 1),
    )
    return ChorusEvidence(
        evidence=float(evidence),
        coverage=float(best_coverage),
        supporting_windows=int(supporting),
        source_window_start_seconds=anchor_start,
        source_window_end_seconds=anchor_end,
        chorus_start_seconds=chorus_start,
        chorus_end_seconds=chorus_end,
    )


__all__ = [
    "ChorusEvidence",
    "ChorusMetadataError",
    "chorus_evidence_for_candidate",
    "load_chorus_metadata",
    "standardize_candidate_scores",
]
