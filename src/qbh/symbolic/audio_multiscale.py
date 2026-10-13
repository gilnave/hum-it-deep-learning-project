# This file compares chroma at a few different time scales.
"""Boundary-robust multiscale chroma retrieval for development selection.

The validated baseline compresses every complete hum and reference excerpt to
the same 32-frame grid.  Real recordings often add or omit a phrase at either
edge.  This versioned experimental backend keeps the baseline representation
but evaluates a small, fixed crop bank before aggregation.  It does not use
DTW, learned weights, or any protected-test information.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qbh.symbolic.audio_local import (
    AudioChromaConfig,
    AudioReferenceTrack,
    extract_cached_chroma,
    prepare_chroma,
    score_chroma_tracks,
)
from qbh.symbolic.evaluation import EvaluationError, RankedCandidate, canonical_sha256


@dataclass(frozen=True)
class MultiScaleChromaConfig:
    """Fixed crop-bank settings selected only on external development data."""

    audio: AudioChromaConfig = AudioChromaConfig()  # noqa: RUF009
    crop_fractions: tuple[float, ...] = (1.0, 0.75)
    aggregation_top_k: int = 2

    def __post_init__(self) -> None:
        if not self.crop_fractions or 1.0 not in self.crop_fractions:
            raise ValueError("multiscale crop fractions must include the full sequence")
        if tuple(sorted(set(self.crop_fractions), reverse=True)) != self.crop_fractions:
            raise ValueError("multiscale crop fractions must be unique and descending")
        if any(not 0.4 <= fraction <= 1.0 for fraction in self.crop_fractions):
            raise ValueError("multiscale crop fractions must lie in [0.4, 1.0]")
        if self.aggregation_top_k < 1:
            raise ValueError("multiscale aggregation_top_k must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> MultiScaleChromaConfig:
        values = values or {}
        return cls(
            audio=AudioChromaConfig.from_mapping(values),
            crop_fractions=tuple(
                float(value) for value in values.get("crop_fractions", (1.0, 0.75))
            ),
            aggregation_top_k=int(values.get("aggregation_top_k", 2)),
        )


def _crop_ranges(frame_count: int, fractions: Sequence[float]) -> list[tuple[int, int]]:
    """Return deterministic full/left/center/right windows with no duplicates."""

    if frame_count < 2:
        raise ValueError("multiscale chroma needs at least two source frames")
    ranges: list[tuple[int, int]] = []
    for fraction in fractions:
        width = min(frame_count, max(2, round(frame_count * float(fraction))))
        maximum_start = frame_count - width
        starts = (0,) if maximum_start == 0 else (0, round(maximum_start / 2), maximum_start)
        for start in starts:
            item = (int(start), int(start + width))
            if item not in ranges:
                ranges.append(item)
    return ranges


def prepare_multiscale_chroma(
    chroma: NDArray[np.floating[Any]], config: MultiScaleChromaConfig
) -> tuple[NDArray[np.float32], tuple[tuple[int, int], ...]]:
    """Prepare each fixed crop on the baseline's common time grid."""

    values = np.asarray(chroma, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 12 or values.shape[1] < 2:
        raise ValueError("multiscale chroma must have shape [12, time>=2]")
    ranges = tuple(_crop_ranges(values.shape[1], config.crop_fractions))
    prepared = np.stack(
        [
            prepare_chroma(values[:, start:end], config.audio.comparison_frames)
            for start, end in ranges
        ]
    )
    return prepared.astype(np.float32, copy=False), ranges


def score_multiscale_tracks(
    query_crops: NDArray[np.float32],
    reference_crops: NDArray[np.float32],
    *,
    top_k: int,
    transposition_invariant: bool,
) -> tuple[NDArray[np.float32], NDArray[np.int16], NDArray[np.int16], NDArray[np.int16]]:
    """Aggregate the strongest fixed crop-pair scores for every reference."""

    if query_crops.ndim != 3 or reference_crops.ndim != 4:
        raise ValueError("multiscale prepared arrays have invalid rank")
    if query_crops.shape[1:] != reference_crops.shape[2:]:
        raise ValueError("multiscale query/reference feature shapes disagree")
    pair_scores: list[NDArray[np.float32]] = []
    pair_shifts: list[NDArray[np.int16]] = []
    pair_indices: list[tuple[int, int]] = []
    for query_index, query in enumerate(query_crops):
        for reference_index in range(reference_crops.shape[1]):
            scores, shifts = score_chroma_tracks(
                query,
                reference_crops[:, reference_index],
                transposition_invariant=transposition_invariant,
            )
            pair_scores.append(scores)
            pair_shifts.append(shifts)
            pair_indices.append((query_index, reference_index))
    score_matrix = np.stack(pair_scores)
    shift_matrix = np.stack(pair_shifts)
    resolved_k = min(int(top_k), score_matrix.shape[0])
    strongest = np.argpartition(score_matrix, -resolved_k, axis=0)[-resolved_k:]
    aggregated = np.take_along_axis(score_matrix, strongest, axis=0).mean(axis=0)
    best_flat = np.argmax(score_matrix, axis=0)
    track_indices = np.arange(score_matrix.shape[1])
    best_shifts = shift_matrix[best_flat, track_indices]
    best_query = np.asarray([pair_indices[index][0] for index in best_flat], dtype=np.int16)
    best_reference = np.asarray([pair_indices[index][1] for index in best_flat], dtype=np.int16)
    return (
        aggregated.astype(np.float32, copy=False),
        best_shifts.astype(np.int16, copy=False),
        best_query,
        best_reference,
    )


class MultiScaleAudioChromaBackend:
    """Rank songs using a fixed, boundary-robust crop bank."""

    name = "local_audio_multiscale_chroma"

    def __init__(
        self,
        tracks: Sequence[AudioReferenceTrack],
        cache_directory: str | Path,
        config: MultiScaleChromaConfig,
        *,
        selection_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not tracks:
            raise EvaluationError("multiscale backend needs reference tracks")
        self.tracks = tuple(tracks)
        self.cache_directory = Path(cache_directory).expanduser().resolve()
        self.config = config
        self.selection_metadata = dict(selection_metadata or {})
        self.candidate_song_ids = tuple(sorted({track.song_id for track in self.tracks}))
        prepared: list[NDArray[np.float32]] = []
        self.reference_ranges: list[tuple[tuple[int, int], ...]] = []
        self.reference_cache_status: dict[str, int] = {"hit": 0, "built": 0}
        for track in self.tracks:
            chroma, digest, status = extract_cached_chroma(
                track.audio_path, self.cache_directory, self.config.audio
            )
            if digest != track.source_sha256:
                raise EvaluationError(f"reference audio changed: {track.track_id}")
            self.reference_cache_status[status] += 1
            crops, ranges = prepare_multiscale_chroma(chroma, self.config)
            prepared.append(crops)
            self.reference_ranges.append(ranges)
        crop_counts = {values.shape[0] for values in prepared}
        if len(crop_counts) != 1:
            raise EvaluationError("multiscale references produced inconsistent crop counts")
        self.reference_chroma = np.stack(prepared)

    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "config": asdict(self.config),
            "selection": self.selection_metadata,
            "cache_directory": str(self.cache_directory),
            "reference_cache_status": self.reference_cache_status,
            "tracks_sha256": canonical_sha256(
                [
                    {
                        "song_id": track.song_id,
                        "track_id": track.track_id,
                        "source_sha256": track.source_sha256,
                    }
                    for track in self.tracks
                ]
            ),
        }

    def rank(self, query: Any) -> Sequence[RankedCandidate]:
        if not isinstance(query, Mapping) or not query.get("audio_path"):
            raise EvaluationError("multiscale query must provide audio_path")
        chroma, _, cache_status = extract_cached_chroma(
            str(query["audio_path"]), self.cache_directory, self.config.audio
        )
        query_crops, query_ranges = prepare_multiscale_chroma(chroma, self.config)
        scores, shifts, query_crop_indices, reference_crop_indices = score_multiscale_tracks(
            query_crops,
            self.reference_chroma,
            top_k=self.config.aggregation_top_k,
            transposition_invariant=self.config.audio.transposition_invariant,
        )
        best_by_song: dict[str, RankedCandidate] = {}
        for index, (track, score, shift) in enumerate(
            zip(self.tracks, scores, shifts, strict=True)
        ):
            query_crop_index = int(query_crop_indices[index])
            reference_crop_index = int(reference_crop_indices[index])
            candidate = RankedCandidate(
                song_id=track.song_id,
                track_id=track.track_id,
                title=track.title,
                score=float(score),
                confidence=float(np.clip(score, 0.0, 1.0)),
                details={
                    "backend": self.name,
                    "reference_id": track.track_id,
                    "query_cache_status": cache_status,
                    "query_transposition_semitones": int(shift),
                    "multiscale_chroma_similarity": float(score),
                    "aggregation_top_k": self.config.aggregation_top_k,
                    "best_query_crop_frames": query_ranges[query_crop_index],
                    "best_reference_crop_frames": self.reference_ranges[index][
                        reference_crop_index
                    ],
                },
            )
            current = best_by_song.get(track.song_id)
            if current is None or (candidate.score, candidate.track_id) > (
                current.score,
                current.track_id,
            ):
                best_by_song[track.song_id] = candidate
        return sorted(best_by_song.values(), key=lambda item: (-item.score, item.song_id))


__all__ = [
    "MultiScaleAudioChromaBackend",
    "MultiScaleChromaConfig",
    "prepare_multiscale_chroma",
    "score_multiscale_tracks",
]
