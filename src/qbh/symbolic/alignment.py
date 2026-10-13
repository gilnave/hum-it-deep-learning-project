# These functions align two melody tracks before comparing them.
"""Explainable, key-invariant alignment for symbolic melody sequences."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np


class SymbolicAlignmentError(ValueError):
    """Raised when a symbolic melody is malformed or too short."""


@runtime_checkable
class NoteEventProtocol(Protocol):
    midi_pitch: int
    onset_seconds: float
    duration_seconds: float
    confidence: float


@dataclass(frozen=True)
class _Event:
    midi_pitch: int
    onset_seconds: float
    duration_seconds: float
    confidence: float


@dataclass(frozen=True)
class MelodyTrack:
    """Adapter around either the catalog dataclass or its JSON form."""

    song_id: str
    track_id: str
    title: str
    events: tuple[Any, ...]


@dataclass(frozen=True)
class AlignmentSettings:
    minimum_notes: int = 4
    ngram_sizes: tuple[int, ...] = (1, 2, 3)
    coarse_candidates: int = 60
    ordinary_gap_penalty: float = 0.65
    repeated_note_gap_penalty: float = 0.18
    merge_penalty: float = 0.16
    pitch_weight: float = 0.60
    rhythm_weight: float = 0.15
    ngram_weight: float = 0.15
    gap_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.minimum_notes < 3 or self.coarse_candidates < 1:
            raise ValueError("minimum_notes/coarse_candidates are invalid")
        if not self.ngram_sizes or min(self.ngram_sizes) < 1:
            raise ValueError("ngram sizes must be positive")
        penalties = (
            self.ordinary_gap_penalty,
            self.repeated_note_gap_penalty,
            self.merge_penalty,
        )
        weights = (self.pitch_weight, self.rhythm_weight, self.ngram_weight, self.gap_weight)
        if min(penalties) < 0 or min(weights) < 0 or not np.isclose(sum(weights), 1.0):
            raise ValueError("penalties must be non-negative and score weights must sum to one")


@dataclass(frozen=True)
class AlignmentOperation:
    operation: str
    query_start: int
    query_end: int
    reference_start: int
    reference_end: int
    pitch_cost: float
    rhythm_cost: float

    def to_dict(self) -> dict[str, int | float | str]:
        return vars(self).copy()


@dataclass(frozen=True)
class AlignmentTrace:
    total_distance: float
    pitch_distance: float
    rhythm_distance: float
    ngram_similarity: float
    gap_fraction: float
    query_start_note: int
    query_end_note: int
    reference_start_note: int
    reference_end_note: int
    operations: tuple[AlignmentOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        values = vars(self).copy()
        values["operations"] = [operation.to_dict() for operation in self.operations]
        return values


@dataclass(frozen=True)
class SymbolicMatch:
    rank: int
    song_id: str
    track_id: str
    title: str
    query_hypothesis: int
    trace: AlignmentTrace

    def to_dict(self) -> dict[str, Any]:
        values = vars(self).copy()
        values["trace"] = self.trace.to_dict()
        return values


def _field(value: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _coerce_events(value: Any, minimum_notes: int) -> tuple[_Event, ...]:
    raw_events = _field(value, ("events", "notes"), value)
    if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
        raise SymbolicAlignmentError("melody must expose an events/notes sequence")
    events: list[_Event] = []
    for raw in raw_events:
        pitch = _field(raw, ("midi_pitch", "pitch_midi", "pitch"))
        onset = _field(raw, ("onset_seconds", "onset", "start_seconds"))
        duration = _field(raw, ("duration_seconds", "duration"))
        confidence = _field(raw, ("confidence",), 1.0)
        if pitch is None or onset is None or duration is None:
            raise SymbolicAlignmentError("each note needs MIDI pitch, onset, and duration")
        event = _Event(round(float(pitch)), float(onset), float(duration), float(confidence))
        if (
            not 0 <= event.midi_pitch <= 127
            or event.onset_seconds < 0
            or event.duration_seconds <= 0
            or not 0 <= event.confidence <= 1
        ):
            raise SymbolicAlignmentError("note values are outside valid ranges")
        events.append(event)
    events.sort(key=lambda item: item.onset_seconds)
    if len(events) < minimum_notes:
        raise SymbolicAlignmentError(f"melody needs at least {minimum_notes} notes")
    return tuple(events)


def coerce_track(value: Any, minimum_notes: int = 4) -> MelodyTrack:
    """Keep schema parsing behind one adapter while the schema evolves."""

    song_id = str(_field(value, ("song_id", "item_id", "id"), "unknown"))
    track_id = str(_field(value, ("track_id", "melody_id", "source"), "main"))
    title = str(_field(value, ("title", "song_title", "display_name"), song_id))
    events = _coerce_events(value, minimum_notes)
    return MelodyTrack(song_id, track_id, title, events)


def _wrap_interval(value: float) -> int:
    wrapped = round((value + 6.0) % 12.0 - 6.0)
    return 6 if wrapped == -6 and value > 0 else wrapped


def _intervals(events: tuple[_Event, ...]) -> np.ndarray[Any, np.dtype[np.int16]]:
    return np.asarray(
        [
            _wrap_interval(second.midi_pitch - first.midi_pitch)
            for first, second in itertools.pairwise(events)
        ],
        dtype=np.int16,
    )


def _rhythm(events: tuple[_Event, ...]) -> np.ndarray[Any, np.dtype[np.float64]]:
    values = np.asarray(
        [
            max(1e-3, second.onset_seconds - first.onset_seconds)
            for first, second in itertools.pairwise(events)
        ],
        dtype=np.float64,
    )
    return values / max(float(np.median(values)), 1e-3)


def _counter(values: np.ndarray[Any, Any], size: int) -> Counter[tuple[int, ...]]:
    tokens = values.astype(np.int16).tolist()
    return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def coarse_interval_similarity(
    query: Sequence[int], reference: Sequence[int], sizes: Sequence[int] = (1, 2, 3)
) -> float:
    """Query-normalized n-gram containment that does not penalize long songs."""

    query_values = np.asarray(query, dtype=np.int16)
    reference_values = np.asarray(reference, dtype=np.int16)
    scores: list[float] = []
    for size in sizes:
        query_counter = _counter(query_values, size)
        if query_counter:
            overlap = sum((query_counter & _counter(reference_values, size)).values())
            scores.append(overlap / sum(query_counter.values()))
    return float(np.mean(scores)) if scores else 0.0


def _pitch_cost(query_value: int, reference_value: int) -> float:
    return min(abs(_wrap_interval(query_value - reference_value)) / 6.0, 1.0)


def _gap_cost(value: int, settings: AlignmentSettings) -> float:
    return settings.repeated_note_gap_penalty if value == 0 else settings.ordinary_gap_penalty


def _alignment_table(
    query: np.ndarray[Any, Any], reference: np.ndarray[Any, Any], settings: AlignmentSettings
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Semi-global edit table with split/merge transitions."""

    costs = np.full((len(query) + 1, len(reference) + 1), np.inf, dtype=np.float64)
    previous = np.zeros(costs.shape, dtype=np.uint8)
    costs[0, :] = 0.0  # free full-song prefix
    for i in range(1, len(query) + 1):
        costs[i, 0] = costs[i - 1, 0] + _gap_cost(int(query[i - 1]), settings)
        previous[i, 0] = 2
    for i in range(1, len(query) + 1):
        for j in range(1, len(reference) + 1):
            candidates = [
                (costs[i - 1, j - 1] + _pitch_cost(int(query[i - 1]), int(reference[j - 1])), 1),
                (costs[i - 1, j] + _gap_cost(int(query[i - 1]), settings), 2),
                (costs[i, j - 1] + _gap_cost(int(reference[j - 1]), settings), 3),
            ]
            if j >= 2:
                combined = _wrap_interval(int(reference[j - 2]) + int(reference[j - 1]))
                candidates.append(
                    (
                        costs[i - 1, j - 2]
                        + _pitch_cost(int(query[i - 1]), combined)
                        + settings.merge_penalty,
                        4,
                    )
                )
            if i >= 2:
                combined = _wrap_interval(int(query[i - 2]) + int(query[i - 1]))
                candidates.append(
                    (
                        costs[i - 2, j - 1]
                        + _pitch_cost(combined, int(reference[j - 1]))
                        + settings.merge_penalty,
                        5,
                    )
                )
            costs[i, j], previous[i, j] = min(candidates, key=lambda item: item[0])
    return costs, previous


def _rhythm_cost(
    query: np.ndarray[Any, Any],
    reference: np.ndarray[Any, Any],
    query_slice: slice,
    reference_slice: slice,
) -> float:
    query_total = float(query[query_slice].sum())
    reference_total = float(reference[reference_slice].sum())
    if query_total <= 0 or reference_total <= 0:
        return 1.0
    return float(min(abs(np.log(query_total / reference_total)) / 2.0, 1.0))


def align_symbolic_melodies(
    query: Any, reference: Any, settings: AlignmentSettings | None = None
) -> AlignmentTrace:
    """Align the complete query against the best local reference phrase."""

    resolved = settings or AlignmentSettings()
    query_events = _coerce_events(query, resolved.minimum_notes)
    reference_events = _coerce_events(reference, resolved.minimum_notes)
    query_intervals = _intervals(query_events)
    reference_intervals = _intervals(reference_events)
    query_rhythm, reference_rhythm = _rhythm(query_events), _rhythm(reference_events)
    costs, previous = _alignment_table(query_intervals, reference_intervals, resolved)
    reference_end = int(np.argmin(costs[len(query_intervals), 1:])) + 1
    i, j = len(query_intervals), reference_end
    reverse_operations: list[AlignmentOperation] = []
    pitch_costs: list[float] = []
    rhythm_costs: list[float] = []
    gaps = 0
    matched_query = 0

    while i > 0:
        code = int(previous[i, j])
        steps = {
            1: (1, 1, "match"),
            2: (1, 0, "query_insertion"),
            3: (0, 1, "reference_insertion"),
            4: (1, 2, "reference_note_merge"),
            5: (2, 1, "query_note_merge"),
        }
        if code not in steps:
            raise RuntimeError("invalid alignment backtracking state")
        query_step, reference_step, operation_name = steps[code]
        query_start, reference_start = i - query_step, j - reference_step
        if query_step and reference_step:
            query_value = _wrap_interval(int(query_intervals[query_start:i].sum()))
            reference_value = _wrap_interval(int(reference_intervals[reference_start:j].sum()))
            pitch_cost = _pitch_cost(query_value, reference_value)
            rhythm_cost = _rhythm_cost(
                query_rhythm, reference_rhythm, slice(query_start, i), slice(reference_start, j)
            )
            pitch_costs.append(pitch_cost)
            rhythm_costs.append(rhythm_cost)
            matched_query += query_step
        else:
            pitch_cost = rhythm_cost = 1.0
            gaps += max(query_step, reference_step)
        reverse_operations.append(
            AlignmentOperation(
                operation_name, query_start, i, reference_start, j, pitch_cost, rhythm_cost
            )
        )
        i, j = query_start, reference_start

    operations = tuple(reversed(reverse_operations))
    reference_start = j
    pitch_distance = float(np.mean(pitch_costs)) if pitch_costs else 1.0
    rhythm_distance = float(np.mean(rhythm_costs)) if rhythm_costs else 1.0
    reference_span = max(1, reference_end - reference_start)
    gap_fraction = min(1.0, gaps / max(len(query_intervals), reference_span))
    ngram_similarity = coarse_interval_similarity(
        query_intervals, reference_intervals, resolved.ngram_sizes
    )
    total_distance = (
        resolved.pitch_weight * pitch_distance
        + resolved.rhythm_weight * rhythm_distance
        + resolved.ngram_weight * (1.0 - ngram_similarity)
        + resolved.gap_weight * gap_fraction
    )
    if matched_query == 0:
        total_distance = 1.0
    return AlignmentTrace(
        float(total_distance),
        pitch_distance,
        rhythm_distance,
        ngram_similarity,
        float(gap_fraction),
        0,
        len(query_events),
        reference_start,
        min(len(reference_events), reference_end + 1),
        operations,
    )


def rank_symbolic_tracks(
    query_hypotheses: Sequence[Any],
    tracks: Sequence[Any],
    *,
    settings: AlignmentSettings | None = None,
    limit: int = 10,
) -> list[SymbolicMatch]:
    """Coarse n-gram filtering followed by fine alignment and song grouping."""

    resolved = settings or AlignmentSettings()
    if not query_hypotheses or not tracks:
        raise SymbolicAlignmentError("query hypotheses and tracks cannot be empty")
    if limit < 1:
        raise ValueError("limit must be positive")
    queries = [_coerce_events(item, resolved.minimum_notes) for item in query_hypotheses]
    catalog = [coerce_track(item, resolved.minimum_notes) for item in tracks]
    pairs: list[tuple[float, int, int]] = []
    for query_index, query_events in enumerate(queries):
        for track_index, track in enumerate(catalog):
            similarity = coarse_interval_similarity(
                _intervals(query_events),
                _intervals(_coerce_events(track.events, resolved.minimum_notes)),
                resolved.ngram_sizes,
            )
            pairs.append((similarity, query_index, track_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_by_song: dict[str, SymbolicMatch] = {}
    for _, query_index, track_index in pairs[: resolved.coarse_candidates]:
        track = catalog[track_index]
        trace = align_symbolic_melodies(queries[query_index], track.events, resolved)
        candidate = SymbolicMatch(0, track.song_id, track.track_id, track.title, query_index, trace)
        current = best_by_song.get(track.song_id)
        if current is None or trace.total_distance < current.trace.total_distance:
            best_by_song[track.song_id] = candidate
    ordered = sorted(
        best_by_song.values(),
        key=lambda item: (item.trace.total_distance, item.song_id, item.track_id),
    )
    return [replace(item, rank=rank) for rank, item in enumerate(ordered[:limit], 1)]


__all__ = [
    "AlignmentOperation",
    "AlignmentSettings",
    "AlignmentTrace",
    "MelodyTrack",
    "NoteEventProtocol",
    "SymbolicAlignmentError",
    "SymbolicMatch",
    "align_symbolic_melodies",
    "coarse_interval_similarity",
    "coerce_track",
    "rank_symbolic_tracks",
]
