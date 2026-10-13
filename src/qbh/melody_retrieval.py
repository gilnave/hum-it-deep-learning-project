# These tools turn audio into a simple melody line.
"""General query-by-humming retrieval from vocal melody contours.

The original project compares a hum''s full-mix log-Mel embedding with song
audio.  That representation is dominated by production and timbre.  This module
instead compares the musical object shared by both inputs: a sequence of pitch
changes.  Catalog contours come from a pretrained Omnizart vocal-contour model;
queries use monophonic pYIN, which is well suited to a single humming voice.

Matching is invariant to singer key and robust to moderate tempo differences:

* F0 is quantized and compressed into note runs.
* Adjacent pitch changes are wrapped to the nearest octave-equivalent interval.
* Constrained subsequence DTW locates the query anywhere inside a full song.
* Local interval n-grams and cumulative melody shape break DTW ties.

No personal recording is required to build this index.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from qbh.audio import load_audio, trim_silence

FloatArray = NDArray[np.float32]


class MelodyContourError(ValueError):
    """Raised when audio or a cached contour contains too little melody."""


@dataclass(frozen=True)
class MelodySettings:
    """Fixed signal and matching choices selected independently of user tests."""

    sample_rate: int = 16_000
    frame_length: int = 2_048
    hop_length: int = 320  # 20 ms, matching the Omnizart catalog contours.
    fmin_hz: float = float(librosa.note_to_hz("C2"))
    fmax_hz: float = float(librosa.note_to_hz("C6"))
    trim_top_db: float = 35.0
    # pYIN's voiced_flag already applies its probabilistic voicing model.  A
    # second 0.5 cutoff discarded most valid notes in public CHAD recordings.
    minimum_voiced_probability: float = 0.0
    median_filter_frames: int = 5
    maximum_gap_frames: int = 3
    minimum_note_frames: int = 3
    minimum_notes: int = 6
    interval_ngram_sizes: tuple[int, ...] = (2, 3)
    ngram_mismatch_weight: float = 0.75
    shape_distance_weight: float = 0.15
    rhythm_distance_weight: float = 0.0
    direction_distance_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_length <= 0 or self.hop_length <= 0:
            raise ValueError("sample rate, frame length, and hop length must be positive")
        if not 0 < self.fmin_hz < self.fmax_hz <= self.sample_rate / 2:
            raise ValueError("pitch limits must be positive, ordered, and below Nyquist")
        if self.median_filter_frames < 1 or self.median_filter_frames % 2 == 0:
            raise ValueError("median_filter_frames must be a positive odd integer")
        if self.maximum_gap_frames < 0 or self.minimum_note_frames < 1:
            raise ValueError("gap and note frame limits are invalid")
        if not 0.0 <= self.minimum_voiced_probability <= 1.0:
            raise ValueError("minimum_voiced_probability must be between zero and one")
        if self.minimum_notes < 3:
            raise ValueError("minimum_notes must be at least three")
        if not self.interval_ngram_sizes or min(self.interval_ngram_sizes) < 1:
            raise ValueError("interval n-gram sizes must be positive")
        if (
            min(
                self.ngram_mismatch_weight,
                self.shape_distance_weight,
                self.rhythm_distance_weight,
                self.direction_distance_weight,
            )
            < 0
        ):
            raise ValueError("melody score weights cannot be negative")


@dataclass(frozen=True)
class MelodySequence:
    """A tempo-light melody representation made from stable note runs."""

    pitches: FloatArray
    durations_seconds: FloatArray

    def __post_init__(self) -> None:
        pitches = np.asarray(self.pitches, dtype=np.float32).reshape(-1)
        durations = np.asarray(self.durations_seconds, dtype=np.float32).reshape(-1)
        if pitches.size != durations.size or pitches.size < 2:
            raise MelodyContourError("melody needs aligned pitch/duration arrays and two notes")
        if not np.isfinite(pitches).all() or not np.isfinite(durations).all():
            raise MelodyContourError("melody sequence must contain only finite values")
        if np.any(durations <= 0):
            raise MelodyContourError("note durations must be positive")
        object.__setattr__(self, "pitches", pitches)
        object.__setattr__(self, "durations_seconds", durations)

    @property
    def intervals(self) -> FloatArray:
        """Return signed octave-equivalent steps in the range [-6, 6]."""

        return wrap_intervals(np.diff(self.pitches))

    @property
    def relative_shape(self) -> FloatArray:
        """Integrate robust intervals into a transposition-free melody shape."""

        shape = np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(self.intervals)))
        shape -= np.median(shape)
        return shape.astype(np.float32, copy=False)


@dataclass(frozen=True)
class ReferenceMelody:
    song_id: str
    title: str
    contour_path: Path
    melody: MelodySequence


@dataclass(frozen=True)
class MelodyMatch:
    rank: int
    song_id: str
    title: str
    distance: float
    interval_dtw_distance: float
    ngram_similarity: float
    shape_dtw_distance: float
    rhythm_dtw_distance: float
    direction_dtw_distance: float
    reference_start_note: int
    reference_end_note: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "song_id": self.song_id,
            "title": self.title,
            "distance": self.distance,
            "interval_dtw_distance": self.interval_dtw_distance,
            "ngram_similarity": self.ngram_similarity,
            "shape_dtw_distance": self.shape_dtw_distance,
            "rhythm_dtw_distance": self.rhythm_dtw_distance,
            "direction_dtw_distance": self.direction_dtw_distance,
            "reference_start_note": self.reference_start_note,
            "reference_end_note": self.reference_end_note,
        }


@dataclass(frozen=True)
class _Alignment:
    distance: float
    reference_start: int
    reference_end: int


def wrap_intervals(intervals: NDArray[np.floating[Any]]) -> FloatArray:
    """Map leaps to their nearest signed octave-equivalent displacement.

    Vocal trackers occasionally jump by an octave, while people also change
    octave mid-hum.  Query-by-humming cares about melodic direction and pitch
    class, so +12 becomes 0 and +7 becomes -5.  Exact tritones retain their
    original direction to avoid an arbitrary sign flip at six semitones.
    """

    values = np.asarray(intervals, dtype=np.float32)
    wrapped = np.mod(values + 6.0, 12.0) - 6.0
    positive_tritone = np.isclose(wrapped, -6.0) & (values > 0)
    wrapped[positive_tritone] = 6.0
    return wrapped.astype(np.float32, copy=False)


def _bridge_short_gaps(midi: FloatArray, maximum_gap_frames: int) -> FloatArray:
    """Linearly bridge only tiny unvoiced gaps inside otherwise voiced audio."""

    output = midi.copy()
    finite = np.isfinite(output)
    index = 0
    while index < len(output):
        if finite[index]:
            index += 1
            continue
        gap_start = index
        while index < len(output) and not finite[index]:
            index += 1
        gap_end = index
        gap_size = gap_end - gap_start
        has_voiced_edges = gap_start > 0 and gap_end < len(output)
        if has_voiced_edges and gap_size <= maximum_gap_frames:
            output[gap_start:gap_end] = np.linspace(
                output[gap_start - 1],
                output[gap_end],
                gap_size + 2,
                dtype=np.float32,
            )[1:-1]
            finite[gap_start:gap_end] = True
    return output


def _smooth_voiced_regions(midi: FloatArray, filter_frames: int) -> FloatArray:
    """Median-filter voiced regions independently so silence cannot leak in."""

    output = midi.copy()
    finite = np.isfinite(output)
    boundaries = np.diff(np.concatenate(([False], finite, [False])).astype(np.int8))
    starts = np.flatnonzero(boundaries == 1)
    ends = np.flatnonzero(boundaries == -1)
    for start, end in zip(starts, ends, strict=True):
        region = output[start:end]
        if len(region) >= filter_frames:
            output[start:end] = median_filter(region, size=filter_frames, mode="nearest")
    return output


def melody_from_f0(
    f0_hz: NDArray[np.floating[Any]],
    hop_seconds: float,
    settings: MelodySettings | None = None,
) -> MelodySequence:
    """Clean a frame-level F0 track and compress it into quantized note runs."""

    resolved = settings or MelodySettings()
    f0 = np.asarray(f0_hz, dtype=np.float32).reshape(-1)
    if f0.size == 0 or not np.isfinite(f0).all() or hop_seconds <= 0:
        raise MelodyContourError("F0 contour and hop must be finite and non-empty")

    midi = np.full(f0.shape, np.nan, dtype=np.float32)
    voiced = f0 > 0
    midi[voiced] = librosa.hz_to_midi(f0[voiced]).astype(np.float32)
    midi = _bridge_short_gaps(midi, resolved.maximum_gap_frames)
    midi = _smooth_voiced_regions(midi, resolved.median_filter_frames)
    quantized = np.round(midi).astype(np.float32)

    pitches: list[float] = []
    durations: list[float] = []
    index = 0
    while index < len(quantized):
        if not np.isfinite(quantized[index]):
            index += 1
            continue
        run_start = index
        pitch = float(quantized[index])
        index += 1
        while index < len(quantized) and quantized[index] == pitch:
            index += 1
        run_frames = index - run_start
        if run_frames >= resolved.minimum_note_frames:
            pitches.append(pitch)
            durations.append(run_frames * hop_seconds)

    if len(pitches) < resolved.minimum_notes:
        raise MelodyContourError(
            f"contour produced {len(pitches)} stable notes; "
            f"at least {resolved.minimum_notes} are required"
        )
    return MelodySequence(
        pitches=np.asarray(pitches, dtype=np.float32),
        durations_seconds=np.asarray(durations, dtype=np.float32),
    )


def extract_hum_melody(
    audio_path: str | Path,
    settings: MelodySettings | None = None,
) -> MelodySequence:
    """Extract a monophonic melody from a microphone recording with pYIN."""

    resolved = settings or MelodySettings()
    waveform = load_audio(audio_path, sample_rate=resolved.sample_rate, mono=True)
    waveform = trim_silence(
        waveform,
        top_db=resolved.trim_top_db,
        frame_length=resolved.frame_length,
        hop_length=resolved.hop_length,
    )
    if waveform.shape[-1] < resolved.frame_length:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, resolved.frame_length - waveform.shape[-1]),
        )
    f0_hz, voiced_flag, voiced_probability = librosa.pyin(
        waveform.squeeze(0).cpu().numpy(),
        fmin=resolved.fmin_hz,
        fmax=resolved.fmax_hz,
        sr=resolved.sample_rate,
        frame_length=resolved.frame_length,
        hop_length=resolved.hop_length,
        center=True,
        fill_na=np.nan,
    )
    reliable = (
        np.asarray(voiced_flag, dtype=bool)
        & np.isfinite(f0_hz)
        & (np.asarray(voiced_probability) >= resolved.minimum_voiced_probability)
    )
    clean_f0 = np.zeros_like(f0_hz, dtype=np.float32)
    clean_f0[reliable] = np.asarray(f0_hz[reliable], dtype=np.float32)
    return melody_from_f0(
        clean_f0,
        resolved.hop_length / resolved.sample_rate,
        resolved,
    )


def _constrained_subsequence_dtw(query: FloatArray, reference: FloatArray) -> _Alignment:
    """Align with insert/delete steps but no unbounded one-sided repetitions."""

    query_values = np.asarray(query, dtype=np.float32).reshape(-1)
    reference_values = np.asarray(reference, dtype=np.float32).reshape(-1)
    if query_values.size < 2 or reference_values.size < 2:
        raise MelodyContourError("DTW needs at least two values in each sequence")

    use_subsequence = len(query_values) <= len(reference_values)
    # Diagonal, one-query/two-reference, and two-query/one-reference steps allow
    # skipped or inserted notes while bounding the local tempo ratio to 1:2.
    steps = np.asarray(((1, 1), (1, 2), (2, 1)), dtype=np.uint32)
    try:
        _, path = librosa.sequence.dtw(
            X=query_values[np.newaxis, :],
            Y=reference_values[np.newaxis, :],
            metric="euclidean",
            subseq=use_subsequence,
            backtrack=True,
            step_sizes_sigma=steps,
        )
    except librosa.util.exceptions.ParameterError:
        # A small number of public CHAD reference hums contain only a handful
        # of stable notes while their query contains dozens.  No path can obey
        # the 1:2 slope in that case.  Default DTW is the honest, total-coverage
        # fallback; normal full-song catalog references retain the constraint.
        _, path = librosa.sequence.dtw(
            X=query_values[np.newaxis, :],
            Y=reference_values[np.newaxis, :],
            metric="euclidean",
            subseq=use_subsequence,
            backtrack=True,
        )
    if path.size == 0:
        raise RuntimeError("librosa returned an empty melody alignment")
    query_indices = path[:, 0]
    reference_indices = path[:, 1]
    costs = np.abs(query_values[query_indices] - reference_values[reference_indices])
    return _Alignment(
        distance=float(np.mean(costs)),
        reference_start=int(reference_indices.min()),
        reference_end=int(reference_indices.max()),
    )


def _ngram_counter(values: FloatArray, size: int) -> Counter[tuple[int, ...]]:
    tokens = np.rint(values).astype(np.int16).tolist()
    return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _ngram_similarity(
    query: FloatArray,
    reference: FloatArray,
    sizes: Iterable[int],
) -> float:
    """Return mean multiset Dice overlap for local interval n-grams."""

    similarities: list[float] = []
    for size in sizes:
        query_counter = _ngram_counter(query, size)
        reference_counter = _ngram_counter(reference, size)
        denominator = sum(query_counter.values()) + sum(reference_counter.values())
        if denominator == 0:
            continue
        overlap = sum((query_counter & reference_counter).values())
        similarities.append(2.0 * overlap / denominator)
    return float(np.mean(similarities)) if similarities else 0.0


def _global_shape_dtw(query: FloatArray, reference: FloatArray) -> float:
    """Compare two already-localized relative pitch shapes with global DTW."""

    _, path = librosa.sequence.dtw(
        X=query[np.newaxis, :],
        Y=reference[np.newaxis, :],
        metric="euclidean",
        subseq=False,
        backtrack=True,
    )
    costs = np.abs(query[path[:, 0]] - reference[path[:, 1]])
    return float(np.mean(np.minimum(costs, 12.0)))


def _relative_log_durations(durations: FloatArray) -> FloatArray:
    """Normalize note lengths by local median and express them in log space."""

    values = np.asarray(durations, dtype=np.float32)
    normalized = np.log(np.maximum(values, 1e-4) / np.median(values))
    return np.clip(normalized, -2.0, 2.0).astype(np.float32, copy=False)


def _global_rhythm_dtw(query: FloatArray, reference: FloatArray) -> float:
    """Return a tempo-invariant distance between normalized note lengths."""

    _, path = librosa.sequence.dtw(
        X=query[np.newaxis, :],
        Y=reference[np.newaxis, :],
        metric="euclidean",
        subseq=False,
        backtrack=True,
    )
    costs = np.abs(query[path[:, 0]] - reference[path[:, 1]])
    return float(np.mean(np.minimum(costs, 2.0)))


def _score_melodies(
    query: MelodySequence,
    reference: MelodySequence,
    settings: MelodySettings,
) -> tuple[float, float, float, float, float, float, int, int]:
    query_intervals = query.intervals
    reference_intervals = reference.intervals
    alignment = _constrained_subsequence_dtw(query_intervals, reference_intervals)

    # Include a small margin because an interval index addresses the notes on
    # both sides of it.  Restricting n-grams to this neighborhood avoids giving
    # long songs an unfair chance to collect matches from unrelated sections.
    margin = max(settings.interval_ngram_sizes)
    local_start = max(0, alignment.reference_start - margin)
    local_end = min(len(reference_intervals), alignment.reference_end + margin + 1)
    local_intervals = reference_intervals[local_start:local_end]
    ngram_similarity = _ngram_similarity(
        query_intervals,
        local_intervals,
        settings.interval_ngram_sizes,
    )

    query_shape = query.relative_shape
    reference_shape = reference.relative_shape[local_start : local_end + 1].copy()
    query_shape -= np.median(query_shape)
    reference_shape -= np.median(reference_shape)
    shape_distance = _global_shape_dtw(query_shape, reference_shape)

    query_rhythm = _relative_log_durations(query.durations_seconds)
    reference_rhythm = _relative_log_durations(
        reference.durations_seconds[local_start : local_end + 1]
    )
    rhythm_distance = _global_rhythm_dtw(query_rhythm, reference_rhythm)
    direction_distance = _constrained_subsequence_dtw(
        np.sign(query_intervals).astype(np.float32),
        np.sign(reference_intervals).astype(np.float32),
    ).distance

    total_distance = (
        alignment.distance
        + settings.ngram_mismatch_weight * (1.0 - ngram_similarity)
        + settings.shape_distance_weight * shape_distance
        + settings.rhythm_distance_weight * rhythm_distance
        + settings.direction_distance_weight * direction_distance
    )
    return (
        float(total_distance),
        alignment.distance,
        ngram_similarity,
        shape_distance,
        rhythm_distance,
        direction_distance,
        alignment.reference_start,
        alignment.reference_end + 2,
    )


class MelodyContourIndex:
    """In-memory search index for a small or medium catalog of song melodies."""

    def __init__(
        self,
        references: Sequence[ReferenceMelody],
        settings: MelodySettings | None = None,
    ) -> None:
        if not references:
            raise ValueError("melody index cannot be empty")
        reference_keys = [
            (reference.song_id, str(reference.contour_path).casefold()) for reference in references
        ]
        if len(reference_keys) != len(set(reference_keys)):
            raise ValueError("melody index cannot contain the same reference twice")
        self.references = tuple(references)
        self.settings = settings or MelodySettings()

    @classmethod
    def from_contour_manifest(
        cls,
        manifest_path: str | Path,
        settings: MelodySettings | None = None,
    ) -> MelodyContourIndex:
        resolved_settings = settings or MelodySettings()
        manifest = Path(manifest_path).expanduser().resolve()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        references: list[ReferenceMelody] = []
        for record in payload["songs"]:
            contour_path = Path(record["contour_path"]).expanduser()
            if not contour_path.is_absolute():
                contour_path = manifest.parent / contour_path
            with np.load(contour_path, allow_pickle=False) as archive:
                f0_hz = np.asarray(archive["f0_hz"], dtype=np.float32)
                metadata = json.loads(str(archive["metadata_json"]))
            melody = melody_from_f0(
                f0_hz,
                float(metadata["hop_seconds"]),
                resolved_settings,
            )
            references.append(
                ReferenceMelody(
                    song_id=str(record["song_id"]),
                    title=str(record["title"]),
                    contour_path=contour_path.resolve(),
                    melody=melody,
                )
            )
        return cls(references, resolved_settings)

    def search_sequence(
        self,
        query: MelodySequence,
        *,
        top_k: int = 10,
    ) -> list[MelodyMatch]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored: list[
            tuple[
                ReferenceMelody,
                tuple[float, float, float, float, float, float, int, int],
            ]
        ] = []
        for reference in self.references:
            scored.append(
                (
                    reference,
                    _score_melodies(query, reference.melody, self.settings),
                )
            )
        scored.sort(key=lambda item: (item[1][0], item[0].song_id))

        results: list[MelodyMatch] = []
        returned_song_ids: set[str] = set()
        for reference, score in scored:
            # Public datasets and future catalogs may hold multiple melody
            # sections or versions for one song.  The closest reference wins,
            # but a song must appear only once in the ranked list.
            if reference.song_id in returned_song_ids:
                continue
            returned_song_ids.add(reference.song_id)
            rank = len(results) + 1
            total, interval_distance, ngram, shape, rhythm, direction, start, end = score
            results.append(
                MelodyMatch(
                    rank=rank,
                    song_id=reference.song_id,
                    title=reference.title,
                    distance=total,
                    interval_dtw_distance=interval_distance,
                    ngram_similarity=ngram,
                    shape_dtw_distance=shape,
                    rhythm_dtw_distance=rhythm,
                    direction_dtw_distance=direction,
                    reference_start_note=start,
                    reference_end_note=end,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def search_audio(self, audio_path: str | Path, *, top_k: int = 10) -> list[MelodyMatch]:
        query = extract_hum_melody(audio_path, self.settings)
        return self.search_sequence(query, top_k=top_k)
