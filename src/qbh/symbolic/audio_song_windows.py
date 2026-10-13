# This file makes short reference windows from the full songs.
"""Equal-count full-song window retrieval for uncovered melody sections.

CHAD supplies only one to five short reference excerpts per selected song.  A
person may hum a perfectly recognizable hook that is absent from those anchors.
This experimental backend samples the same fixed number of generic windows from
every local MP3, preserving key invariance and coarse temporal order while
avoiding hand-picked, label-dependent chorus locations.
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
    extract_cached_chroma,
    prepare_chroma,
    score_chroma_tracks,
)
from qbh.symbolic.audio_multiscale import MultiScaleChromaConfig, prepare_multiscale_chroma
from qbh.symbolic.closed_catalog import load_closed_catalog_config
from qbh.symbolic.evaluation import EvaluationError, RankedCandidate, canonical_sha256
from qbh.utils import file_sha256


@dataclass(frozen=True)
class SongAudioSource:
    song_id: str
    title: str
    audio_path: Path
    source_sha256: str


@dataclass(frozen=True)
class SongWindowChromaConfig:
    """Generic window-bank and conservative score-aggregation settings."""

    audio: AudioChromaConfig = AudioChromaConfig()  # noqa: RUF009
    window_seconds: tuple[float, ...] = (12.0,)
    windows_per_duration: int = 24
    query_crop_fractions: tuple[float, ...] = (1.0, 0.75, 0.5)
    query_aggregation_top_k: int = 1
    song_aggregation_top_k: int = 1

    def __post_init__(self) -> None:
        if not self.window_seconds or any(value < 4.0 for value in self.window_seconds):
            raise ValueError("song window durations must all be at least four seconds")
        if tuple(sorted(set(self.window_seconds))) != self.window_seconds:
            raise ValueError("song window durations must be unique and ascending")
        if self.windows_per_duration < 2:
            raise ValueError("song windows_per_duration must be at least two")
        MultiScaleChromaConfig(
            audio=self.audio,
            crop_fractions=self.query_crop_fractions,
            aggregation_top_k=self.query_aggregation_top_k,
        )
        if self.song_aggregation_top_k < 1:
            raise ValueError("song aggregation top-k must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> SongWindowChromaConfig:
        values = values or {}
        return cls(
            audio=AudioChromaConfig.from_mapping(values),
            window_seconds=tuple(float(value) for value in values.get("window_seconds", (12.0,))),
            windows_per_duration=int(values.get("windows_per_duration", 24)),
            query_crop_fractions=tuple(
                float(value) for value in values.get("query_crop_fractions", (1.0, 0.75, 0.5))
            ),
            query_aggregation_top_k=int(values.get("query_aggregation_top_k", 1)),
            song_aggregation_top_k=int(values.get("song_aggregation_top_k", 1)),
        )


def load_song_audio_sources(catalog_config: str | Path) -> list[SongAudioSource]:
    """Load every declared local MP3 with stable identity and content hash."""

    catalog = load_closed_catalog_config(catalog_config)
    sources: list[SongAudioSource] = []
    for song in catalog["songs"]:
        path = (catalog["catalog_root"] / song["audio"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"catalog song audio is missing: {path}")
        sources.append(
            SongAudioSource(
                song_id=str(song["song_id"]),
                title=str(song["title"]),
                audio_path=path,
                source_sha256=file_sha256(path),
            )
        )
    return sorted(sources, key=lambda source: source.song_id)


def even_window_ranges(
    frame_count: int, window_frames: int, count: int
) -> tuple[tuple[int, int], ...]:
    """Return exactly ``count`` evenly spaced windows without label knowledge."""

    if frame_count < 2 or window_frames < 2 or count < 1:
        raise ValueError("song window dimensions/count must be positive")
    width = min(frame_count, window_frames)
    maximum_start = frame_count - width
    if maximum_start + 1 < count:
        raise ValueError("song is too short for the requested number of distinct windows")
    starts = np.rint(np.linspace(0, maximum_start, count)).astype(np.int64)
    if len({int(value) for value in starts}) != count:
        raise ValueError("even song-window sampling produced duplicate starts")
    return tuple((int(start), int(start + width)) for start in starts)


class FullSongWindowChromaBackend:
    """Rank songs by equal-count generic local windows from their full MP3s."""

    name = "full_song_window_chroma"

    def __init__(
        self,
        sources: Sequence[SongAudioSource],
        cache_directory: str | Path,
        config: SongWindowChromaConfig,
        *,
        selection_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not sources:
            raise EvaluationError("full-song window backend needs audio sources")
        self.sources = tuple(sources)
        self.cache_directory = Path(cache_directory).expanduser().resolve()
        self.config = config
        self.selection_metadata = dict(selection_metadata or {})
        self.candidate_song_ids = tuple(sorted(source.song_id for source in self.sources))
        if len(self.candidate_song_ids) != len(set(self.candidate_song_ids)):
            raise EvaluationError("full-song window source IDs must be unique")

        prepared: list[NDArray[np.float32]] = []
        self.window_metadata: list[dict[str, Any]] = []
        self.source_cache_status: dict[str, int] = {"hit": 0, "built": 0}
        frames_per_second = self.config.audio.sample_rate / self.config.audio.hop_length
        for source in self.sources:
            chroma, digest, status = extract_cached_chroma(
                source.audio_path, self.cache_directory, self.config.audio
            )
            if digest != source.source_sha256:
                raise EvaluationError(f"catalog song audio changed: {source.song_id}")
            self.source_cache_status[status] += 1
            for duration in self.config.window_seconds:
                width = round(float(duration) * frames_per_second)
                ranges = even_window_ranges(
                    chroma.shape[1], width, self.config.windows_per_duration
                )
                for start, end in ranges:
                    prepared.append(
                        prepare_chroma(chroma[:, start:end], self.config.audio.comparison_frames)
                    )
                    self.window_metadata.append(
                        {
                            "song_id": source.song_id,
                            "title": source.title,
                            "duration_seconds": float(duration),
                            "start_frame": start,
                            "end_frame": end,
                            "start_seconds": start / frames_per_second,
                            "end_seconds": end / frames_per_second,
                        }
                    )
        self.window_chroma = np.stack(prepared)
        expected_per_song = len(config.window_seconds) * config.windows_per_duration
        counts = Counter(metadata["song_id"] for metadata in self.window_metadata)
        if set(counts.values()) != {expected_per_song}:
            raise EvaluationError("full-song window counts are not equal across candidates")

    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "config": asdict(self.config),
            "selection": self.selection_metadata,
            "cache_directory": str(self.cache_directory),
            "source_cache_status": self.source_cache_status,
            "windows": len(self.window_metadata),
            "windows_per_song": len(self.config.window_seconds) * self.config.windows_per_duration,
            "sources_sha256": canonical_sha256(
                [
                    {
                        "song_id": source.song_id,
                        "source_sha256": source.source_sha256,
                    }
                    for source in self.sources
                ]
            ),
        }

    def rank(self, query: Any) -> Sequence[RankedCandidate]:
        if not isinstance(query, Mapping) or not query.get("audio_path"):
            raise EvaluationError("full-song window query must provide audio_path")
        chroma, _, cache_status = extract_cached_chroma(
            str(query["audio_path"]), self.cache_directory, self.config.audio
        )
        query_config = MultiScaleChromaConfig(
            audio=self.config.audio,
            crop_fractions=self.config.query_crop_fractions,
            aggregation_top_k=self.config.query_aggregation_top_k,
        )
        query_crops, query_ranges = prepare_multiscale_chroma(chroma, query_config)
        by_query: list[NDArray[np.float32]] = []
        by_query_shifts: list[NDArray[np.int16]] = []
        for query_crop in query_crops:
            scores, shifts = score_chroma_tracks(
                query_crop,
                self.window_chroma,
                transposition_invariant=self.config.audio.transposition_invariant,
            )
            by_query.append(scores)
            by_query_shifts.append(shifts)
        score_matrix = np.stack(by_query)
        shift_matrix = np.stack(by_query_shifts)
        query_k = min(self.config.query_aggregation_top_k, score_matrix.shape[0])
        strongest_query = np.argpartition(score_matrix, -query_k, axis=0)[-query_k:]
        window_scores = np.take_along_axis(score_matrix, strongest_query, axis=0).mean(axis=0)
        best_query = np.argmax(score_matrix, axis=0)
        window_indices = np.arange(score_matrix.shape[1])
        best_shifts = shift_matrix[best_query, window_indices]

        windows_by_song: dict[str, list[int]] = {}
        for index, metadata in enumerate(self.window_metadata):
            windows_by_song.setdefault(str(metadata["song_id"]), []).append(index)
        candidates: list[RankedCandidate] = []
        for song_id in self.candidate_song_ids:
            indices = np.asarray(windows_by_song[song_id], dtype=np.int64)
            song_scores = window_scores[indices]
            song_k = min(self.config.song_aggregation_top_k, len(indices))
            strongest_local = np.argpartition(song_scores, -song_k)[-song_k:]
            score = float(np.mean(song_scores[strongest_local]))
            best_local = int(np.argmax(song_scores))
            best_window_index = int(indices[best_local])
            metadata = self.window_metadata[best_window_index]
            query_crop_index = int(best_query[best_window_index])
            candidates.append(
                RankedCandidate(
                    song_id=song_id,
                    track_id=(
                        f"{song_id}:window:{metadata['start_frame']}-{metadata['end_frame']}"
                    ),
                    title=str(metadata["title"]),
                    score=score,
                    confidence=float(np.clip(score, 0.0, 1.0)),
                    details={
                        "backend": self.name,
                        "query_cache_status": cache_status,
                        "query_transposition_semitones": int(best_shifts[best_window_index]),
                        "best_query_crop_frames": query_ranges[query_crop_index],
                        "best_window_start_seconds": float(metadata["start_seconds"]),
                        "best_window_end_seconds": float(metadata["end_seconds"]),
                        "window_duration_seconds": float(metadata["duration_seconds"]),
                        "query_aggregation_top_k": self.config.query_aggregation_top_k,
                        "song_aggregation_top_k": self.config.song_aggregation_top_k,
                    },
                )
            )
        return sorted(candidates, key=lambda item: (-item.score, item.song_id))


# Kept local to avoid importing collections in callers and to make the equal-
# window invariant explicit in this module's construction path.
from collections import Counter

__all__ = [
    "FullSongWindowChromaBackend",
    "SongAudioSource",
    "SongWindowChromaConfig",
    "even_window_ranges",
    "load_song_audio_sources",
]
