# This file extracts chroma and compares a hum with song audio.
"""Cached, transposition-invariant local-audio matching for CHAD fragments.

Unlike the collapsed global embedding, this backend keeps the short hummed and
song excerpts as time-varying chroma sequences.  Each sequence is normalized to
a small fixed time grid, all twelve key shifts are compared, and the best local
fragment score is retained for each song.  The representation is deliberately
simple: validation showed that constrained DTW and global-histogram fusion both
reduced retrieval quality, while short-time Fourier chroma was substantially
more reliable than symbolic notes on these aligned real-world recordings.

Feature extraction is content-addressed and atomic.  Interrupted evaluations
can resume without decoding audio that already has a valid cache entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from qbh.symbolic.chad_sequences import read_symbolic_pairs
from qbh.symbolic.closed_catalog import load_closed_catalog_config
from qbh.symbolic.evaluation import (
    EvaluationCase,
    EvaluationError,
    RankedCandidate,
    canonical_sha256,
)
from qbh.utils import file_sha256


@dataclass(frozen=True)
class AudioChromaConfig:
    """Extraction and comparison settings frozen after validation selection."""

    sample_rate: int = 11_025
    hop_length: int = 512
    n_fft: int = 2_048
    trim_top_db: float = 35.0
    comparison_frames: int = 32
    chroma_kind: str = "stft"
    transposition_invariant: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate < 4_000 or self.hop_length < 32 or self.n_fft < self.hop_length:
            raise ValueError("invalid chroma sample rate, hop length, or FFT size")
        if self.trim_top_db <= 0 or self.comparison_frames < 8:
            raise ValueError("invalid silence threshold or comparison frame count")
        if self.chroma_kind not in {"stft", "cqt", "cens"}:
            raise ValueError("chroma_kind must be stft, cqt, or cens")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> AudioChromaConfig:
        values = values or {}
        return cls(
            sample_rate=int(values.get("sample_rate", cls.sample_rate)),
            hop_length=int(values.get("hop_length", cls.hop_length)),
            n_fft=int(values.get("n_fft", cls.n_fft)),
            trim_top_db=float(values.get("trim_top_db", cls.trim_top_db)),
            comparison_frames=int(values.get("comparison_frames", cls.comparison_frames)),
            chroma_kind=str(values.get("chroma_kind", cls.chroma_kind)),
            transposition_invariant=bool(
                values.get("transposition_invariant", cls.transposition_invariant)
            ),
        )

    def extraction_dict(self) -> dict[str, Any]:
        """Settings that affect cached raw chroma, excluding score-time resampling."""

        values = asdict(self)
        values.pop("comparison_frames")
        values.pop("transposition_invariant")
        return values


@dataclass(frozen=True)
class AudioReferenceTrack:
    song_id: str
    track_id: str
    title: str
    audio_path: Path
    source_sha256: str


def _fingerprint(source_sha256: str, config: AudioChromaConfig) -> str:
    payload = {
        "format_version": 1,
        "source_sha256": source_sha256,
        "extraction": config.extraction_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_cached_chroma(
    audio_path: str | Path,
    cache_directory: str | Path,
    config: AudioChromaConfig,
) -> tuple[NDArray[np.float32], str, str]:
    """Return ``(chroma, source_hash, cache_status)`` for one public recording."""

    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"audio fragment does not exist: {path}")
    source_sha256 = file_sha256(path)
    fingerprint = _fingerprint(source_sha256, config)
    cache_path = (
        Path(cache_directory).expanduser().resolve() / fingerprint[:2] / (fingerprint + ".npz")
    )
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                chroma = cached["chroma"].astype(np.float32, copy=False)
                cached_hash = str(cached["source_sha256"].item())
                cached_fingerprint = str(cached["fingerprint"].item())
            if (
                chroma.ndim == 2
                and chroma.shape[0] == 12
                and chroma.shape[1] >= 2
                and np.isfinite(chroma).all()
                and cached_hash == source_sha256
                and cached_fingerprint == fingerprint
            ):
                return chroma, source_sha256, "hit"
        except (OSError, KeyError, ValueError):
            # A partial/corrupt entry is safely rebuilt below and atomically replaced.
            pass

    audio, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=config.trim_top_db)
    if audio.size < config.n_fft:
        raise ValueError(f"audio fragment is too short for chroma extraction: {path}")
    if config.chroma_kind == "stft":
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
    elif config.chroma_kind == "cqt":
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=config.sample_rate, hop_length=config.hop_length
        )
    else:
        chroma = librosa.feature.chroma_cens(
            y=audio, sr=config.sample_rate, hop_length=config.hop_length
        )
    chroma = np.asarray(chroma, dtype=np.float32)
    if chroma.shape[0] != 12 or chroma.shape[1] < 2 or not np.isfinite(chroma).all():
        raise ValueError(f"invalid chroma extracted from: {path}")
    _atomic_npz(
        cache_path,
        chroma=chroma,
        source_sha256=np.asarray(source_sha256),
        fingerprint=np.asarray(fingerprint),
    )
    return chroma, source_sha256, "built"


def prepare_chroma(
    chroma: NDArray[np.floating[Any]], comparison_frames: int
) -> NDArray[np.float32]:
    """Tempo-normalize one chromagram while preserving its coarse time order."""

    values = np.asarray(chroma, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 12 or values.shape[1] < 2:
        raise ValueError("chroma must have shape [12, time>=2]")
    resized = np.maximum(resample(values, comparison_frames, axis=1), 0.0)
    normalized = librosa.util.normalize(resized, norm=2, axis=0)
    return normalized.astype(np.float32, copy=False)


def score_chroma_tracks(
    query: NDArray[np.float32],
    references: NDArray[np.float32],
    *,
    transposition_invariant: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.int16]]:
    """Score one prepared query against prepared tracks and return best key shifts."""

    query_values = np.asarray(query, dtype=np.float32)
    reference_values = np.asarray(references, dtype=np.float32)
    if (
        query_values.ndim != 2
        or reference_values.ndim != 3
        or query_values.shape[0] != 12
        or reference_values.shape[1:] != query_values.shape
    ):
        raise ValueError("query/reference prepared chroma shapes do not agree")
    shifts = range(12) if transposition_invariant else range(1)
    by_shift = np.stack(
        [
            np.einsum(
                "cf,ncf->n",
                np.roll(query_values, shift, axis=0),
                reference_values,
            )
            / query_values.shape[1]
            for shift in shifts
        ]
    )
    best = np.argmax(by_shift, axis=0)
    scores = by_shift[best, np.arange(reference_values.shape[0])]
    return scores.astype(np.float32, copy=False), best.astype(np.int16, copy=False)


def load_closed_chad_audio_catalog(
    sequence_directory: str | Path,
    catalog_config: str | Path,
    audio_root: str | Path,
    *,
    query_split: str,
    maximum_queries: int | None = None,
) -> tuple[list[EvaluationCase], list[AudioReferenceTrack], dict[str, Any]]:
    """Build the fixed 30-song audio catalog and one held-out query split."""

    if query_split not in {"validation", "test"}:
        raise EvaluationError("audio closed-catalog queries must use validation or test")
    if maximum_queries is not None and maximum_queries < 1:
        raise EvaluationError("maximum_queries must be positive when provided")
    config = load_closed_catalog_config(catalog_config)
    songs = {song["song_id"]: song for song in config["songs"]}
    sequence_root = Path(sequence_directory).expanduser().resolve()
    fragment_root = Path(audio_root).expanduser().resolve()

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    sequence_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = sequence_root / f"{split}.jsonl"
        records = read_symbolic_pairs(path)
        if {str(record.get("split", "")) for record in records} != {split}:
            raise EvaluationError(f"symbolic JSONL split disagrees with filename: {path}")
        records_by_split[split] = records
        sequence_hashes[split] = file_sha256(path)

    references: dict[str, AudioReferenceTrack] = {}
    observed_splits: dict[str, set[str]] = {song_id: set() for song_id in songs}
    for split, records in records_by_split.items():
        for record in records:
            song_id = str(record["song_id"])
            if song_id not in songs:
                continue
            observed_splits[song_id].add(split)
            reference_id = str(record["reference_id"])
            path = (
                fragment_root / song_id / str(record["fragment_id"]) / f"{reference_id}.wav"
            ).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"CHAD reference audio is missing: {path}")
            existing = references.get(reference_id)
            if existing is not None and (
                existing.song_id != song_id or existing.audio_path != path
            ):
                raise EvaluationError(f"reference ID changed identity: {reference_id}")
            references.setdefault(
                reference_id,
                AudioReferenceTrack(
                    song_id=song_id,
                    track_id=reference_id,
                    title=songs[song_id]["title"],
                    audio_path=path,
                    source_sha256=file_sha256(path),
                ),
            )
    for song_id, observed in observed_splits.items():
        expected = songs[song_id]["split"]
        if observed != {expected}:
            raise EvaluationError(
                f"declared split for {song_id} is {expected}, observed {sorted(observed)}"
            )

    cases: list[EvaluationCase] = []
    for record in sorted(records_by_split[query_split], key=lambda item: str(item["query_id"])):
        song_id = str(record["song_id"])
        if song_id not in songs:
            continue
        query_id = str(record["query_id"])
        path = (fragment_root / song_id / str(record["fragment_id"]) / f"{query_id}.wav").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CHAD humming audio is missing: {path}")
        cases.append(
            EvaluationCase(
                query_id=query_id,
                target_song_id=song_id,
                split=query_split,
                query={"audio_path": str(path)},
                source_sha256=file_sha256(path),
                metadata={
                    "pair_id": str(record["pair_id"]),
                    "fragment_id": str(record["fragment_id"]),
                    "reference_id": str(record["reference_id"]),
                },
            )
        )
    if maximum_queries is not None:
        cases = cases[:maximum_queries]
    tracks = sorted(references.values(), key=lambda item: (item.song_id, item.track_id))
    if not cases or len({track.song_id for track in tracks}) != len(songs):
        raise EvaluationError("audio closed catalog is missing queries or candidate songs")
    provenance = {
        "source": "public_CHAD_fixed_local_audio_catalog",
        "catalog_name": config["name"],
        "catalog_config": str(config["path"]),
        "catalog_config_sha256": file_sha256(config["path"]),
        "query_split": query_split,
        "queries_used": len(cases),
        "query_songs": len({case.target_song_id for case in cases}),
        "candidate_songs": len(songs),
        "reference_tracks": len(tracks),
        "sequence_sha256": sequence_hashes,
        "maximum_queries": maximum_queries,
        "protocol_note": (
            "Original-song fragments are fixed catalog references; only humming "
            "recordings from query_split are evaluated."
        ),
    }
    return cases, tracks, provenance


class AudioChromaLocalBackend:
    """Rank songs by the best transposition-invariant local chroma match."""

    name = "local_audio_chroma"

    def __init__(
        self,
        tracks: Sequence[AudioReferenceTrack],
        cache_directory: str | Path,
        config: AudioChromaConfig | None = None,
        *,
        selection_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not tracks:
            raise EvaluationError("audio chroma backend needs reference tracks")
        self.tracks = tuple(tracks)
        self.cache_directory = Path(cache_directory).expanduser().resolve()
        self.config = config or AudioChromaConfig()
        self.selection_metadata = dict(selection_metadata or {})
        self.candidate_song_ids = tuple(sorted({track.song_id for track in self.tracks}))
        prepared: list[NDArray[np.float32]] = []
        self.reference_cache_status: dict[str, int] = {"hit": 0, "built": 0}
        for track in self.tracks:
            chroma, digest, status = extract_cached_chroma(
                track.audio_path, self.cache_directory, self.config
            )
            if digest != track.source_sha256:
                raise EvaluationError(f"reference audio changed: {track.track_id}")
            self.reference_cache_status[status] += 1
            prepared.append(prepare_chroma(chroma, self.config.comparison_frames))
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
            raise EvaluationError("audio chroma query must provide audio_path")
        chroma, _, cache_status = extract_cached_chroma(
            str(query["audio_path"]), self.cache_directory, self.config
        )
        prepared = prepare_chroma(chroma, self.config.comparison_frames)
        scores, shifts = score_chroma_tracks(
            prepared,
            self.reference_chroma,
            transposition_invariant=self.config.transposition_invariant,
        )
        best_by_song: dict[str, RankedCandidate] = {}
        for track, score, shift in zip(self.tracks, scores, shifts, strict=True):
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
                    "chroma_similarity": float(score),
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
    "AudioChromaConfig",
    "AudioChromaLocalBackend",
    "AudioReferenceTrack",
    "extract_cached_chroma",
    "load_closed_chad_audio_catalog",
    "prepare_chroma",
    "score_chroma_tracks",
]
