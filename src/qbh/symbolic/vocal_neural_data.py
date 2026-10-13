# This file builds the tensors used by the vocal model.
"""Leakage-safe, resumable data cache for neural hum-to-vocal matching.

CHAD supplies a song label for each humming query, while the searchable catalog
contains 24 evenly sampled Demucs-vocal windows per song.  This module turns that
weak supervision into multiple-instance training bags: one correct song and hard
negative songs, each represented by the same number of its strongest baseline
windows. Validation bags always contain every configured catalog candidate.

Every query is cached as an independent content-addressed shard.  An interrupted
build therefore resumes at the next missing hum instead of recomputing chroma for
all recordings.  The public test JSONL is deliberately rejected here; model and
configuration selection may use only CHAD train and validation queries.
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

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from qbh.symbolic.audio_local import extract_cached_chroma, score_chroma_tracks
from qbh.symbolic.audio_multiscale import MultiScaleChromaConfig, prepare_multiscale_chroma
from qbh.symbolic.audio_song_windows import FullSongWindowChromaBackend
from qbh.symbolic.chad_sequences import read_symbolic_pairs
from qbh.symbolic.evaluation import EvaluationError, canonical_sha256
from qbh.utils import file_sha256


@dataclass(frozen=True)
class VocalNeuralDataConfig:
    """Candidate mining and cached tensor settings selected before validation."""

    training_candidates: int = 12
    windows_per_candidate: int = 4
    query_crop_fractions: tuple[float, ...] = (1.0, 0.75, 0.5)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> VocalNeuralDataConfig:
        values = values or {}
        result = cls(
            training_candidates=int(
                values.get("training_candidates", cls.training_candidates)
            ),
            windows_per_candidate=int(
                values.get("windows_per_candidate", cls.windows_per_candidate)
            ),
            query_crop_fractions=tuple(
                float(value)
                for value in values.get("query_crop_fractions", cls.query_crop_fractions)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.training_candidates < 2 or self.windows_per_candidate < 1:
            raise ValueError("neural vocal data needs at least two songs and one window")
        if (
            not self.query_crop_fractions
            or any(not 0.0 < value <= 1.0 for value in self.query_crop_fractions)
            or tuple(sorted(set(self.query_crop_fractions), reverse=True))
            != self.query_crop_fractions
        ):
            raise ValueError("query crop fractions must be unique and descending in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CachedVocalNeuralDataset(Dataset[dict[str, torch.Tensor]]):
    """Eager in-memory view of small, independently resumable query shards."""

    def __init__(
        self,
        shard_paths: Sequence[Path],
        *,
        query_ids: Sequence[str],
        target_song_ids: Sequence[str],
        candidate_song_ids: Sequence[str],
    ) -> None:
        if not shard_paths or len(shard_paths) != len(query_ids):
            raise ValueError("vocal neural dataset needs one shard per query")
        loaded: dict[str, list[NDArray[Any]]] = {
            key: []
            for key in (
                "query",
                "vocal",
                "raw_similarity",
                "key_proximity",
                "candidate_indices",
                "target_index",
                "window_indices",
                "shifts",
            )
        }
        for path in shard_paths:
            with np.load(path, allow_pickle=False) as shard:
                for key, values in loaded.items():
                    values.append(np.asarray(shard[key]))
        self.query = torch.from_numpy(np.stack(loaded["query"]))
        self.vocal = torch.from_numpy(np.stack(loaded["vocal"]))
        self.raw_similarity = torch.from_numpy(np.stack(loaded["raw_similarity"]))
        self.key_proximity = torch.from_numpy(np.stack(loaded["key_proximity"]))
        self.candidate_indices = torch.from_numpy(np.stack(loaded["candidate_indices"]))
        self.target_index = torch.from_numpy(np.stack(loaded["target_index"])).long().reshape(-1)
        self.window_indices = torch.from_numpy(np.stack(loaded["window_indices"]))
        self.shifts = torch.from_numpy(np.stack(loaded["shifts"]))
        self.query_ids = tuple(query_ids)
        self.target_song_ids = tuple(target_song_ids)
        self.candidate_song_ids = tuple(candidate_song_ids)

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "query": self.query[index],
            "vocal": self.vocal[index],
            "raw_similarity": self.raw_similarity[index],
            "key_proximity": self.key_proximity[index],
            "candidate_indices": self.candidate_indices[index],
            "target_index": self.target_index[index],
            "window_indices": self.window_indices[index],
            "shifts": self.shifts[index],
        }


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_shard(path: Path, fingerprint: str, candidates: int, windows: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as shard:
            return (
                str(shard["fingerprint"].item()) == fingerprint
                and shard["query"].shape[:2] == (candidates, windows)
                and shard["query"].shape == shard["vocal"].shape
                and shard["query"].shape[2] == 12
                and shard["raw_similarity"].shape == (candidates, windows)
                and shard["key_proximity"].shape == (candidates, windows)
                and shard["candidate_indices"].shape == (candidates,)
                and shard["window_indices"].shape == (candidates, windows)
                and shard["shifts"].shape == (candidates, windows)
                and np.isfinite(shard["query"]).all()
                and np.isfinite(shard["vocal"]).all()
                and np.isfinite(shard["raw_similarity"]).all()
            )
    except (OSError, KeyError, ValueError):
        return False


def _windows_by_song(backend: FullSongWindowChromaBackend) -> dict[str, NDArray[np.int64]]:
    grouped: dict[str, list[int]] = {song_id: [] for song_id in backend.candidate_song_ids}
    for index, metadata in enumerate(backend.window_metadata):
        grouped[str(metadata["song_id"])].append(index)
    result = {
        song_id: np.asarray(indices, dtype=np.int64) for song_id, indices in grouped.items()
    }
    if not result or len({len(indices) for indices in result.values()}) != 1:
        raise EvaluationError("neural vocal candidate songs must have equal window counts")
    return result


def _stable_reference_identity(provenance: Mapping[str, Any]) -> str:
    """Hash only fields that can change the reference tensors.

    Backend provenance also reports where its extraction cache lives and whether
    each source was a cache hit or build. Those operational details are useful in
    reports, but they must not invalidate query shards between the first run
    (sources built) and a resumed run (sources hit).
    """

    stable = dict(provenance)
    stable.pop("cache_directory", None)
    stable.pop("source_cache_status", None)
    stable.pop("reference_cache_status", None)
    return canonical_sha256(stable)


def prepare_vocal_neural_query(
    query_path: str | Path,
    *,
    backend: FullSongWindowChromaBackend,
    cache_directory: str | Path,
    data_config: VocalNeuralDataConfig | None = None,
    priority_window_indices_by_song: Mapping[str, Sequence[int]] | None = None,
    priority_window_groups_by_song: (
        Mapping[str, Sequence[Sequence[int]]] | None
    ) = None,
    priority_windows_per_candidate: int = 2,
) -> dict[str, NDArray[Any]]:
    """Prepare all catalog candidate bags for one new, unlabeled humming query.

    This is the inference counterpart of the training shard builder.  It uses
    every candidate song, retains the same number of top baseline windows, and
    returns window identities so a browser can explain the winning location.
    No label or expected song is accepted by this function.
    """

    config = data_config or VocalNeuralDataConfig()
    config.validate()
    path = Path(query_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"neural vocal query does not exist: {path}")
    extraction_cache = Path(cache_directory).expanduser().resolve()
    chroma, _, _ = extract_cached_chroma(path, extraction_cache, backend.config.audio)
    multiscale = MultiScaleChromaConfig(
        audio=backend.config.audio,
        crop_fractions=config.query_crop_fractions,
        aggregation_top_k=1,
    )
    query_crops, _ = prepare_multiscale_chroma(chroma, multiscale)
    by_crop_scores: list[NDArray[np.float32]] = []
    by_crop_shifts: list[NDArray[np.int16]] = []
    for crop in query_crops:
        scores, shifts = score_chroma_tracks(
            crop,
            backend.window_chroma,
            transposition_invariant=backend.config.audio.transposition_invariant,
        )
        by_crop_scores.append(scores)
        by_crop_shifts.append(shifts)
    score_matrix = np.stack(by_crop_scores)
    shift_matrix = np.stack(by_crop_shifts)
    best_crop = np.argmax(score_matrix, axis=0)
    window_axis = np.arange(score_matrix.shape[1])
    best_scores = score_matrix[best_crop, window_axis]
    best_shifts = shift_matrix[best_crop, window_axis]
    grouped_windows = _windows_by_song(backend)

    candidate_song_ids = tuple(backend.candidate_song_ids)
    candidates = len(candidate_song_ids)
    extra_windows = (
        priority_windows_per_candidate
        if priority_window_indices_by_song is not None
        else 0
    )
    if extra_windows < 0:
        raise ValueError("priority windows per candidate cannot be negative")
    windows = config.windows_per_candidate + extra_windows
    frames = int(backend.config.audio.comparison_frames)
    query_values = np.empty((candidates, windows, 12, frames), dtype=np.float16)
    vocal_values = np.empty_like(query_values)
    raw_values = np.empty((candidates, windows), dtype=np.float32)
    proximity_values = np.empty_like(raw_values)
    selected_windows = np.empty((candidates, windows), dtype=np.int16)
    selected_shifts = np.empty((candidates, windows), dtype=np.int8)

    for candidate_index, song_id in enumerate(candidate_song_ids):
        available = grouped_windows[song_id]
        ordered = sorted(available.tolist(), key=lambda index: (-best_scores[index], index))
        # Preserve the model's original four strongest generic windows first;
        # they are the exact input distribution used during training. Append
        # the strongest documented-chorus windows only for the separate
        # chorus-restricted melodic decision. This prevents a true chorus from
        # disappearing merely because four unrelated windows scored slightly
        # higher during the initial chroma screening.
        chosen = ordered[: config.windows_per_candidate]
        if priority_window_indices_by_song is not None:
            allowed = {int(index) for index in available}
            priority = [
                int(index)
                for index in priority_window_indices_by_song.get(song_id, ())
                if int(index) in allowed
            ]
            priority.sort(key=lambda index: (-best_scores[index], index))
            # When callers provide distinct chorus-occurrence groups, retain
            # the best acoustic window from every occurrence before filling
            # the remaining slots. This prevents six overlapping windows from
            # the first chorus from hiding the second and third choruses.
            representatives: list[int] = []
            if priority_window_groups_by_song is not None:
                for group in priority_window_groups_by_song.get(song_id, ()):
                    valid_group = [
                        int(index)
                        for index in group
                        if int(index) in allowed and int(index) in priority
                    ]
                    if valid_group:
                        representatives.append(
                            max(
                                valid_group,
                                key=lambda index: (best_scores[index], -index),
                            )
                        )
            chosen.extend(
                index
                for index in (*representatives, *priority)
                if index not in chosen
            )
            chosen = chosen[:windows]
            chosen.extend(index for index in ordered if index not in chosen)
            chosen = chosen[:windows]
        if len(chosen) != windows:
            raise EvaluationError(f"not enough vocal windows for {song_id}")
        for window_position, window_index in enumerate(chosen):
            crop_index = int(best_crop[window_index])
            shift = int(best_shifts[window_index])
            query_values[candidate_index, window_position] = np.roll(
                query_crops[crop_index], shift, axis=0
            ).astype(np.float16)
            vocal_values[candidate_index, window_position] = backend.window_chroma[
                window_index
            ].astype(np.float16)
            raw_values[candidate_index, window_position] = best_scores[window_index]
            distance = min(shift % 12, (-shift) % 12)
            proximity_values[candidate_index, window_position] = 1.0 - distance / 6.0
            selected_windows[candidate_index, window_position] = window_index
            selected_shifts[candidate_index, window_position] = shift
    return {
        "query": query_values,
        "vocal": vocal_values,
        "raw_similarity": raw_values,
        "key_proximity": proximity_values,
        "candidate_indices": np.arange(candidates, dtype=np.int16),
        "window_indices": selected_windows,
        "shifts": selected_shifts,
    }


def _build_query_shard(
    *,
    record: Mapping[str, Any],
    split: str,
    query_path: Path,
    backend: FullSongWindowChromaBackend,
    windows_by_song: Mapping[str, NDArray[np.int64]],
    candidate_song_ids: Sequence[str],
    data_config: VocalNeuralDataConfig,
    extraction_cache: Path,
    shard_path: Path,
    fingerprint: str,
) -> None:
    chroma, _, _ = extract_cached_chroma(query_path, extraction_cache, backend.config.audio)
    multiscale = MultiScaleChromaConfig(
        audio=backend.config.audio,
        crop_fractions=data_config.query_crop_fractions,
        aggregation_top_k=1,
    )
    query_crops, _ = prepare_multiscale_chroma(chroma, multiscale)
    by_crop_scores: list[NDArray[np.float32]] = []
    by_crop_shifts: list[NDArray[np.int16]] = []
    for crop in query_crops:
        scores, shifts = score_chroma_tracks(
            crop,
            backend.window_chroma,
            transposition_invariant=backend.config.audio.transposition_invariant,
        )
        by_crop_scores.append(scores)
        by_crop_shifts.append(shifts)
    score_matrix = np.stack(by_crop_scores)
    shift_matrix = np.stack(by_crop_shifts)
    best_crop = np.argmax(score_matrix, axis=0)
    window_axis = np.arange(score_matrix.shape[1])
    best_scores = score_matrix[best_crop, window_axis]
    best_shifts = shift_matrix[best_crop, window_axis]

    global_song_index = {song_id: index for index, song_id in enumerate(candidate_song_ids)}
    target_song_id = str(record["song_id"])
    target_global_index = global_song_index[target_song_id]
    song_strength = np.asarray(
        [float(best_scores[windows_by_song[song_id]].max()) for song_id in candidate_song_ids]
    )
    if split == "train":
        wrong = [index for index in np.argsort(-song_strength) if index != target_global_index]
        candidate_indices = np.asarray(
            [target_global_index, *wrong[: data_config.training_candidates - 1]],
            dtype=np.int16,
        )
        target_index = 0
    else:
        candidate_indices = np.arange(len(candidate_song_ids), dtype=np.int16)
        target_index = target_global_index

    candidates = len(candidate_indices)
    windows = data_config.windows_per_candidate
    frames = int(backend.config.audio.comparison_frames)
    query_values = np.empty((candidates, windows, 12, frames), dtype=np.float16)
    vocal_values = np.empty_like(query_values)
    raw_values = np.empty((candidates, windows), dtype=np.float32)
    proximity_values = np.empty_like(raw_values)
    selected_windows = np.empty((candidates, windows), dtype=np.int16)
    selected_shifts = np.empty((candidates, windows), dtype=np.int8)

    for candidate_position, global_index in enumerate(candidate_indices):
        song_id = candidate_song_ids[int(global_index)]
        available = windows_by_song[song_id]
        ordered = sorted(available.tolist(), key=lambda index: (-best_scores[index], index))
        chosen = ordered[:windows]
        if len(chosen) != windows:
            raise EvaluationError(f"not enough vocal windows for {song_id}")
        for window_position, window_index in enumerate(chosen):
            crop_index = int(best_crop[window_index])
            shift = int(best_shifts[window_index])
            query_values[candidate_position, window_position] = np.roll(
                query_crops[crop_index], shift, axis=0
            ).astype(np.float16)
            vocal_values[candidate_position, window_position] = backend.window_chroma[
                window_index
            ].astype(np.float16)
            raw_values[candidate_position, window_position] = best_scores[window_index]
            distance = min(shift % 12, (-shift) % 12)
            proximity_values[candidate_position, window_position] = 1.0 - distance / 6.0
            selected_windows[candidate_position, window_position] = window_index
            selected_shifts[candidate_position, window_position] = shift

    _atomic_npz(
        shard_path,
        fingerprint=np.asarray(fingerprint),
        query=query_values,
        vocal=vocal_values,
        raw_similarity=raw_values,
        key_proximity=proximity_values,
        candidate_indices=candidate_indices,
        target_index=np.asarray(target_index, dtype=np.int16),
        window_indices=selected_windows,
        shifts=selected_shifts,
    )


def build_vocal_neural_dataset(
    *,
    sequence_path: str | Path,
    audio_root: str | Path,
    allowed_target_song_ids: Sequence[str],
    backend: FullSongWindowChromaBackend,
    cache_directory: str | Path,
    split: str,
    data_config: VocalNeuralDataConfig | None = None,
) -> tuple[CachedVocalNeuralDataset, dict[str, Any]]:
    """Build or resume one train/validation dataset without opening test labels."""

    if split not in {"train", "validation"}:
        raise ValueError("vocal neural data may use only train or validation queries")
    config = data_config or VocalNeuralDataConfig()
    config.validate()
    if split == "train" and config.training_candidates > len(backend.candidate_song_ids):
        raise ValueError("training candidate count exceeds the vocal catalog")
    if config.windows_per_candidate > backend.config.windows_per_duration * len(
        backend.config.window_seconds
    ):
        raise ValueError("requested neural windows exceed the indexed vocal windows")

    path = Path(sequence_path).expanduser().resolve()
    if "test" in path.name.lower() or path.stem != split:
        raise ValueError("sequence filename must agree with the non-test split")
    records = read_symbolic_pairs(path)
    allowed = set(allowed_target_song_ids)
    selected = sorted(
        (record for record in records if str(record["song_id"]) in allowed),
        key=lambda record: str(record["query_id"]),
    )
    if not selected or {str(record.get("split")) for record in selected} != {split}:
        raise EvaluationError(f"no consistent selected CHAD records for {split}")
    if {str(record["song_id"]) for record in selected} != allowed:
        raise EvaluationError(f"selected {split} query songs differ from the catalog split")

    candidate_song_ids = tuple(backend.candidate_song_ids)
    if not allowed.issubset(candidate_song_ids):
        raise EvaluationError("query targets are absent from the vocal catalog")
    grouped_windows = _windows_by_song(backend)
    cache_root = Path(cache_directory).expanduser().resolve()
    extraction_cache = cache_root / "chroma"
    shard_root = cache_root / "query_shards" / split
    audio_base = Path(audio_root).expanduser().resolve()
    reference_identity = _stable_reference_identity(backend.provenance())
    sequence_sha = file_sha256(path)
    query_ids: list[str] = []
    target_song_ids: list[str] = []
    shards: list[Path] = []
    statuses = {"hit": 0, "built": 0}

    for record in selected:
        query_id = str(record["query_id"])
        query_path = (
            audio_base
            / str(record["song_id"])
            / str(record["fragment_id"])
            / f"{query_id}.wav"
        ).resolve()
        if not query_path.is_file():
            raise FileNotFoundError(f"CHAD neural humming query is missing: {query_path}")
        identity = {
            "format_version": 1,
            "split": split,
            "query_id": query_id,
            "target_song_id": str(record["song_id"]),
            "query_sha256": file_sha256(query_path),
            "sequence_sha256": sequence_sha,
            "reference_identity": reference_identity,
            "data_config": config.to_dict(),
        }
        fingerprint = _fingerprint(identity)
        shard_path = shard_root / fingerprint[:2] / f"{fingerprint}.npz"
        candidate_count = (
            config.training_candidates if split == "train" else len(candidate_song_ids)
        )
        if _valid_shard(
            shard_path, fingerprint, candidate_count, config.windows_per_candidate
        ):
            statuses["hit"] += 1
        else:
            _build_query_shard(
                record=record,
                split=split,
                query_path=query_path,
                backend=backend,
                windows_by_song=grouped_windows,
                candidate_song_ids=candidate_song_ids,
                data_config=config,
                extraction_cache=extraction_cache,
                shard_path=shard_path,
                fingerprint=fingerprint,
            )
            statuses["built"] += 1
        query_ids.append(query_id)
        target_song_ids.append(str(record["song_id"]))
        shards.append(shard_path)
        completed = len(shards)
        if completed == 1 or completed % 25 == 0 or completed == len(selected):
            print(
                f"{split} vocal-neural cache: {completed}/{len(selected)} "
                f"(hit={statuses['hit']}, built={statuses['built']})",
                flush=True,
            )

    dataset = CachedVocalNeuralDataset(
        shards,
        query_ids=query_ids,
        target_song_ids=target_song_ids,
        candidate_song_ids=candidate_song_ids,
    )
    provenance = {
        "format_version": 1,
        "split": split,
        "queries": len(dataset),
        "query_songs": len(set(target_song_ids)),
        "candidate_songs": len(candidate_song_ids),
        "candidates_per_query": int(dataset.query.shape[1]),
        "windows_per_candidate": int(dataset.query.shape[2]),
        "cache_status": statuses,
        "sequence_path": str(path),
        "sequence_sha256": sequence_sha,
        "reference_identity": reference_identity,
        "data_config": config.to_dict(),
        "shards_sha256": canonical_sha256(
            [file_sha256(shard) for shard in shards]
        ),
    }
    return dataset, provenance


__all__ = [
    "CachedVocalNeuralDataset",
    "VocalNeuralDataConfig",
    "build_vocal_neural_dataset",
    "prepare_vocal_neural_query",
]
