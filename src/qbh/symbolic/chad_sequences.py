# This file reads the CHAD split and stores melody sequences.
"""Leakage-safe CHAD pairs represented as symbolic note sequences.

This module deliberately contains no raw-audio neural network.  Audio is
converted once into a compact, human-readable note representation and cached;
subsequent model training reads only JSON Lines files.  The manifest-provided
train/validation/test split remains authoritative and is checked before any
audio is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from qbh.data import assert_song_disjoint
from qbh.melody_retrieval import MelodySequence, extract_hum_melody

SYMBOLIC_SCHEMA_VERSION = 1
FEATURE_NAMES = ("interval_semitones", "relative_log_duration", "confidence", "direction")
REQUIRED_MANIFEST_COLUMNS = {
    "pair_id",
    "song_id",
    "fragment_id",
    "query_id",
    "reference_id",
    "query_path",
    "reference_path",
    "split",
}


class SymbolicSequenceError(ValueError):
    """Raised when a symbolic sequence or its provenance is invalid."""


@dataclass(frozen=True)
class SymbolicSequence:
    """Timestamp-light note sheet used at the learned model boundary.

    MIDI pitch is retained for diagnostics, but model features use adjacent
    intervals.  Therefore two performances in different keys have identical
    pitch features.  Confidence is optional in upstream extractors and defaults
    to one, which keeps this adapter compatible with the project's current
    pYIN/Omnizart contour objects and a richer future shared schema.
    """

    pitches_midi: tuple[float, ...]
    durations_seconds: tuple[float, ...]
    confidences: tuple[float, ...]

    def __post_init__(self) -> None:
        length = len(self.pitches_midi)
        if length < 3:
            raise SymbolicSequenceError("a symbolic sequence needs at least three notes")
        if len(self.durations_seconds) != length or len(self.confidences) != length:
            raise SymbolicSequenceError("pitch, duration, and confidence lengths must match")
        values = self.pitches_midi + self.durations_seconds + self.confidences
        if not all(math.isfinite(float(value)) for value in values):
            raise SymbolicSequenceError("symbolic values must be finite")
        if any(float(value) <= 0 for value in self.durations_seconds):
            raise SymbolicSequenceError("durations must be positive")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.confidences):
            raise SymbolicSequenceError("confidences must lie in [0, 1]")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SymbolicSequence:
        """Accept the local schema and common names used by contour extractors."""

        pitches = payload.get("pitches_midi", payload.get("pitches"))
        durations = payload.get("durations_seconds", payload.get("durations"))
        if pitches is None or durations is None:
            raise SymbolicSequenceError("sequence must provide pitches and durations")
        pitch_values = tuple(float(value) for value in pitches)
        duration_values = tuple(float(value) for value in durations)
        confidences = payload.get("confidences", payload.get("confidence"))
        if confidences is None:
            confidence_values = (1.0,) * len(pitch_values)
        elif isinstance(confidences, (int, float)):
            confidence_values = (float(confidences),) * len(pitch_values)
        else:
            confidence_values = tuple(float(value) for value in confidences)
        return cls(pitch_values, duration_values, confidence_values)

    @classmethod
    def from_melody_sequence(cls, melody: MelodySequence) -> SymbolicSequence:
        return cls(
            tuple(float(value) for value in melody.pitches),
            tuple(float(value) for value in melody.durations_seconds),
            (1.0,) * len(melody.pitches),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitches_midi": list(self.pitches_midi),
            "durations_seconds": list(self.durations_seconds),
            "confidences": list(self.confidences),
        }


@dataclass(frozen=True)
class SymbolicAugmentConfig:
    """Training-only symbolic corruptions approximating real humming errors."""

    enabled: bool = True
    transpose_semitones: int = 6
    tempo_range: tuple[float, float] = (0.80, 1.25)
    local_tempo_jitter: float = 0.12
    note_delete_probability: float = 0.08
    note_insert_probability: float = 0.06
    octave_error_probability: float = 0.04
    pitch_error_probability: float = 0.08
    merge_probability: float = 0.05

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> SymbolicAugmentConfig:
        values = values or {}
        tempo = tuple(float(value) for value in values.get("tempo_range", cls.tempo_range))
        if len(tempo) != 2:
            raise ValueError("tempo_range must contain two values")
        result = cls(
            enabled=bool(values.get("enabled", cls.enabled)),
            transpose_semitones=int(values.get("transpose_semitones", cls.transpose_semitones)),
            tempo_range=(tempo[0], tempo[1]),
            local_tempo_jitter=float(values.get("local_tempo_jitter", cls.local_tempo_jitter)),
            note_delete_probability=float(
                values.get("note_delete_probability", cls.note_delete_probability)
            ),
            note_insert_probability=float(
                values.get("note_insert_probability", cls.note_insert_probability)
            ),
            octave_error_probability=float(
                values.get("octave_error_probability", cls.octave_error_probability)
            ),
            pitch_error_probability=float(
                values.get("pitch_error_probability", cls.pitch_error_probability)
            ),
            merge_probability=float(values.get("merge_probability", cls.merge_probability)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.transpose_semitones < 0:
            raise ValueError("transpose_semitones cannot be negative")
        if not 0 < self.tempo_range[0] <= self.tempo_range[1]:
            raise ValueError("tempo_range must be positive and ordered")
        probabilities = (
            self.note_delete_probability,
            self.note_insert_probability,
            self.octave_error_probability,
            self.pitch_error_probability,
            self.merge_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("augmentation probabilities must lie in [0, 1]")
        if self.local_tempo_jitter < 0:
            raise ValueError("local_tempo_jitter cannot be negative")


def _nearest_octave_interval(values: NDArray[np.float32]) -> NDArray[np.float32]:
    wrapped = np.mod(values + 6.0, 12.0) - 6.0
    wrapped[np.isclose(wrapped, -6.0) & (values > 0)] = 6.0
    return wrapped.astype(np.float32, copy=False)


def sequence_features(sequence: SymbolicSequence) -> NDArray[np.float32]:
    """Convert notes to key- and global-tempo-invariant model features."""

    pitches = np.asarray(sequence.pitches_midi, dtype=np.float32)
    durations = np.asarray(sequence.durations_seconds, dtype=np.float32)
    confidence = np.asarray(sequence.confidences, dtype=np.float32)
    intervals = np.concatenate(
        (np.zeros(1, dtype=np.float32), _nearest_octave_interval(np.diff(pitches)))
    )
    relative_duration = np.log(np.maximum(durations, 1e-4) / np.median(durations))
    relative_duration = np.clip(relative_duration, -2.5, 2.5).astype(np.float32)
    direction = np.sign(intervals).astype(np.float32)
    return np.column_stack((intervals, relative_duration, confidence, direction)).astype(
        np.float32, copy=False
    )


def augment_sequence(
    sequence: SymbolicSequence,
    rng: random.Random,
    config: SymbolicAugmentConfig,
) -> SymbolicSequence:
    """Apply deterministic (given ``rng``) note-domain humming corruptions."""

    if not config.enabled:
        return sequence
    pitches = list(sequence.pitches_midi)
    durations = list(sequence.durations_seconds)
    confidences = list(sequence.confidences)

    # Global transposition is intentionally included even though interval input
    # should make it a no-op.  It is a useful invariant regression test.
    if config.transpose_semitones:
        shift = rng.randint(-config.transpose_semitones, config.transpose_semitones)
        pitches = [pitch + shift for pitch in pitches]
    tempo = rng.uniform(*config.tempo_range)
    durations = [
        max(0.02, duration * tempo * math.exp(rng.gauss(0.0, config.local_tempo_jitter)))
        for duration in durations
    ]

    for index in range(len(pitches)):
        if rng.random() < config.octave_error_probability:
            pitches[index] += rng.choice((-12.0, 12.0))
        if rng.random() < config.pitch_error_probability:
            pitches[index] += rng.choice((-2.0, -1.0, 1.0, 2.0))

    kept = [
        index for index in range(len(pitches)) if rng.random() >= config.note_delete_probability
    ]
    if len(kept) >= 3:
        pitches = [pitches[index] for index in kept]
        durations = [durations[index] for index in kept]
        confidences = [confidences[index] for index in kept]

    index = 0
    while index < len(pitches) - 1:
        if len(pitches) > 3 and rng.random() < config.merge_probability:
            durations[index] += durations[index + 1]
            confidences[index] = min(confidences[index], confidences[index + 1])
            del pitches[index + 1]
            del durations[index + 1]
            del confidences[index + 1]
            continue
        if rng.random() < config.note_insert_probability:
            pitches.insert(index + 1, (pitches[index] + pitches[index + 1]) / 2.0)
            durations.insert(index + 1, max(0.02, durations[index] * 0.5))
            confidences.insert(index + 1, min(confidences[index], 0.6))
            index += 1
        index += 1
    return SymbolicSequence(tuple(pitches), tuple(durations), tuple(confidences))


def _stable_seed(seed: int, epoch: int, index: int, side: str) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{index}:{side}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class CHADSymbolicPairDataset(Dataset[dict[str, Any]]):
    """JSONL-backed symbolic pairs with deterministic training augmentation."""

    def __init__(
        self,
        path: str | Path,
        *,
        training: bool = False,
        seed: int = 42,
        augmentation: SymbolicAugmentConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.records = read_symbolic_pairs(self.path)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.augmentation = (
            augmentation
            if isinstance(augmentation, SymbolicAugmentConfig)
            else SymbolicAugmentConfig.from_mapping(augmentation)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        query = SymbolicSequence.from_mapping(record["query"])
        reference = SymbolicSequence.from_mapping(record["reference"])
        if self.training:
            query = augment_sequence(
                query,
                random.Random(_stable_seed(self.seed, self.epoch, index, "query")),
                self.augmentation,
            )
            # A lighter independent view keeps both towers honest while the
            # shared weights still see the cleaner reference most of the time.
            reference = augment_sequence(
                reference,
                random.Random(_stable_seed(self.seed, self.epoch, index, "reference")),
                self.augmentation,
            )
        return {
            "query": torch.from_numpy(sequence_features(query)),
            "reference": torch.from_numpy(sequence_features(reference)),
            "pair_id": str(record["pair_id"]),
            "song_id": str(record["song_id"]),
            "fragment_id": str(record["fragment_id"]),
            "reference_id": str(record["reference_id"]),
        }


def _pad_features(items: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    if not items:
        raise ValueError("cannot collate an empty symbolic batch")
    maximum = max(int(item.shape[0]) for item in items)
    feature_dim = int(items[0].shape[1])
    features = torch.zeros((len(items), maximum, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(items), maximum), dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item.shape[0])
        features[index, :length] = item
        mask[index, :length] = True
    return features, mask


def collate_symbolic_pairs(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    query, query_mask = _pad_features([item["query"] for item in batch])
    reference, reference_mask = _pad_features([item["reference"] for item in batch])
    return {
        "query": query,
        "query_mask": query_mask,
        "reference": reference,
        "reference_mask": reference_mask,
        "pair_id": [str(item["pair_id"]) for item in batch],
        "song_id": [str(item["song_id"]) for item in batch],
        "fragment_id": [str(item["fragment_id"]) for item in batch],
        "reference_id": [str(item["reference_id"]) for item in batch],
    }


def read_symbolic_pairs(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("schema_version", -1)) != SYMBOLIC_SCHEMA_VERSION:
                raise SymbolicSequenceError(f"unsupported schema at {resolved}:{line_number}")
            SymbolicSequence.from_mapping(record["query"])
            SymbolicSequence.from_mapping(record["reference"])
            records.append(record)
    if not records:
        raise SymbolicSequenceError(f"symbolic pair file is empty: {resolved}")
    return records


def assert_symbolic_splits_song_disjoint(paths: Mapping[str, str | Path]) -> None:
    frames = {
        split: pd.DataFrame(
            {
                "song_id": [record["song_id"] for record in read_symbolic_pairs(path)],
                "pair_id": [record["pair_id"] for record in read_symbolic_pairs(path)],
                "query_id": [record["query_id"] for record in read_symbolic_pairs(path)],
            }
        )
        for split, path in paths.items()
    }
    assert_song_disjoint(frames)


def _relocate_chad_path(original: str, audio_root: Path) -> Path:
    normalized = str(original).replace("/", "\\")
    marker = "\\fragments\\"
    if marker not in normalized.casefold():
        raise SymbolicSequenceError(f"manifest path is not inside CHAD fragments: {original}")
    offset = normalized.casefold().index(marker) + len(marker)
    candidate = (audio_root / Path(normalized[offset:])).resolve()
    root = audio_root.resolve()
    if root != candidate and root not in candidate.parents:
        raise SymbolicSequenceError("relocated audio path escaped the CHAD audio root")
    if not candidate.is_file():
        raise FileNotFoundError(f"CHAD audio file does not exist: {candidate}")
    return candidate


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _default_extractor(path: Path) -> SymbolicSequence:
    return SymbolicSequence.from_melody_sequence(extract_hum_melody(path))


def _extract_cached(
    path: Path,
    cache_dir: Path,
    extractor: Callable[[Path], SymbolicSequence | Mapping[str, Any]],
) -> SymbolicSequence:
    cache_key = hashlib.sha256(str(path).casefold().encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return SymbolicSequence.from_mapping(payload["sequence"])
    extracted = extractor(path)
    sequence = (
        extracted
        if isinstance(extracted, SymbolicSequence)
        else SymbolicSequence.from_mapping(extracted)
    )
    _atomic_json(
        cache_path,
        {
            "schema_version": SYMBOLIC_SCHEMA_VERSION,
            "source_relative": "/".join(path.parts[-3:]),
            "sequence": sequence.to_dict(),
        },
    )
    return sequence


def build_chad_symbolic_sequences(
    manifest_dir: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    *,
    extractor: Callable[[Path], SymbolicSequence | Mapping[str, Any]] | None = None,
    max_pairs_per_split: int | None = None,
) -> dict[str, Path]:
    """Build resumable JSONL symbolic pairs from the frozen CHAD manifests.

    Only files relocated beneath ``audio_root`` may be opened.  This is an
    explicit guard against accidentally using the project's protected local
    recordings for training or checkpoint selection.
    """

    manifests = Path(manifest_dir).expanduser().resolve()
    audio = Path(audio_root).expanduser().resolve()
    if "recordings" in {part.casefold() for part in audio.parts}:
        raise SymbolicSequenceError("protected recordings cannot be used as CHAD training audio")
    output = Path(output_dir).expanduser().resolve()
    split_frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(manifests / f"{split}.csv", dtype=str)
        missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
        if missing:
            raise SymbolicSequenceError(
                f"{split} manifest is missing columns: {', '.join(sorted(missing))}"
            )
        if set(frame["split"]) != {split}:
            raise SymbolicSequenceError(f"{split} manifest contains another split label")
        split_frames[split] = frame
    assert_song_disjoint(split_frames)

    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    chosen_extractor = extractor or _default_extractor
    outputs: dict[str, Path] = {}
    summary: dict[str, Any] = {
        "schema_version": SYMBOLIC_SCHEMA_VERSION,
        "selection_policy": "frozen_CHAD_song_disjoint_manifests",
        "splits": {},
    }
    for split, frame in split_frames.items():
        if max_pairs_per_split is not None:
            frame = frame.iloc[: int(max_pairs_per_split)]
        records: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for row in frame.to_dict(orient="records"):
            try:
                query_path = _relocate_chad_path(row["query_path"], audio)
                reference_path = _relocate_chad_path(row["reference_path"], audio)
                query = _extract_cached(query_path, cache_dir, chosen_extractor)
                reference = _extract_cached(reference_path, cache_dir, chosen_extractor)
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                failures.append({"pair_id": row["pair_id"], "error": str(error)})
                continue
            records.append(
                {
                    "schema_version": SYMBOLIC_SCHEMA_VERSION,
                    "split": split,
                    "pair_id": row["pair_id"],
                    "song_id": row["song_id"],
                    "fragment_id": row["fragment_id"],
                    "query_id": row["query_id"],
                    "reference_id": row["reference_id"],
                    "query": query.to_dict(),
                    "reference": reference.to_dict(),
                }
            )
        if not records:
            raise SymbolicSequenceError(f"no usable symbolic pairs were produced for {split}")
        destination = output / f"{split}.jsonl"
        temporary = destination.with_suffix(f".jsonl.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, destination)
        outputs[split] = destination
        summary["splits"][split] = {
            "manifest_pairs": len(frame),
            "usable_pairs": len(records),
            "failed_pairs": len(failures),
            "songs": len({record["song_id"] for record in records}),
            "failures": failures,
        }
    assert_symbolic_splits_song_disjoint(outputs)
    summary_path = output / "summary.json"
    _atomic_json(summary_path, summary)
    outputs["summary"] = summary_path
    return outputs


__all__ = [
    "FEATURE_NAMES",
    "SYMBOLIC_SCHEMA_VERSION",
    "CHADSymbolicPairDataset",
    "SymbolicAugmentConfig",
    "SymbolicSequence",
    "SymbolicSequenceError",
    "assert_symbolic_splits_song_disjoint",
    "augment_sequence",
    "build_chad_symbolic_sequences",
    "collate_symbolic_pairs",
    "read_symbolic_pairs",
    "sequence_features",
]
