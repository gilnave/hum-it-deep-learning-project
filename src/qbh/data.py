# This file reads CHAD pairs and makes training batches.
"""CHAD metadata preparation, leakage-safe splits, and PyTorch datasets.

CHAD's hierarchy is ``group_id/fragment_id/id.ext``.  In this project a
``group_id`` is treated as the song identity.  Consequently, splitting rows at
random would be a serious experimental error: another fragment of the same song
could appear in both training and test.  This module performs the split once at
song level and writes auditable CSV manifests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from qbh.audio import AudioSettings, LogMelExtractor, load_audio, trim_silence

LOGGER = logging.getLogger(__name__)

# These are the official columns required by the project.  CHAD currently also
# supplies correlation and crowdsource-check columns; they are useful metadata
# but are not needed to define a humming/original pair.
REQUIRED_CHAD_COLUMNS = (
    "group_id",
    "fragment_id",
    "id",
    "audio_type",
    "youtube_id",
    "interval",
    "is_available",
    "duration",
)

MANIFEST_COLUMNS = (
    "pair_id",
    "song_id",
    "fragment_id",
    "query_id",
    "reference_id",
    "query_path",
    "reference_path",
    "query_duration",
    "reference_duration",
    "query_interval",
    "reference_interval",
    "reference_youtube_id",
    "split",
)

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


class ManifestError(ValueError):
    """Raised when CHAD metadata or a generated manifest violates an invariant."""


class WaveformCache:
    """Byte-bounded, thread-safe cache for decoded CPU waveforms.

    Google Drive file decoding dominates repeated laptop training epochs. This
    cache stores only the decoded/resampled mono waveform; random cropping,
    waveform noise/gain, and spectrogram masking still run anew on every access.
    The scientific training distribution is therefore unchanged.

    A single cache is shared by the train/validation/test dataset objects. It is
    intentionally used only with num_workers=0 because Windows worker processes
    would copy rather than share the cache and could exhaust memory.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("waveform cache max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[tuple[str, int, bool], torch.Tensor] = OrderedDict()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = Lock()

    @staticmethod
    def _tensor_bytes(waveform: torch.Tensor) -> int:
        """Return tensor payload bytes without counting small Python metadata."""

        return int(waveform.nelement() * waveform.element_size())

    def load(self, path: str | Path, sample_rate: int, *, mono: bool = True) -> torch.Tensor:
        """Return one decoded waveform, evicting least-recently-used entries."""

        key = (str(Path(path)), int(sample_rate), bool(mono))
        with self._lock:
            cached = self._items.pop(key, None)
            if cached is not None:
                self._items[key] = cached
                self._hits += 1
                return cached
            self._misses += 1

        # Decode outside the lock. Training currently uses one loader process,
        # but this keeps the class well behaved for ordinary threaded callers.
        waveform = load_audio(path, sample_rate, mono=mono)
        item_bytes = self._tensor_bytes(waveform)
        if item_bytes > self.max_bytes:
            return waveform

        with self._lock:
            # A second caller may have populated the same key while this caller
            # decoded. Prefer the resident tensor and discard the duplicate.
            cached = self._items.pop(key, None)
            if cached is not None:
                self._items[key] = cached
                return cached
            while self._items and self._current_bytes + item_bytes > self.max_bytes:
                _, evicted = self._items.popitem(last=False)
                self._current_bytes -= self._tensor_bytes(evicted)
            self._items[key] = waveform
            self._current_bytes += item_bytes
        return waveform

    def stats(self) -> dict[str, int]:
        """Return a small snapshot suitable for logs and diagnostics."""

        with self._lock:
            return {
                "items": len(self._items),
                "bytes": self._current_bytes,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
            }


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read one value from a mapping or namespace without coupling to YAML code."""

    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _nested_config(config: Any, *names: str, default: Any = None) -> Any:
    """Walk nested mapping/namespace values such as ``training.batch_size``."""

    current = config
    for name in names:
        current = _config_value(current, name, None)
        if current is None:
            return default
    return current


def _parse_available(value: Any) -> bool:
    """Interpret CHAD's CSV boolean without Python's misleading ``bool('False')``."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ManifestError(f"Unrecognized is_available value: {value!r}")


def read_chad_metadata(path: str | Path) -> pd.DataFrame:
    """Read and validate the official ``metadata.csv`` representation.

    Identifier columns are deliberately read as strings.  This preserves leading
    zeros in ``group_id`` and keeps generated paths identical across machines.
    """

    metadata_path = Path(path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"CHAD metadata file does not exist: {metadata_path}")

    frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    missing_columns = sorted(set(REQUIRED_CHAD_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ManifestError(
            "CHAD metadata is missing required columns: " + ", ".join(missing_columns)
        )

    for column in ("group_id", "fragment_id", "id", "audio_type"):
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ManifestError(f"CHAD metadata contains an empty {column!r} value")

    frame["audio_type"] = frame["audio_type"].str.lower()
    frame["is_available"] = frame["is_available"].map(_parse_available)
    frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce")
    return frame


def _audio_priority(path: Path) -> int:
    """Prefer lossless files when duplicate encodings of one CHAD ID exist."""

    try:
        return AUDIO_EXTENSIONS.index(path.suffix.lower())
    except ValueError:
        return len(AUDIO_EXTENSIONS)


def _index_audio_files(audio_root: Path) -> dict[tuple[str, str, str], Path]:
    """Index the CHAD hierarchy once instead of probing DriveFS for every CSV row."""

    if not audio_root.is_dir():
        raise NotADirectoryError(f"CHAD audio root does not exist: {audio_root}")

    # Users may point either to ``chad`` or directly to ``chad/fragments``.
    fragments_root = audio_root / "fragments"
    search_root = fragments_root if fragments_root.is_dir() else audio_root

    indexed: dict[tuple[str, str, str], Path] = {}
    # CHAD guarantees group/fragment/audio. Fixed-depth globbing avoids a costly
    # is_file and resolve round trip for every item on streamed Google Drive.
    for candidate in search_root.glob("*/*/*"):
        if candidate.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        key = (candidate.parent.parent.name, candidate.parent.name, candidate.stem)
        previous = indexed.get(key)
        if previous is None or _audio_priority(candidate) < _audio_priority(previous):
            indexed[key] = candidate
    return indexed


def _largest_remainder_counts(number_of_songs: int) -> dict[str, int]:
    """Convert 70/15/15 ratios to integer counts without losing a song."""

    if number_of_songs <= 0:
        raise ManifestError("At least one usable song is required")

    raw_counts = {name: number_of_songs * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: math.floor(value) for name, value in raw_counts.items()}
    remaining = number_of_songs - sum(counts.values())
    fractional_order = sorted(
        SPLIT_RATIOS,
        key=lambda name: (-(raw_counts[name] - counts[name]), list(SPLIT_RATIOS).index(name)),
    )
    for split_name in fractional_order[:remaining]:
        counts[split_name] += 1

    # Tiny synthetic datasets still benefit from all three partitions.  The real
    # CHAD subset has hundreds of songs, so this branch affects tests only.
    if number_of_songs >= 3:
        for split_name in ("validation", "test"):
            if counts[split_name] == 0:
                donor = max(counts, key=counts.get)  # type: ignore[arg-type]
                counts[donor] -= 1
                counts[split_name] += 1
    return counts


def split_song_ids(song_ids: Sequence[str], seed: int) -> dict[str, set[str]]:
    """Return deterministic, exactly disjoint song sets for all three splits."""

    unique_song_ids = sorted({str(song_id) for song_id in song_ids})
    counts = _largest_remainder_counts(len(unique_song_ids))
    random.Random(seed).shuffle(unique_song_ids)

    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    return {
        "train": set(unique_song_ids[:train_end]),
        "validation": set(unique_song_ids[train_end:validation_end]),
        "test": set(unique_song_ids[validation_end:]),
    }


def assert_song_disjoint(split_frames: Mapping[str, pd.DataFrame]) -> None:
    """Raise a clear error if a song, query, or pair crosses split boundaries."""

    expected = set(SPLIT_RATIOS)
    missing = expected - set(split_frames)
    if missing:
        raise ManifestError(f"Missing split frames: {', '.join(sorted(missing))}")

    song_sets = {name: set(frame["song_id"].astype(str)) for name, frame in split_frames.items()}
    for left_index, left_name in enumerate(SPLIT_RATIOS):
        for right_name in list(SPLIT_RATIOS)[left_index + 1 :]:
            overlap = song_sets[left_name] & song_sets[right_name]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ManifestError(
                    f"Song leakage between {left_name} and {right_name}: {examples}"
                )

    combined = pd.concat(split_frames.values(), ignore_index=True)
    if combined["pair_id"].duplicated().any():
        raise ManifestError("A pair_id occurs more than once across generated manifests")
    if combined["query_id"].duplicated().any():
        raise ManifestError("A humming query occurs more than once across generated manifests")


def _stable_pair_id(song_id: str, fragment_id: str, query_id: str, reference_id: str) -> str:
    """Create a compact identifier stable across output directories and machines."""

    payload = f"{song_id}|{fragment_id}|{query_id}|{reference_id}".encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def _fragment_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Return exact song/fragment keys while preserving identifier strings."""

    return set(
        zip(
            frame["group_id"].astype(str),
            frame["fragment_id"].astype(str),
            strict=True,
        )
    )


def deterministic_pair_subset(
    frame: pd.DataFrame,
    maximum_pairs: int | None,
    seed: int,
) -> pd.DataFrame:
    """Select a reproducible smoke subset while maximizing distinct songs."""

    if maximum_pairs is None or maximum_pairs >= len(frame):
        return frame.reset_index(drop=True)
    if maximum_pairs <= 0:
        raise ValueError("maximum_pairs must be positive when provided")

    shuffled = frame.sample(frac=1.0, random_state=seed)
    first_per_song = shuffled.drop_duplicates("song_id", keep="first")
    if len(first_per_song) >= maximum_pairs:
        selected = first_per_song.head(maximum_pairs)
    else:
        remaining = shuffled.drop(index=first_per_song.index)
        selected = pd.concat(
            [first_per_song, remaining.head(maximum_pairs - len(first_per_song))],
            axis=0,
        )
    return selected.sort_index(kind="stable").reset_index(drop=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Avoid leaving a half-written manifest if Drive synchronization is interrupted."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write a human-readable preparation summary atomically."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def prepare_chad_manifests(
    metadata_path: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, Path]:
    """Build canonical humming/original pairs and song-disjoint CSV manifests.

    Only rows whose audio files actually exist are included.  This matters because
    CHAD's original recordings are hosted by third parties and some may become
    unavailable after the metadata release.  Missing material is counted in
    ``summary.json`` rather than causing leakage-prone manual repairs.
    """

    metadata = read_chad_metadata(metadata_path)
    resolved_audio_root = Path(audio_root).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    audio_index = _index_audio_files(resolved_audio_root)

    # The release-time availability flag is historical, not ground truth. The
    # exact local audio ID is authoritative, allowing a recovered original whose
    # old flag is False while still excluding files that are actually absent.
    hums = metadata[metadata["audio_type"].eq("humming")].copy()
    all_originals = metadata[metadata["audio_type"].eq("original")].copy()
    humming_fragment_keys = _fragment_keys(hums)
    originals = all_originals.loc[
        [
            (group_id, fragment_id) in humming_fragment_keys
            for group_id, fragment_id in zip(
                all_originals["group_id"].astype(str),
                all_originals["fragment_id"].astype(str),
                strict=True,
            )
        ]
    ].copy()
    if hums.empty or originals.empty:
        raise ManifestError(
            "Metadata must contain humming rows and originals aligned to their song/fragment keys"
        )

    def indexed_path(row: pd.Series) -> str | None:
        key = (str(row["group_id"]), str(row["fragment_id"]), str(row["id"]))
        path = audio_index.get(key)
        return str(path) if path is not None else None

    hums["audio_path"] = hums.apply(indexed_path, axis=1)
    originals["audio_path"] = originals.apply(indexed_path, axis=1)
    present_hums = hums[hums["audio_path"].notna()].copy()
    present_originals = originals[originals["audio_path"].notna()].copy()

    # CHAD normally contains one original per fragment.  If a future release has
    # several, sorting by ID chooses the same canonical reference every time.
    canonical_originals = (
        present_originals.sort_values(["group_id", "fragment_id", "id"], kind="stable")
        .drop_duplicates(["group_id", "fragment_id"], keep="first")
        .copy()
    )
    merged = present_hums.merge(
        canonical_originals,
        on=["group_id", "fragment_id"],
        how="inner",
        suffixes=("_query", "_reference"),
        validate="many_to_one",
    )
    if merged.empty:
        raise ManifestError(
            "No complete humming/original pairs were found. Download original fragments "
            "before preparing manifests."
        )

    pairs = pd.DataFrame(
        {
            "song_id": merged["group_id"].astype(str),
            "fragment_id": merged["fragment_id"].astype(str),
            "query_id": merged["id_query"].astype(str),
            "reference_id": merged["id_reference"].astype(str),
            "query_path": merged["audio_path_query"].astype(str),
            "reference_path": merged["audio_path_reference"].astype(str),
            "query_duration": merged["duration_query"],
            "reference_duration": merged["duration_reference"],
            "query_interval": merged["interval_query"].astype(str),
            "reference_interval": merged["interval_reference"].astype(str),
            "reference_youtube_id": merged["youtube_id_reference"].astype(str),
        }
    )
    pairs.insert(
        0,
        "pair_id",
        [
            _stable_pair_id(song, fragment, query, reference)
            for song, fragment, query, reference in zip(
                pairs["song_id"],
                pairs["fragment_id"],
                pairs["query_id"],
                pairs["reference_id"],
                strict=True,
            )
        ],
    )
    if pairs["query_id"].duplicated().any():
        raise ManifestError("One humming ID maps to more than one canonical reference")

    song_splits = split_song_ids(pairs["song_id"].tolist(), seed)
    song_to_split = {
        song_id: split_name for split_name, songs in song_splits.items() for song_id in songs
    }
    pairs["split"] = pairs["song_id"].map(song_to_split)
    pairs = pairs.loc[:, MANIFEST_COLUMNS].sort_values(
        ["split", "song_id", "fragment_id", "query_id"], kind="stable"
    )
    split_frames = {
        split_name: pairs[pairs["split"].eq(split_name)].reset_index(drop=True)
        for split_name in SPLIT_RATIOS
    }
    assert_song_disjoint(split_frames)

    output_paths = {"all": resolved_output_dir / "all.csv"}
    _atomic_csv(pairs.reset_index(drop=True), output_paths["all"])
    for split_name, split_frame in split_frames.items():
        output_paths[split_name] = resolved_output_dir / f"{split_name}.csv"
        _atomic_csv(split_frame, output_paths[split_name])

    present_humming_fragment_keys = _fragment_keys(present_hums)
    present_original_fragment_keys = _fragment_keys(present_originals)
    paired_fragment_keys = set(zip(pairs["song_id"], pairs["fragment_id"], strict=False))
    missing_humming_rows = int(hums["audio_path"].isna().sum())
    missing_original_rows = int(originals["audio_path"].isna().sum())
    summary: dict[str, Any] = {
        "metadata_path": str(Path(metadata_path).expanduser().resolve()),
        "audio_root": str(resolved_audio_root),
        "seed": int(seed),
        "audio_files_indexed": len(audio_index),
        "metadata_rows": len(metadata),
        "metadata_humming_rows": len(hums),
        "metadata_original_rows": len(all_originals),
        "metadata_aligned_original_rows": len(originals),
        "present_humming_rows": len(present_hums),
        "present_original_rows": len(present_originals),
        "missing_humming_rows": missing_humming_rows,
        "missing_original_rows": missing_original_rows,
        # Retain the earlier file-oriented names for report compatibility.
        "missing_humming_files": missing_humming_rows,
        "missing_original_files": missing_original_rows,
        "present_stale_flagged_humming_rows": int((~present_hums["is_available"]).sum()),
        "present_stale_flagged_original_rows": int((~present_originals["is_available"]).sum()),
        "metadata_humming_fragments": len(humming_fragment_keys),
        "present_humming_fragments": len(present_humming_fragment_keys),
        "present_humming_fragments_without_reference": len(
            present_humming_fragment_keys - present_original_fragment_keys
        ),
        # This legacy aggregate is intentionally based on humming keys. An
        # unrelated original-only fragment is outside the experiment scope.
        "fragments_without_paired_audio": len(humming_fragment_keys - paired_fragment_keys),
        "usable_pairs": len(pairs),
        "usable_songs": int(pairs["song_id"].nunique()),
        "splits": {
            name: {
                "pairs": len(frame),
                "songs": int(frame["song_id"].nunique()),
            }
            for name, frame in split_frames.items()
        },
    }
    summary_path = resolved_output_dir / "summary.json"
    _atomic_json(summary, summary_path)
    output_paths["summary"] = summary_path

    LOGGER.info(
        "Prepared %d CHAD pairs from %d songs in %s",
        len(pairs),
        pairs["song_id"].nunique(),
        resolved_output_dir,
    )
    return output_paths


class CHADPairDataset(Dataset[dict[str, Any]]):
    """Load a pair manifest and return model-ready query/reference tensors."""

    REQUIRED_COLUMNS: ClassVar[set[str]] = {
        "song_id",
        "query_id",
        "reference_id",
        "query_path",
        "reference_path",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        audio_config: Any,
        training: bool = False,
        maximum_pairs: int | None = None,
        subset_seed: int = 42,
        waveform_cache: WaveformCache | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Pair manifest does not exist: {self.manifest_path}")

        self.frame = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        missing = self.REQUIRED_COLUMNS - set(self.frame.columns)
        if missing:
            raise ManifestError(
                f"Manifest {self.manifest_path} is missing: {', '.join(sorted(missing))}"
            )
        if self.frame.empty:
            raise ManifestError(f"Pair manifest is empty: {self.manifest_path}")
        self.frame = deterministic_pair_subset(
            self.frame,
            maximum_pairs,
            subset_seed,
        )

        self.training = bool(training)
        self.settings = AudioSettings.from_config(audio_config)
        self.frontend = LogMelExtractor(self.settings)
        self.waveform_cache = waveform_cache
        configured_root = _config_value(audio_config, "audio_root", None)
        self.audio_root = Path(configured_root).expanduser().resolve() if configured_root else None

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def song_ids(self) -> list[str]:
        """Return song IDs in dataset-index order for leakage-safe batch sampling."""

        return self.frame["song_id"].astype(str).tolist()

    def _resolve_audio_path(self, stored_path: str) -> Path:
        """Resolve absolute manifests while also supporting portable relative ones."""

        candidate = Path(stored_path).expanduser()
        if candidate.is_absolute():
            return candidate

        search_candidates = [self.manifest_path.parent / candidate]
        if self.audio_root is not None:
            search_candidates.insert(0, self.audio_root / candidate)
        for possible_path in search_candidates:
            if possible_path.is_file():
                return possible_path.resolve()
        # Return the most likely location so load_audio's error is informative.
        return search_candidates[0].resolve()

    def _load_waveform(self, path: Path) -> torch.Tensor:
        """Decode directly or reuse the shared bounded waveform cache."""

        if self.waveform_cache is None:
            return load_audio(path, self.settings.sample_rate, mono=True)
        return self.waveform_cache.load(path, self.settings.sample_rate, mono=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        query_path = self._resolve_audio_path(str(row["query_path"]))
        reference_path = self._resolve_audio_path(str(row["reference_path"]))

        query_waveform = self._load_waveform(query_path)
        query_waveform = trim_silence(
            query_waveform,
            top_db=self.settings.trim_top_db,
            frame_length=self.settings.trim_frame_length,
            hop_length=self.settings.trim_hop_length,
        )
        reference_waveform = self._load_waveform(reference_path)

        # DataLoader assigns every worker a different torch seed.  Deriving two
        # generators from it makes query/reference augmentation independent while
        # remaining reproducible when the experiment seed is fixed.
        seed = (torch.initial_seed() + 2 * int(index)) % (2**63 - 1)
        query_generator = torch.Generator().manual_seed(seed)
        reference_generator = torch.Generator().manual_seed(seed + 1)
        query = self.frontend(
            query_waveform,
            training=self.training,
            generator=query_generator,
        )
        reference = self.frontend(
            reference_waveform,
            training=self.training,
            generator=reference_generator,
        )

        return {
            "query": query,
            "reference": reference,
            "song_id": str(row["song_id"]),
            "query_id": str(row["query_id"]),
            "reference_id": str(row["reference_id"]),
        }


class UniqueSongBatchSampler(Sampler[list[int]]):
    """Yield batches containing at most one pair from each song.

    In-batch contrastive learning treats other references as negatives.  Two
    fragments from the same song would therefore create a false negative.  This
    sampler groups indices by song and draws no more than one index from each
    group in a batch.
    """

    def __init__(
        self,
        dataset_or_song_ids: CHADPairDataset | Sequence[str],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        song_ids = (
            dataset_or_song_ids.song_ids
            if isinstance(dataset_or_song_ids, CHADPairDataset)
            else [str(song_id) for song_id in dataset_or_song_ids]
        )
        if not song_ids:
            raise ValueError("UniqueSongBatchSampler requires at least one example")

        grouped: dict[str, list[int]] = defaultdict(list)
        for index, song_id in enumerate(song_ids):
            grouped[song_id].append(index)
        self.groups = dict(grouped)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic but different ordering for a new epoch."""

        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        random_generator = random.Random(self.seed + self.epoch)
        queues: dict[str, deque[int]] = {}
        for song_id, indices in self.groups.items():
            copied_indices = list(indices)
            if self.shuffle:
                random_generator.shuffle(copied_indices)
            queues[song_id] = deque(copied_indices)

        while True:
            active_songs = [song_id for song_id, queue in queues.items() if queue]
            if not active_songs:
                break
            if self.shuffle:
                random_generator.shuffle(active_songs)

            # Each active song contributes once in this round.  Splitting the
            # round into chunks preserves the unique-song invariant.
            for offset in range(0, len(active_songs), self.batch_size):
                selected_songs = active_songs[offset : offset + self.batch_size]
                batch = [queues[song_id].popleft() for song_id in selected_songs]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        maximum_examples = max(len(indices) for indices in self.groups.values())
        number_of_batches = 0
        for round_index in range(maximum_examples):
            active_count = sum(len(indices) > round_index for indices in self.groups.values())
            if self.drop_last:
                number_of_batches += active_count // self.batch_size
            else:
                number_of_batches += math.ceil(active_count / self.batch_size)
        return number_of_batches


def _find_manifest(manifest_dir: Path, split_name: str) -> Path:
    """Locate a split manifest, accepting ``val.csv`` as a friendly alias."""

    candidates = [manifest_dir / f"{split_name}.csv"]
    if split_name == "validation":
        candidates.append(manifest_dir / "val.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No {split_name} manifest found; tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def build_dataloaders(config: Any) -> dict[str, DataLoader]:
    """Build train/validation/test loaders from the shared experiment config."""

    manifest_value = _nested_config(config, "data", "manifest_dir")
    if manifest_value is None:
        raise ValueError("Configuration must define data.manifest_dir")
    manifest_dir = Path(manifest_value).expanduser().resolve()
    audio_config = _config_value(config, "audio", None)

    batch_size = int(
        _nested_config(
            config,
            "training",
            "batch_size",
            default=_nested_config(config, "data", "batch_size", default=16),
        )
    )
    number_of_workers = int(
        _nested_config(
            config,
            "data",
            "num_workers",
            default=_nested_config(config, "training", "num_workers", default=0),
        )
    )
    seed = int(_config_value(config, "seed", 42))
    configured_limits = _nested_config(config, "data", "max_pairs_per_split", default={})
    if configured_limits is None:
        configured_limits = {}
    if not isinstance(configured_limits, Mapping):
        raise TypeError("data.max_pairs_per_split must be a mapping of split names to limits")
    pin_memory = bool(_nested_config(config, "training", "pin_memory", default=False))
    drop_last = bool(_nested_config(config, "training", "drop_last", default=True))
    cache_config = _nested_config(config, "data", "waveform_cache", default={}) or {}
    cache_enabled = bool(_config_value(cache_config, "enabled", False))
    waveform_cache: WaveformCache | None = None
    if cache_enabled:
        if number_of_workers != 0:
            raise ValueError("data.waveform_cache requires num_workers=0 on Windows")
        maximum_gib = float(_config_value(cache_config, "max_gib", 4.0))
        if maximum_gib <= 0:
            raise ValueError("data.waveform_cache.max_gib must be positive")
        waveform_cache = WaveformCache(round(maximum_gib * 1024**3))
        LOGGER.info("Enabled shared %.2f GiB decoded-waveform cache", maximum_gib)

    datasets = {
        split_name: CHADPairDataset(
            _find_manifest(manifest_dir, split_name),
            audio_config,
            training=split_name == "train",
            maximum_pairs=(
                int(configured_limits[split_name])
                if configured_limits.get(split_name) is not None
                else None
            ),
            subset_seed=seed,
            waveform_cache=waveform_cache,
        )
        for split_name in SPLIT_RATIOS
    }
    train_sampler = UniqueSongBatchSampler(
        datasets["train"],
        batch_size,
        shuffle=True,
        drop_last=drop_last,
        seed=seed,
    )
    common_arguments = {
        "num_workers": number_of_workers,
        "pin_memory": pin_memory,
        "persistent_workers": number_of_workers > 0,
    }
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_sampler=train_sampler,
            generator=torch.Generator().manual_seed(seed),
            **common_arguments,
        )
    }
    for split_name in ("validation", "test"):
        loaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common_arguments,
        )
    return loaders


__all__ = [
    "REQUIRED_CHAD_COLUMNS",
    "CHADPairDataset",
    "ManifestError",
    "UniqueSongBatchSampler",
    "WaveformCache",
    "assert_song_disjoint",
    "build_dataloaders",
    "deterministic_pair_subset",
    "prepare_chad_manifests",
    "read_chad_metadata",
    "split_song_ids",
]
