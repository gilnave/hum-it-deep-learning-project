# This file reads and checks the fixed list of candidate songs.
"""Build a fixed, multi-split CHAD reference catalog for closed-set evaluation.

CHAD stores a humming query and its aligned original-song fragment in the same
JSONL row.  Normal split evaluation therefore builds candidates only from the
queried split.  A deployed song catalog is different: all song references are
known in advance, while only the humming recordings must remain held out.

This module constructs that fixed reference catalog from a declared song list.
It reads every selected *reference* once, then creates evaluation cases only
from the requested validation or test split.  The separation is deliberate and
recorded in provenance so a 30-song closed-catalog result cannot be confused
with the wider public CHAD promotion gate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qbh.symbolic.alignment import MelodyTrack
from qbh.symbolic.chad_sequences import SymbolicSequence, read_symbolic_pairs
from qbh.symbolic.evaluation import EvaluationCase, EvaluationError, canonical_sha256
from qbh.utils import file_sha256


def _events(sequence: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Convert one validated compact sequence to timestamped note events."""

    melody = SymbolicSequence.from_mapping(sequence)
    onset = 0.0
    events: list[dict[str, Any]] = []
    for pitch, duration, confidence in zip(
        melody.pitches_midi,
        melody.durations_seconds,
        melody.confidences,
        strict=True,
    ):
        events.append(
            {
                "midi_pitch": round(float(pitch)),
                "onset_seconds": onset,
                "duration_seconds": float(duration),
                "confidence": float(confidence),
            }
        )
        onset += float(duration)
    return tuple(events)


def load_closed_catalog_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a declared CHAD-to-local-song mapping."""

    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"could not read closed-catalog config: {resolved}") from error
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise EvaluationError("closed-catalog config must use schema_version 1")
    songs = payload.get("songs")
    if not isinstance(songs, list) or not songs:
        raise EvaluationError("closed-catalog config needs a non-empty songs list")

    normalized: list[dict[str, str]] = []
    for raw in songs:
        if not isinstance(raw, Mapping):
            raise EvaluationError("each closed-catalog song must be an object")
        song = {
            "song_id": str(raw.get("song_id", "")).strip(),
            "title": str(raw.get("title", "")).strip(),
            "split": str(raw.get("split", "")).strip(),
            "audio": str(raw.get("audio", "")).strip(),
        }
        if not song["song_id"] or not song["title"] or not song["audio"]:
            raise EvaluationError("each closed-catalog song needs song_id, title, and audio")
        if song["split"] not in {"train", "validation", "test"}:
            raise EvaluationError("closed-catalog song split must be train/validation/test")
        normalized.append(song)
    song_ids = [song["song_id"] for song in normalized]
    if len(song_ids) != len(set(song_ids)):
        raise EvaluationError("closed-catalog song IDs must be unique")

    catalog_root = Path(str(payload.get("catalog_root", ""))).expanduser()
    if not catalog_root.is_absolute():
        catalog_root = (resolved.parent / catalog_root).resolve()
    else:
        catalog_root = catalog_root.resolve()
    return {
        "schema_version": 1,
        "name": str(payload.get("name", resolved.stem)),
        "path": resolved,
        "catalog_root": catalog_root,
        "songs": normalized,
    }


def load_closed_chad_catalog(
    sequence_directory: str | Path,
    catalog_config: str | Path,
    *,
    query_split: str,
    maximum_queries: int | None = None,
) -> tuple[list[EvaluationCase], list[MelodyTrack], dict[str, Any]]:
    """Load held-out queries and a fixed reference catalog for selected songs.

    References may come from all CHAD splits because they are the database
    content.  Query cases come exclusively from ``query_split``.  No humming
    sequence from another split is returned or used for ranking.
    """

    if query_split not in {"validation", "test"}:
        raise EvaluationError("closed-catalog queries must use validation or test")
    if maximum_queries is not None and maximum_queries < 1:
        raise EvaluationError("maximum_queries must be positive when provided")

    sequence_root = Path(sequence_directory).expanduser().resolve()
    if "recordings" in {part.casefold() for part in sequence_root.parts}:
        raise EvaluationError("protected recordings paths are forbidden in public evaluation")
    config = load_closed_catalog_config(catalog_config)
    songs = {song["song_id"]: song for song in config["songs"]}

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    sequence_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = sequence_root / f"{split}.jsonl"
        records = read_symbolic_pairs(path)
        if {str(record.get("split", "")) for record in records} != {split}:
            raise EvaluationError(f"symbolic JSONL split disagrees with filename: {path}")
        records_by_split[split] = records
        sequence_hashes[split] = file_sha256(path)

    references: dict[str, MelodyTrack] = {}
    reference_hashes: dict[str, str] = {}
    observed_song_splits: dict[str, set[str]] = {song_id: set() for song_id in songs}
    for split, records in records_by_split.items():
        for record in records:
            song_id = str(record["song_id"])
            if song_id not in songs:
                continue
            observed_song_splits[song_id].add(split)
            reference_id = str(record["reference_id"])
            reference = record["reference"]
            digest = canonical_sha256(reference)
            previous = reference_hashes.get(reference_id)
            if previous is not None and previous != digest:
                raise EvaluationError(f"reference sequence changed: {reference_id}")
            reference_hashes[reference_id] = digest
            existing = references.get(reference_id)
            if existing is not None and existing.song_id != song_id:
                raise EvaluationError(f"reference ID crosses songs: {reference_id}")
            references.setdefault(
                reference_id,
                MelodyTrack(
                    song_id,
                    reference_id,
                    songs[song_id]["title"],
                    _events(reference),
                ),
            )

    missing_songs = sorted(
        song_id for song_id, splits in observed_song_splits.items() if not splits
    )
    if missing_songs:
        raise EvaluationError(
            "selected songs absent from symbolic cache: " + ", ".join(missing_songs)
        )
    for song_id, splits in observed_song_splits.items():
        expected = songs[song_id]["split"]
        if splits != {expected}:
            raise EvaluationError(
                f"declared split for {song_id} is {expected}, observed {sorted(splits)}"
            )

    cases: list[EvaluationCase] = []
    for record in sorted(records_by_split[query_split], key=lambda item: str(item["query_id"])):
        song_id = str(record["song_id"])
        if song_id not in songs:
            continue
        query = record["query"]
        cases.append(
            EvaluationCase(
                query_id=str(record["query_id"]),
                target_song_id=song_id,
                split=query_split,
                query={"events": list(_events(query))},
                source_sha256=canonical_sha256(query),
                metadata={
                    "pair_id": str(record["pair_id"]),
                    "fragment_id": str(record["fragment_id"]),
                    "reference_id": str(record["reference_id"]),
                },
            )
        )
    if maximum_queries is not None:
        cases = cases[:maximum_queries]
    tracks = sorted(references.values(), key=lambda track: (track.song_id, track.track_id))
    if not cases or not tracks:
        raise EvaluationError("closed CHAD catalog contained no usable cases or references")

    missing_audio = [
        song["song_id"]
        for song in config["songs"]
        if not (config["catalog_root"] / song["audio"]).is_file()
    ]
    provenance = {
        "source": "public_CHAD_fixed_closed_catalog",
        "catalog_name": config["name"],
        "catalog_config": str(config["path"]),
        "catalog_config_sha256": file_sha256(config["path"]),
        "catalog_root": str(config["catalog_root"]),
        "query_split": query_split,
        "queries_used": len(cases),
        "query_songs": len({case.target_song_id for case in cases}),
        "candidate_songs": len(songs),
        "reference_tracks": len(tracks),
        "split_song_counts": {
            split: sum(song["split"] == split for song in config["songs"])
            for split in ("train", "validation", "test")
        },
        "missing_catalog_audio_song_ids": missing_audio,
        "sequence_sha256": sequence_hashes,
        "maximum_queries": maximum_queries,
        "protocol_note": (
            "Reference fragments are fixed catalog content from all selected songs; "
            "humming queries come only from query_split."
        ),
    }
    return cases, tracks, provenance


__all__ = ["load_closed_catalog_config", "load_closed_chad_catalog"]
