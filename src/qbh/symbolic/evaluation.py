# These tools calculate ranks and test scores in one fixed way.
"""Independent, backend-neutral evaluation for symbolic melody retrieval.

Only pre-extracted symbolic notes enter this module.  Audio decoding belongs to
the extraction stage, which makes it straightforward to prove that protected
local recordings were not consulted during public model selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from qbh.data import assert_song_disjoint
from qbh.symbolic.alignment import (
    AlignmentSettings,
    MelodyTrack,
    coerce_track,
    rank_symbolic_tracks,
)
from qbh.symbolic.chad_sequences import SymbolicSequence, sequence_features
from qbh.symbolic.sequence_model import SymbolicSequenceEncoder
from qbh.utils import file_sha256


class EvaluationError(ValueError):
    """Raised when evaluation provenance or ranking output is invalid."""


@dataclass(frozen=True)
class EvaluationCase:
    """One labeled public query whose payload is already symbolic."""

    query_id: str
    target_song_id: str
    split: str
    query: Any
    source_sha256: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.query_id or not self.target_song_id:
            raise EvaluationError("query and target song IDs cannot be empty")
        if self.split not in {"validation", "test", "synthetic"}:
            raise EvaluationError("split must be validation, test, or synthetic")
        if self.source_sha256 is not None and (
            len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256.lower())
        ):
            raise EvaluationError("source_sha256 must be a hexadecimal SHA-256 digest")


@dataclass(frozen=True)
class RankedCandidate:
    """Common ranking result emitted by classical and learned adapters."""

    song_id: str
    track_id: str
    title: str
    score: float
    confidence: float
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.song_id or not self.track_id or not math.isfinite(self.score):
            raise EvaluationError("ranked candidate identity/score is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise EvaluationError("candidate confidence must lie in [0, 1]")


class RankingBackend(Protocol):
    """Minimal interface kept independent of retrieval implementation."""

    name: str
    candidate_song_ids: tuple[str, ...]

    def rank(self, query: Any) -> Sequence[RankedCandidate]: ...

    def provenance(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PromotionGate:
    """Predeclared public held-out gate required before runtime promotion."""

    required_split: str = "test"
    minimum_queries: int = 100
    minimum_candidate_songs: int = 40
    minimum_top1: float = 0.15
    minimum_top10: float = 0.55
    minimum_mrr: float = 0.25
    maximum_failure_rate: float = 0.02

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> PromotionGate:
        values = values or {}
        return cls(
            required_split=str(values.get("required_split", cls.required_split)),
            minimum_queries=int(values.get("minimum_queries", cls.minimum_queries)),
            minimum_candidate_songs=int(
                values.get("minimum_candidate_songs", cls.minimum_candidate_songs)
            ),
            minimum_top1=float(values.get("minimum_top1", cls.minimum_top1)),
            minimum_top10=float(values.get("minimum_top10", cls.minimum_top10)),
            minimum_mrr=float(values.get("minimum_mrr", cls.minimum_mrr)),
            maximum_failure_rate=float(
                values.get("maximum_failure_rate", cls.maximum_failure_rate)
            ),
        )

    def assess(
        self, metrics: Mapping[str, Any], *, split: str, candidate_songs: int
    ) -> dict[str, Any]:
        checks = {
            "held_out_public_split": split == self.required_split,
            "enough_queries": int(metrics["queries"]) >= self.minimum_queries,
            "enough_candidate_songs": candidate_songs >= self.minimum_candidate_songs,
            "top1": float(metrics["top1"]) >= self.minimum_top1,
            "top10": float(metrics["top10"]) >= self.minimum_top10,
            "mrr": float(metrics["mrr"]) >= self.minimum_mrr,
            "failure_rate": float(metrics["failure_rate"]) <= self.maximum_failure_rate,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": asdict(self),
            "policy": (
                "Promotion requires one frozen run on the public song-disjoint CHAD test split. "
                "Validation and synthetic smoke results are never promotion-eligible."
            ),
        }


def _plain(value: Any) -> Any:
    """Convert dataclasses/tensors/arrays into stable JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence):
        return [_plain(item) for item in value]
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    raise TypeError(f"cannot serialize evaluation value of type {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _events(value: Any) -> list[dict[str, Any]]:
    track = coerce_track(value, minimum_notes=3)
    return [
        {
            "midi_pitch": int(event.midi_pitch),
            "onset_seconds": float(event.onset_seconds),
            "duration_seconds": float(event.duration_seconds),
            "confidence": float(event.confidence),
        }
        for event in track.events
    ]


def _query_hypotheses(query: Any) -> list[Any]:
    if isinstance(query, Mapping) and "hypotheses" in query:
        hypotheses = query["hypotheses"]
        if isinstance(hypotheses, Sequence) and not isinstance(hypotheses, (str, bytes)):
            return list(hypotheses)
    if isinstance(query, Sequence) and not isinstance(query, (str, bytes, Mapping)):
        # A plain list of event dictionaries is one hypothesis, while a list of
        # objects that each expose events is a hypothesis list.
        if query and all(isinstance(item, Mapping) and "midi_pitch" in item for item in query):
            return [query]
        return list(query)
    return [query]


def _mean_event_confidence(value: Any) -> float:
    events = _events(value)
    return float(np.mean([event["confidence"] for event in events]))


class ClassicalAlignmentBackend:
    """Adapter for the explainable n-gram plus local-alignment matcher."""

    name = "classical_symbolic_alignment"

    def __init__(
        self,
        tracks: Sequence[Any],
        settings: AlignmentSettings | None = None,
    ) -> None:
        if not tracks:
            raise EvaluationError("classical backend needs at least one track")
        self.tracks = tuple(coerce_track(track, minimum_notes=3) for track in tracks)
        self.settings = settings or AlignmentSettings(minimum_notes=3)
        self.candidate_song_ids = tuple(sorted({track.song_id for track in self.tracks}))

    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "settings": asdict(self.settings),
            "tracks_sha256": canonical_sha256(self.tracks),
        }

    def rank(self, query: Any) -> Sequence[RankedCandidate]:
        hypotheses = _query_hypotheses(query)
        # Full ranking is necessary for exact MRR/median rank.  The runtime
        # product may use a smaller coarse shortlist after evaluation passes.
        full_settings = replace(
            self.settings,
            coarse_candidates=max(
                self.settings.coarse_candidates, len(self.tracks) * len(hypotheses)
            ),
        )
        matches = rank_symbolic_tracks(
            hypotheses,
            self.tracks,
            settings=full_settings,
            limit=len(self.candidate_song_ids),
        )
        output: list[RankedCandidate] = []
        for match in matches:
            hypothesis = hypotheses[match.query_hypothesis]
            reference = next(
                track
                for track in self.tracks
                if track.song_id == match.song_id and track.track_id == match.track_id
            )
            query_confidence = _mean_event_confidence(hypothesis)
            reference_confidence = _mean_event_confidence(reference.events)
            confidence = math.sqrt(query_confidence * reference_confidence) * max(
                0.0, 1.0 - match.trace.total_distance
            )
            output.append(
                RankedCandidate(
                    song_id=match.song_id,
                    track_id=match.track_id,
                    title=match.title,
                    score=-match.trace.total_distance,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    details={
                        "backend": self.name,
                        "trace": match.trace.to_dict(),
                        "query_notes": _events(hypothesis),
                        "reference_notes": _events(reference.events),
                    },
                )
            )
        return output


def _symbolic_sequence(value: Any) -> SymbolicSequence:
    events = _events(value)
    return SymbolicSequence(
        tuple(float(event["midi_pitch"]) for event in events),
        tuple(float(event["duration_seconds"]) for event in events),
        tuple(float(event["confidence"]) for event in events),
    )


def _encode_one(model: SymbolicSequenceEncoder, value: Any) -> Tensor:
    features = torch.from_numpy(sequence_features(_symbolic_sequence(value))).unsqueeze(0)
    mask = torch.ones(features.shape[:2], dtype=torch.bool)
    return model(features, mask).squeeze(0)


class LearnedSequenceBackend:
    """CPU adapter for the learned symbolic sequence encoder."""

    name = "learned_symbolic_sequence"

    def __init__(self, model: SymbolicSequenceEncoder, tracks: Sequence[Any]) -> None:
        if not tracks:
            raise EvaluationError("learned backend needs at least one track")
        self.model = model.cpu().eval()
        self.tracks = tuple(coerce_track(track, minimum_notes=3) for track in tracks)
        self.candidate_song_ids = tuple(sorted({track.song_id for track in self.tracks}))
        with torch.inference_mode():
            self.reference_embeddings = torch.stack(
                [_encode_one(self.model, track.events) for track in self.tracks]
            )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, tracks: Sequence[Any]
    ) -> LearnedSequenceBackend:
        path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise EvaluationError("learned checkpoint must contain a mapping")
        if int(checkpoint.get("format_version", -1)) != 1:
            raise EvaluationError("unsupported learned checkpoint format")
        if checkpoint.get("selection_split") != "public_CHAD_validation_only":
            raise EvaluationError(
                "learned checkpoint was not selected only on public CHAD validation"
            )
        model = SymbolicSequenceEncoder.from_config(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        backend = cls(model, tracks)
        backend.checkpoint_path = path
        backend.checkpoint_sha256 = file_sha256(path)
        return backend

    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "model_config": self.model.config.to_dict(),
            "checkpoint": str(getattr(self, "checkpoint_path", "in_memory")),
            "checkpoint_sha256": getattr(self, "checkpoint_sha256", None),
            "tracks_sha256": canonical_sha256(self.tracks),
        }

    def rank(self, query: Any) -> Sequence[RankedCandidate]:
        hypotheses = _query_hypotheses(query)
        with torch.inference_mode():
            query_embeddings = torch.stack([_encode_one(self.model, item) for item in hypotheses])
            similarities = query_embeddings @ self.reference_embeddings.T
        best_by_song: dict[str, RankedCandidate] = {}
        for track_index, track in enumerate(self.tracks):
            hypothesis_index = int(torch.argmax(similarities[:, track_index]))
            score = float(similarities[hypothesis_index, track_index])
            hypothesis = hypotheses[hypothesis_index]
            confidence = math.sqrt(
                _mean_event_confidence(hypothesis) * _mean_event_confidence(track.events)
            ) * max(0.0, min(1.0, (score + 1.0) / 2.0))
            candidate = RankedCandidate(
                song_id=track.song_id,
                track_id=track.track_id,
                title=track.title,
                score=score,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                details={
                    "backend": self.name,
                    "cosine_similarity": score,
                    "query_hypothesis": hypothesis_index,
                    "query_notes": _events(hypothesis),
                    "reference_notes": _events(track.events),
                },
            )
            current = best_by_song.get(track.song_id)
            if current is None or (candidate.score, candidate.track_id) > (
                current.score,
                current.track_id,
            ):
                best_by_song[track.song_id] = candidate
        return sorted(best_by_song.values(), key=lambda item: (-item.score, item.song_id))


def _candidate_dict(candidate: RankedCandidate, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "song_id": candidate.song_id,
        "track_id": candidate.track_id,
        "title": candidate.title,
        "score": candidate.score,
        "confidence": candidate.confidence,
        "details": _plain(candidate.details),
    }


def evaluate_cases(
    backend: RankingBackend,
    cases: Sequence[EvaluationCase],
    *,
    gate: PromotionGate | None = None,
    seed: int = 42,
    report_top_matches: int = 10,
    input_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate exact ranks while capturing per-query failures and diagnostics."""

    if not cases:
        raise EvaluationError("evaluation needs at least one case")
    if report_top_matches < 1:
        raise EvaluationError("report_top_matches must be positive")
    query_ids = [case.query_id for case in cases]
    if len(query_ids) != len(set(query_ids)):
        raise EvaluationError("evaluation query IDs must be unique")
    splits = {case.split for case in cases}
    if len(splits) != 1:
        raise EvaluationError("one evaluation run cannot mix splits")
    split = next(iter(splits))
    candidate_songs = tuple(backend.candidate_song_ids)
    if len(candidate_songs) != len(set(candidate_songs)) or not candidate_songs:
        raise EvaluationError("backend candidate song IDs must be unique and non-empty")
    missing_targets = sorted({case.target_song_id for case in cases} - set(candidate_songs))
    if missing_targets:
        raise EvaluationError(
            "targets absent from candidate catalog: " + ", ".join(missing_targets[:5])
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    ranks: list[int] = []
    failures: list[dict[str, str]] = []
    for case in cases:
        query_started = time.perf_counter()
        try:
            raw_candidates = list(backend.rank(case.query))
            best_by_song: dict[str, RankedCandidate] = {}
            for candidate in raw_candidates:
                if candidate.song_id not in candidate_songs:
                    raise EvaluationError(f"backend returned unknown song: {candidate.song_id}")
                current = best_by_song.get(candidate.song_id)
                if current is None or (candidate.score, candidate.track_id) > (
                    current.score,
                    current.track_id,
                ):
                    best_by_song[candidate.song_id] = candidate
            ordered = sorted(best_by_song.values(), key=lambda item: (-item.score, item.song_id))
            ranked_ids = [candidate.song_id for candidate in ordered]
            if case.target_song_id not in ranked_ids:
                raise EvaluationError("backend did not return a complete target-containing ranking")
            rank = ranked_ids.index(case.target_song_id) + 1
            ranks.append(rank)
            rows.append(
                {
                    "query_id": case.query_id,
                    "target_song_id": case.target_song_id,
                    "rank": rank,
                    "runtime_seconds": time.perf_counter() - query_started,
                    "source_sha256": case.source_sha256,
                    "metadata": _plain(case.metadata or {}),
                    "matches": [
                        _candidate_dict(candidate, index)
                        for index, candidate in enumerate(ordered[:report_top_matches], 1)
                    ],
                }
            )
        except (EvaluationError, RuntimeError, TypeError, ValueError) as error:
            failures.append(
                {
                    "query_id": case.query_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            rows.append(
                {
                    "query_id": case.query_id,
                    "target_song_id": case.target_song_id,
                    "rank": None,
                    "runtime_seconds": time.perf_counter() - query_started,
                    "source_sha256": case.source_sha256,
                    "metadata": _plain(case.metadata or {}),
                    "matches": [],
                }
            )
    denominator = len(cases)
    metrics = {
        "queries": denominator,
        "successful_queries": len(ranks),
        "candidate_songs": len(candidate_songs),
        "top1": sum(rank <= 1 for rank in ranks) / denominator,
        "top5": sum(rank <= 5 for rank in ranks) / denominator,
        "top10": sum(rank <= 10 for rank in ranks) / denominator,
        "mrr": sum(1.0 / rank for rank in ranks) / denominator,
        "median_rank": float(np.median(ranks)) if ranks else None,
        "failure_rate": len(failures) / denominator,
        "runtime_seconds": time.perf_counter() - started,
    }
    resolved_gate = gate or PromotionGate()
    report = {
        "schema_version": 1,
        "evaluation_kind": "independent_symbolic_retrieval",
        "split": split,
        "seed": seed,
        "protected_recordings_used": False,
        "backend": _plain(backend.provenance()),
        "backend_sha256": canonical_sha256(backend.provenance()),
        "cases_sha256": canonical_sha256(
            [
                {
                    "query_id": case.query_id,
                    "target_song_id": case.target_song_id,
                    "split": case.split,
                    "query": case.query,
                    "source_sha256": case.source_sha256,
                }
                for case in cases
            ]
        ),
        "input_provenance": _plain(input_provenance or {}),
        "metrics": metrics,
        "promotion_gate": resolved_gate.assess(
            metrics, split=split, candidate_songs=len(candidate_songs)
        ),
        "failures": failures,
        "queries": rows,
    }
    report["reproducibility_sha256"] = canonical_sha256(
        {
            "split": split,
            "seed": seed,
            "backend_sha256": report["backend_sha256"],
            "cases_sha256": report["cases_sha256"],
            "metrics_without_runtime": {
                key: value for key, value in metrics.items() if key != "runtime_seconds"
            },
            "ranks": [row["rank"] for row in rows],
        }
    )
    return report


def audit_chad_manifests(manifest_dir: str | Path) -> dict[str, Any]:
    """Verify hashes, labels, and song leakage without opening any audio."""

    root = Path(manifest_dir).expanduser().resolve()
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.csv"
        frame = pd.read_csv(path, dtype=str)
        if "split" not in frame or set(frame["split"]) != {split}:
            raise EvaluationError(f"{split} manifest has invalid split labels")
        for column in ("query_path", "reference_path"):
            if column in frame:
                normalized = frame[column].str.replace("/", "\\", regex=False).str.casefold()
                if normalized.str.contains("\\recordings\\", regex=False).any():
                    raise EvaluationError("protected recordings appeared in a CHAD manifest")
        frames[split] = frame
        audit[split] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "pairs": len(frame),
            "songs": int(frame["song_id"].nunique()),
        }
    assert_song_disjoint(frames)
    return {
        "song_disjoint": True,
        "protected_recordings_present": False,
        "splits": audit,
        "audit_sha256": canonical_sha256(audit),
    }


def _cache_melody_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    pitches = payload["pitches"]
    durations = payload["durations_seconds"]
    onset = 0.0
    events: list[dict[str, Any]] = []
    for pitch, duration in zip(pitches, durations, strict=True):
        events.append(
            {
                "midi_pitch": round(float(pitch)),
                "onset_seconds": onset,
                "duration_seconds": float(duration),
                "confidence": 1.0,
            }
        )
        onset += float(duration)
    return events


def load_public_chad_cache(
    cache_path: str | Path,
    manifest_path: str | Path,
    *,
    maximum_queries: int | None = None,
) -> tuple[list[EvaluationCase], list[MelodyTrack], dict[str, Any]]:
    """Adapt the existing public pYIN cache for a quick evaluator smoke run."""

    cache_file = Path(cache_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    if "recordings" in {part.casefold() for part in cache_file.parts + manifest_file.parts}:
        raise EvaluationError("protected recordings paths are forbidden in public evaluation")
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    melodies = cache.get("melodies", {})
    frame = pd.read_csv(manifest_file, dtype=str)
    if len(set(frame["split"])) != 1:
        raise EvaluationError("public smoke manifest must contain exactly one split")
    split = str(frame["split"].iloc[0])
    references: dict[str, MelodyTrack] = {}
    cases: list[EvaluationCase] = []
    for row in frame.sort_values("query_id", kind="stable").to_dict(orient="records"):
        reference_payload = melodies.get(row["reference_path"])
        query_payload = melodies.get(row["query_path"])
        if reference_payload is None or query_payload is None:
            continue
        references.setdefault(
            row["reference_id"],
            MelodyTrack(
                row["song_id"],
                row["reference_id"],
                row["song_id"],
                tuple(_cache_melody_events(reference_payload)),
            ),
        )
        cases.append(
            EvaluationCase(
                query_id=row["query_id"],
                target_song_id=row["song_id"],
                split=split,
                query={"events": _cache_melody_events(query_payload)},
                metadata={"pair_id": row["pair_id"], "reference_id": row["reference_id"]},
            )
        )
    # A valid candidate database must be built before query limiting, otherwise
    # a tiny smoke subset would contain targets absent from its catalog.
    tracks = list(references.values())
    available_song_ids = {track.song_id for track in tracks}
    cases = [case for case in cases if case.target_song_id in available_song_ids]
    if maximum_queries is not None:
        cases = cases[: int(maximum_queries)]
    if not cases or not tracks:
        raise EvaluationError("public CHAD cache contained no usable query/reference pairs")
    provenance = {
        "source": "public_CHAD_preextracted_cache",
        "cache": str(cache_file),
        "cache_sha256": file_sha256(cache_file),
        "manifest": str(manifest_file),
        "manifest_sha256": file_sha256(manifest_file),
        "maximum_queries": maximum_queries,
    }
    return cases, tracks, provenance


__all__ = [
    "ClassicalAlignmentBackend",
    "EvaluationCase",
    "EvaluationError",
    "LearnedSequenceBackend",
    "PromotionGate",
    "RankedCandidate",
    "RankingBackend",
    "audit_chad_manifests",
    "canonical_sha256",
    "evaluate_cases",
    "load_public_chad_cache",
]
