# This file loads the model and ranks all songs for one hum.
"""Strict dual-ranking runtime for the validation-selected model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from qbh.symbolic.audio_song_windows import SongWindowChromaConfig
from qbh.symbolic.audio_vocal_windows import (
    VocalStemWindowChromaBackend,
    load_vocal_stem_sources,
)
from qbh.symbolic.chorus_prior import (
    ChorusEvidence,
    chorus_evidence_for_candidate,
    load_chorus_metadata,
    standardize_candidate_scores,
)
from qbh.symbolic.closed_catalog import load_closed_catalog_config
from qbh.symbolic.vocal_neural import VocalNeuralModelConfig, VocalNeuralScorer
from qbh.symbolic.vocal_neural_data import (
    VocalNeuralDataConfig,
    prepare_vocal_neural_query,
)
from qbh.utils import file_sha256

CHECKPOINT_SHA256 = "d7cddef6a54af2606998c439680326a33a0e4e88bef85a4f5fb242a2dbede124"
TRAINING_SUMMARY_SHA256 = "2d633e3a11311d8b28860a5fe5d0e60c2413f1b5bd6700ab0a92dbff300fde7c"
TRAINING_CONFIG_SHA256 = "536bda12ab7a02a59d3d7fff44cc26426dcdd50c290fa1db88444526c45bb392"
BACKEND_CONFIG_SHA256 = "f356fd36b03f7ebc16dc13afb88b64c5d1c132d06f0ee0f9a69220043906607f"
VOCAL_MANIFEST_SHA256 = "1b1d42e0df36eeaabaac96a86e8ebc0574630060a61c74d2a716ee1fc0d6d5e8"
CATALOG_SHA256 = "8e3bb8916f3b70d4d5c10328a51ca1eb40582a826d37d2b2f7f59059f741cff5"
CHORUS_METADATA_SHA256 = "0b3de1286f70a1ff6bce0f3b968f13016f8d75a4149f082a2817206ceb40e7a8"
CHORUS_REPORT_SHA256 = "d61f4c35ca65c0a61187d4940127004eebcd2967132eddfa8bb37db62f52c733"
CHORUS_PRODUCT_POLICY_SHA256 = "57f394ab8ddee6ec82487f4f5b80bb2b1f77fadfd4af9ec4764ecbedcc146387"
EXPECTED_VALIDATION_QUERIES = 302


class VocalNeuralRuntimeError(RuntimeError):
    """Raised when a pinned model artifact is missing or changed."""


@dataclass(frozen=True)
class VocalComparisonHit:
    """One result in either independent ranking, plus its rank in the other list."""

    rank: int
    other_rank: int
    method: Literal["neural", "alignment"]
    song_id: str
    title: str
    # Decision score is standardized within this query and includes only this
    # method's independently selected chorus prior.  base_score remains the
    # original, unmodified model/alignment value for transparent comparison.
    score: float
    base_score: float
    raw_similarity: float
    window_start_seconds: float
    window_end_seconds: float
    transposition_semitones: int
    circular_key_distance: int
    key_distance_bonus: float
    chorus_evidence: float
    chorus_coverage: float
    chorus_supporting_windows: int
    chorus_bonus: float
    chorus_start_seconds: float | None
    chorus_end_seconds: float | None
    vocal_path: str
    # These retired compatibility fields remain neutral in the general runtime.
    personalization_applied: bool = False
    personal_evidence: float = 0.0
    personal_chroma_distance: float | None = None
    personal_pitch_distance: float | None = None
    # Bounded additive chorus runtimes fill these explanation-only fields.
    chorus_strength_points: float = 0.0
    chorus_coverage_points: float = 0.0
    chorus_repetition_points: float = 0.0
    chorus_matched_occurrences: int = 0
    chorus_total_occurrences: int = 0


@dataclass(frozen=True)
class DualVocalRanking:
    """The two unfused views of exactly the same query and candidate set."""

    neural: tuple[VocalComparisonHit, ...]
    alignment: tuple[VocalComparisonHit, ...]


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocalNeuralRuntimeError(f"could not read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise VocalNeuralRuntimeError(f"{description} root must be an object")
    return payload


def _require_hash(path: Path, expected: str, description: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise VocalNeuralRuntimeError(f"{description} SHA-256 mismatch")


class VocalNeuralRuntime:
    """Produce neural-only and circular-alignment lists without score fusion."""

    def __init__(self, runtime_config: str | Path) -> None:
        self.runtime_config_path = Path(runtime_config).expanduser().resolve()
        raw = _read_object(self.runtime_config_path, "comparison runtime")
        if (
            int(raw.get("format_version", -1)) != 1
            or raw.get("backend") != "neural_hum_to_vocal_late_interaction"
            or raw.get("enabled") is not True
            or raw.get("experimental") is not True
            or raw.get("fusion_enabled") is not False
            or raw.get("chorus_prior_enabled") is not True
        ):
            raise VocalNeuralRuntimeError("unsupported comparison runtime")
        project_root = self.runtime_config_path.parents[1]

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (
                candidate.resolve()
                if candidate.is_absolute()
                else (project_root / candidate).resolve()
            )

        pins = (
            ("checkpoint", CHECKPOINT_SHA256),
            ("training_summary", TRAINING_SUMMARY_SHA256),
            ("training_config", TRAINING_CONFIG_SHA256),
            ("backend_config", BACKEND_CONFIG_SHA256),
            ("vocal_manifest", VOCAL_MANIFEST_SHA256),
            ("catalog", CATALOG_SHA256),
            ("chorus_metadata", CHORUS_METADATA_SHA256),
            ("chorus_validation_report", CHORUS_REPORT_SHA256),
            ("chorus_product_policy", CHORUS_PRODUCT_POLICY_SHA256),
        )
        paths: dict[str, Path] = {}
        for field, expected_hash in pins:
            if str(raw.get(f"{field}_sha256", "")).lower() != expected_hash:
                raise VocalNeuralRuntimeError(f"runtime has an unexpected {field} pin")
            paths[field] = resolve(str(raw[field]))
            _require_hash(paths[field], expected_hash, field.replace("_", " "))

        catalog = load_closed_catalog_config(paths["catalog"])
        self.songs = {
            str(song["song_id"]): dict(song)
            for song in catalog["songs"]
        }
        catalog_size = len(self.songs)
        if catalog_size < 1:
            raise VocalNeuralRuntimeError("catalog is empty")
        self.chorus_metadata = load_chorus_metadata(
            paths["chorus_metadata"], expected_song_ids=tuple(self.songs)
        )
        chorus_report = _read_object(
            paths["chorus_validation_report"], "chorus validation report"
        )
        selection = chorus_report.get("selection", {})
        product_policy = _read_object(
            paths["chorus_product_policy"], "chorus product policy"
        )
        self.neural_chorus_reward = float(raw.get("neural_chorus_reward", -1.0))
        self.alignment_chorus_reward = float(raw.get("alignment_chorus_reward", -1.0))
        self.chorus_padding_seconds = float(raw.get("chorus_padding_seconds", -1.0))
        if (
            chorus_report.get("selection_split") != "public_CHAD_validation_only"
            or chorus_report.get("test_queries_opened") is not False
            or chorus_report.get("personal_recordings_opened") is not False
        ):
            raise VocalNeuralRuntimeError("chorus prior lacks valid validation provenance")
        if (
            product_policy.get("name") != "chorus_restricted_melody_priority_v2"
            or product_policy.get("selection_basis")
            != "explicit_user_product_instruction"
            or product_policy.get("selected_by_validation_metrics") is not False
            or product_policy.get("fixed_before_requested_personal_sample_recognition")
            is not True
            or self.neural_chorus_reward
            != float(product_policy.get("neural_chorus_reward", -1.0))
            or self.alignment_chorus_reward
            != float(product_policy.get("alignment_chorus_reward", -1.0))
            or self.chorus_padding_seconds
            != float(product_policy.get("chorus_padding_seconds", -1.0))
        ):
            raise VocalNeuralRuntimeError("chorus product policy is invalid")

        summary = _read_object(paths["training_summary"], "training summary")
        best_validation = summary.get("best_validation", {})
        baseline = summary.get("validation_baseline", {})
        if (
            summary.get("backend") != "neural_hum_to_vocal_late_interaction"
            or summary.get("selection_split") != "public_CHAD_validation_only"
            or summary.get("test_queries_opened") is not False
            or summary.get("personal_recordings_opened") is not False
            or summary.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or int(best_validation.get("queries", -1)) != EXPECTED_VALIDATION_QUERIES
            or int(baseline.get("queries", -1)) != EXPECTED_VALIDATION_QUERIES
            or float(best_validation.get("mrr", 0.0)) <= float(baseline.get("mrr", 1.0))
        ):
            raise VocalNeuralRuntimeError("checkpoint lacks valid validation provenance")

        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
        if (
            checkpoint.get("backend") != "neural_hum_to_vocal_late_interaction"
            or checkpoint.get("selection_split") != "public_CHAD_validation_only"
            or checkpoint.get("test_queries_opened") is not False
            or checkpoint.get("personal_recordings_opened") is not False
            or int(checkpoint.get("epoch", -1)) != int(checkpoint.get("best_epoch", -2))
        ):
            raise VocalNeuralRuntimeError("neural checkpoint provenance is invalid")
        self.model = VocalNeuralScorer(
            VocalNeuralModelConfig.from_mapping(checkpoint.get("model_config"))
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.data_config = VocalNeuralDataConfig.from_mapping(checkpoint.get("data_config"))

        backend_raw = _read_object(paths["backend_config"], "vocal backend config")
        sources, manifest_identity = load_vocal_stem_sources(
            paths["vocal_manifest"], expected_song_ids=tuple(self.songs)
        )
        self.backend = VocalStemWindowChromaBackend(
            sources,
            resolve(str(raw["vocal_index_cache"])),
            SongWindowChromaConfig.from_mapping(backend_raw.get("settings")),
            vocal_manifest=manifest_identity,
            selection_metadata={
                "role": "catalog_dual_ranking",
                "selection_split": "public_CHAD_validation_only",
                "test_split_used": False,
                "personal_recordings_used_for_training": False,
                "fusion_enabled": False,
            },
        )
        self.chorus_window_indices_by_song: dict[str, tuple[int, ...]] = {}
        for song_id in self.backend.candidate_song_ids:
            global_indices = [
                index
                for index, metadata in enumerate(self.backend.window_metadata)
                if metadata["song_id"] == song_id
            ]
            windows_for_song = [
                self.backend.window_metadata[index] for index in global_indices
            ]
            eligible = tuple(
                global_indices[local_index]
                for local_index in range(len(windows_for_song))
                if chorus_evidence_for_candidate(
                    song_metadata=self.chorus_metadata[song_id],
                    reference_windows=windows_for_song,
                    best_local_window=local_index,
                    padding_seconds=self.chorus_padding_seconds,
                ).evidence
                > 0.0
            )
            if not eligible:
                raise VocalNeuralRuntimeError(
                    f"documented chorus has no indexed vocal window: {song_id}"
                )
            self.chorus_window_indices_by_song[song_id] = eligible
        windows_per_song = int(backend_raw.get("settings", {}).get("windows_per_duration", 0))
        expected_windows = catalog_size * windows_per_song
        if (
            windows_per_song < 1
            or len(self.backend.candidate_song_ids) != catalog_size
            or len(self.backend.window_metadata) != expected_windows
            or self.backend.source_cache_status != {"hit": catalog_size, "built": 0}
        ):
            raise VocalNeuralRuntimeError("vocal reference index is incomplete")

        self.query_cache = resolve(str(raw["query_cache"]))
        self.maximum_top_k = int(raw.get("maximum_top_k", catalog_size))
        if not 1 <= self.maximum_top_k <= catalog_size:
            raise VocalNeuralRuntimeError("maximum_top_k is invalid")
        self.vocal_paths = {source.song_id: source.audio_path for source in sources}
        self.neural_base_validation_metrics = dict(best_validation)
        self.alignment_base_validation_metrics = dict(baseline)
        self.neural_validation_metrics = dict(selection["neural"]["metrics"])
        self.alignment_validation_metrics = dict(selection["alignment"]["metrics"])

    def _make_hit(
        self,
        *,
        candidate_index: int,
        rank: int,
        other_rank: int,
        method: Literal["neural", "alignment"],
        score: float,
        base_score: float,
        chorus_bonus: float,
        evidence: ChorusEvidence,
        local_window: int,
        prepared: dict[str, Any],
    ) -> VocalComparisonHit:
        song_id = self.backend.candidate_song_ids[candidate_index]
        shift = int(prepared["shifts"][candidate_index, local_window])
        distance = min(shift % 12, (-shift) % 12)
        return VocalComparisonHit(
            rank=rank,
            other_rank=other_rank,
            method=method,
            song_id=song_id,
            title=str(self.songs[song_id]["title"]),
            score=score,
            base_score=base_score,
            raw_similarity=float(prepared["raw_similarity"][candidate_index, local_window]),
            window_start_seconds=evidence.source_window_start_seconds,
            window_end_seconds=evidence.source_window_end_seconds,
            transposition_semitones=shift,
            circular_key_distance=distance,
            key_distance_bonus=0.03 * (1.0 - distance / 6.0),
            chorus_evidence=evidence.evidence,
            chorus_coverage=evidence.coverage,
            chorus_supporting_windows=evidence.supporting_windows,
            chorus_bonus=chorus_bonus,
            chorus_start_seconds=evidence.chorus_start_seconds,
            chorus_end_seconds=evidence.chorus_end_seconds,
            vocal_path=str(self.vocal_paths[song_id]),
        )

    @torch.inference_mode()
    def search(self, query_path: str | Path, top_k: int = 10) -> DualVocalRanking:
        """Rank one recording twice, with no interpolation between score vectors."""

        path = Path(query_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"humming query does not exist: {path}")
        limit = min(max(1, int(top_k)), self.maximum_top_k, len(self.songs))
        prepared = prepare_vocal_neural_query(
            path,
            backend=self.backend,
            cache_directory=self.query_cache,
            data_config=self.data_config,
            priority_window_indices_by_song=self.chorus_window_indices_by_song,
            priority_windows_per_candidate=2,
        )
        query = torch.from_numpy(prepared["query"]).unsqueeze(0)
        vocal = torch.from_numpy(prepared["vocal"]).unsqueeze(0)
        raw_similarity = torch.from_numpy(prepared["raw_similarity"]).unsqueeze(0)
        key_proximity = torch.from_numpy(prepared["key_proximity"]).unsqueeze(0)
        trained_windows = self.data_config.windows_per_candidate
        neural_scores = self.model(
            query[:, :, :trained_windows],
            vocal[:, :, :trained_windows],
            raw_similarity[:, :, :trained_windows],
            key_proximity[:, :, :trained_windows],
        ).squeeze(0)
        candidates, windows = raw_similarity.shape[1:]
        neural_window_scores = self.model.score_windows(
            query.reshape(candidates * windows, 12, query.shape[-1]),
            vocal.reshape(candidates * windows, 12, vocal.shape[-1]),
            raw_similarity.reshape(candidates * windows),
            key_proximity.reshape(candidates * windows),
        ).reshape(candidates, windows)

        # Alignment uses the strongest raw local match and its circular-distance
        # bonus. It is never influenced by neural output; standardization below
        # only gives the separately validated chorus reward a stable unit.
        best_alignment_window = raw_similarity.squeeze(0)[:, :trained_windows].argmax(dim=1)
        candidate_axis = torch.arange(candidates)
        alignment_scores = raw_similarity.squeeze(0)[candidate_axis, best_alignment_window]
        alignment_scores += 0.03 * key_proximity.squeeze(0)[
            candidate_axis, best_alignment_window
        ]
        window_evidence: list[list[ChorusEvidence]] = []
        for candidate, song_id in enumerate(self.backend.candidate_song_ids):
            windows_for_song = [
                self.backend.window_metadata[int(index)]
                for index in prepared["window_indices"][candidate]
            ]
            window_evidence.append(
                [
                    chorus_evidence_for_candidate(
                        song_metadata=self.chorus_metadata[song_id],
                        reference_windows=windows_for_song,
                        best_local_window=window,
                        padding_seconds=self.chorus_padding_seconds,
                    )
                    for window in range(len(windows_for_song))
                ]
            )

        def best_chorus_matches(
            window_scores: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, list[ChorusEvidence]]:
            """Select the strongest *melodic* score among chorus windows.

            The old rule rewarded the location of a method's global best
            window. That gave accidental matches a large bonus and missed a
            true chorus whenever a nearby verse scored slightly higher. Here
            the chorus restriction is applied before selecting the melodic
            score, so chorus location alone never creates a reward.
            """

            selected_windows = np.empty(candidates, dtype=np.int64)
            selected_scores = np.empty(candidates, dtype=np.float64)
            selected_evidence: list[ChorusEvidence] = []
            finite_floor = float(np.min(window_scores))
            for candidate in range(candidates):
                eligible = [
                    window
                    for window, evidence in enumerate(window_evidence[candidate])
                    if evidence.evidence > 0.0
                ]
                if eligible:
                    selected = max(
                        eligible, key=lambda window: float(window_scores[candidate, window])
                    )
                    selected_scores[candidate] = float(window_scores[candidate, selected])
                else:
                    selected = int(np.argmax(window_scores[candidate]))
                    selected_scores[candidate] = finite_floor
                selected_windows[candidate] = selected
                selected_evidence.append(window_evidence[candidate][selected])
            return selected_windows, selected_scores, selected_evidence

        neural_chorus_window, neural_chorus_score, neural_evidence = best_chorus_matches(
            neural_window_scores.cpu().numpy().astype(np.float64)
        )
        alignment_window_scores = (
            raw_similarity.squeeze(0) + 0.03 * key_proximity.squeeze(0)
        )
        alignment_chorus_window, alignment_chorus_score, alignment_evidence = (
            best_chorus_matches(alignment_window_scores.cpu().numpy().astype(np.float64))
        )
        neural_base = neural_scores.cpu().numpy().astype(np.float64)
        alignment_base = alignment_scores.cpu().numpy().astype(np.float64)
        neural_chorus_component = self.neural_chorus_reward * standardize_candidate_scores(
            neural_chorus_score
        )
        alignment_chorus_component = (
            self.alignment_chorus_reward
            * standardize_candidate_scores(alignment_chorus_score)
        )
        neural_decision = standardize_candidate_scores(neural_base) + neural_chorus_component
        alignment_decision = (
            standardize_candidate_scores(alignment_base) + alignment_chorus_component
        )
        neural_order = sorted(
            range(candidates),
            key=lambda index: (
                -float(neural_decision[index]),
                self.backend.candidate_song_ids[index],
            ),
        )
        alignment_order = sorted(
            range(candidates),
            key=lambda index: (
                -float(alignment_decision[index]),
                self.backend.candidate_song_ids[index],
            ),
        )
        neural_ranks = {candidate: rank for rank, candidate in enumerate(neural_order, 1)}
        alignment_ranks = {
            candidate: rank for rank, candidate in enumerate(alignment_order, 1)
        }

        neural_hits = tuple(
            self._make_hit(
                candidate_index=candidate,
                rank=rank,
                other_rank=alignment_ranks[candidate],
                method="neural",
                score=float(neural_decision[candidate]),
                base_score=float(neural_base[candidate]),
                chorus_bonus=float(neural_chorus_component[candidate]),
                evidence=neural_evidence[candidate],
                local_window=int(neural_chorus_window[candidate]),
                prepared=prepared,
            )
            for rank, candidate in enumerate(neural_order[:limit], 1)
        )
        alignment_hits = tuple(
            self._make_hit(
                candidate_index=candidate,
                rank=rank,
                other_rank=neural_ranks[candidate],
                method="alignment",
                score=float(alignment_decision[candidate]),
                base_score=float(alignment_base[candidate]),
                chorus_bonus=float(alignment_chorus_component[candidate]),
                evidence=alignment_evidence[candidate],
                local_window=int(alignment_chorus_window[candidate]),
                prepared=prepared,
            )
            for rank, candidate in enumerate(alignment_order[:limit], 1)
        )
        return DualVocalRanking(neural=neural_hits, alignment=alignment_hits)


__all__ = [
    "DualVocalRanking",
    "VocalComparisonHit",
    "VocalNeuralRuntime",
    "VocalNeuralRuntimeError",
]
