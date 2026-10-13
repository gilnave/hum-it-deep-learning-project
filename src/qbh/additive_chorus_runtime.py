# This is the final runtime with the extra chorus points.
"""Catalog runtime using bounded additive chorus points.

This wrapper preserves the frozen general neural model and independent circular
alignment list. It changes only the final chorus decision: fixed non-negative
points replace the legacy four-times standardized chorus term.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from qbh.symbolic.additive_chorus import (
    AdditiveChorusConfig,
    ChorusPointBreakdown,
    award_additive_chorus_points,
)
from qbh.symbolic.chorus_prior import (
    ChorusEvidence,
    chorus_evidence_for_candidate,
    standardize_candidate_scores,
)
from qbh.symbolic.vocal_neural_data import prepare_vocal_neural_query
from qbh.utils import file_sha256
from qbh.vocal_neural_runtime import (
    DualVocalRanking,
    VocalComparisonHit,
    VocalNeuralRuntime,
)


class AdditiveChorusRuntimeError(RuntimeError):
    """Raised when the additive policy or its immutable base pin is invalid."""


@dataclass(frozen=True)
class _MethodComponents:
    base_scores: np.ndarray
    occurrence_scores: tuple[tuple[float, ...], ...]
    occurrence_coverages: tuple[tuple[float, ...], ...]
    occurrence_windows: tuple[tuple[int, ...], ...]
    occurrence_evidence: tuple[tuple[ChorusEvidence, ...], ...]


@dataclass(frozen=True)
class _QueryComponents:
    prepared: dict[str, Any]
    neural: _MethodComponents
    alignment: _MethodComponents


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdditiveChorusRuntimeError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise AdditiveChorusRuntimeError(f"{description} root must be an object")
    return value


class AdditiveChorusRuntime:
    """Rank the catalog with transparent fixed chorus points."""

    def __init__(
        self, runtime_config: str | Path, *, validate_selection: bool = True
    ) -> None:
        self.runtime_config_path = Path(runtime_config).expanduser().resolve()
        raw = _read_object(self.runtime_config_path, "additive chorus runtime")
        if (
            raw.get("format_version") != 1
            or raw.get("backend") != "bounded_additive_chorus_points"
            or raw.get("enabled") is not True
            or raw.get("fusion_enabled") is not False
        ):
            raise AdditiveChorusRuntimeError("unsupported additive chorus runtime")
        project_root = self.runtime_config_path.parents[1]

        def resolve(value: str) -> Path:
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()

        base_path = resolve(str(raw["base_runtime"]))
        policy_path = resolve(str(raw["policy"]))
        if file_sha256(base_path) != str(raw["base_runtime_sha256"]).lower():
            raise AdditiveChorusRuntimeError("base runtime SHA-256 mismatch")
        if file_sha256(policy_path) != str(raw["policy_sha256"]).lower():
            raise AdditiveChorusRuntimeError("additive policy SHA-256 mismatch")
        policy = _read_object(policy_path, "additive chorus policy")
        if (
            policy.get("format_version") != 1
            or policy.get("name") != "bounded_additive_chorus_points_v1"
            or policy.get("selection_split") != "thirty_personal_design_recordings"
            or policy.get("heldout_ten_opened_before_initial_selection") is not False
            or policy.get("final_revision_after_initial_holdout_aggregate") is not True
            or policy.get("fresh_untouched_test_remaining") is not False
            or policy.get("score_multiplication_enabled") is not False
            or policy.get("negative_chorus_points_enabled") is not False
        ):
            raise AdditiveChorusRuntimeError("additive chorus policy provenance is invalid")
        self.point_config = AdditiveChorusConfig.from_mapping(policy.get("points"))
        selection_path = resolve(str(policy["development_selection"]))
        if validate_selection and file_sha256(selection_path) != str(
            policy["development_selection_sha256"]
        ).lower():
            raise AdditiveChorusRuntimeError("development selection SHA-256 mismatch")
        selection = _read_object(selection_path, "additive development selection")
        before = selection.get("baseline", {})
        after = selection.get("selected", {})
        selected_points = after.get("points", {})
        if validate_selection and (
            selection.get("heldout_ten_opened") is not False
            or int(selection.get("design_recordings", -1)) != 30
            or not isinstance(selected_points, dict)
            or any(
                self.point_config.to_dict().get(name) != value
                for name, value in selected_points.items()
            )
            or float(after.get("metrics", {}).get("top1", 0.0))
            <= float(before.get("top1", 1.0))
            or float(after.get("metrics", {}).get("mrr", 0.0))
            <= float(before.get("mrr", 1.0))
        ):
            raise AdditiveChorusRuntimeError("additive development selection is invalid")
        self.priority_windows_per_candidate = int(
            policy.get("priority_windows_per_candidate", -1)
        )
        if not 1 <= self.priority_windows_per_candidate <= 20:
            raise AdditiveChorusRuntimeError("invalid chorus-priority window count")
        self.base = VocalNeuralRuntime(base_path)
        self.songs = self.base.songs
        self.maximum_top_k = self.base.maximum_top_k
        self.vocal_paths = self.base.vocal_paths

        # Group indexed windows by distinct source-song chorus interval. Query
        # preparation keeps one acoustic representative from each group before
        # filling spare slots, so repeated evidence is truly cross-occurrence.
        self.chorus_window_groups_by_song: dict[str, tuple[tuple[int, ...], ...]] = {}
        for song_id in self.base.backend.candidate_song_ids:
            global_indices = [
                index
                for index, metadata in enumerate(self.base.backend.window_metadata)
                if metadata["song_id"] == song_id
            ]
            windows = [self.base.backend.window_metadata[index] for index in global_indices]
            groups: dict[tuple[float, float], list[int]] = {}
            for local, global_index in enumerate(global_indices):
                evidence = chorus_evidence_for_candidate(
                    song_metadata=self.base.chorus_metadata[song_id],
                    reference_windows=windows,
                    best_local_window=local,
                    padding_seconds=self.base.chorus_padding_seconds,
                )
                if evidence.evidence <= 0.0:
                    continue
                assert evidence.chorus_start_seconds is not None
                assert evidence.chorus_end_seconds is not None
                key = (evidence.chorus_start_seconds, evidence.chorus_end_seconds)
                groups.setdefault(key, []).append(global_index)
            if not groups:
                raise AdditiveChorusRuntimeError(
                    f"song has no indexed chorus occurrence: {song_id}"
                )
            self.chorus_window_groups_by_song[song_id] = tuple(
                tuple(groups[key]) for key in sorted(groups)
            )

    @staticmethod
    def _occurrence_components(
        window_scores: np.ndarray,
        window_evidence: list[list[ChorusEvidence]],
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[int, ...], ...],
        tuple[tuple[ChorusEvidence, ...], ...],
    ]:
        all_scores: list[tuple[float, ...]] = []
        all_coverages: list[tuple[float, ...]] = []
        all_windows: list[tuple[int, ...]] = []
        all_evidence: list[tuple[ChorusEvidence, ...]] = []
        for candidate in range(window_scores.shape[0]):
            groups: dict[tuple[float, float], list[int]] = {}
            for local, evidence in enumerate(window_evidence[candidate]):
                if evidence.evidence <= 0.0:
                    continue
                assert evidence.chorus_start_seconds is not None
                assert evidence.chorus_end_seconds is not None
                key = (evidence.chorus_start_seconds, evidence.chorus_end_seconds)
                groups.setdefault(key, []).append(local)
            selected = [
                max(groups[key], key=lambda local: float(window_scores[candidate, local]))
                for key in sorted(groups)
            ]
            if not selected:
                raise AdditiveChorusRuntimeError("prepared candidate has no chorus window")
            all_scores.append(tuple(float(window_scores[candidate, local]) for local in selected))
            all_coverages.append(
                tuple(float(window_evidence[candidate][local].coverage) for local in selected)
            )
            all_windows.append(tuple(selected))
            all_evidence.append(tuple(window_evidence[candidate][local] for local in selected))
        return (
            tuple(all_scores),
            tuple(all_coverages),
            tuple(all_windows),
            tuple(all_evidence),
        )

    @torch.inference_mode()
    def _score_components(self, query_path: str | Path) -> _QueryComponents:
        path = Path(query_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"humming query does not exist: {path}")
        prepared = prepare_vocal_neural_query(
            path,
            backend=self.base.backend,
            cache_directory=self.base.query_cache,
            data_config=self.base.data_config,
            priority_window_indices_by_song=self.base.chorus_window_indices_by_song,
            priority_window_groups_by_song=self.chorus_window_groups_by_song,
            priority_windows_per_candidate=self.priority_windows_per_candidate,
        )
        query = torch.from_numpy(prepared["query"]).unsqueeze(0)
        vocal = torch.from_numpy(prepared["vocal"]).unsqueeze(0)
        raw = torch.from_numpy(prepared["raw_similarity"]).unsqueeze(0)
        proximity = torch.from_numpy(prepared["key_proximity"]).unsqueeze(0)
        trained_windows = self.base.data_config.windows_per_candidate
        neural_base = self.base.model(
            query[:, :, :trained_windows],
            vocal[:, :, :trained_windows],
            raw[:, :, :trained_windows],
            proximity[:, :, :trained_windows],
        ).squeeze(0)
        candidates, windows = raw.shape[1:]
        neural_windows = self.base.model.score_windows(
            query.reshape(candidates * windows, 12, query.shape[-1]),
            vocal.reshape(candidates * windows, 12, vocal.shape[-1]),
            raw.reshape(candidates * windows),
            proximity.reshape(candidates * windows),
        ).reshape(candidates, windows)
        best_alignment_window = raw.squeeze(0)[:, :trained_windows].argmax(dim=1)
        candidate_axis = torch.arange(candidates)
        alignment_base = raw.squeeze(0)[candidate_axis, best_alignment_window]
        alignment_base += 0.03 * proximity.squeeze(0)[
            candidate_axis, best_alignment_window
        ]
        alignment_windows = raw.squeeze(0) + 0.03 * proximity.squeeze(0)

        window_evidence: list[list[ChorusEvidence]] = []
        for candidate, song_id in enumerate(self.base.backend.candidate_song_ids):
            metadata = [
                self.base.backend.window_metadata[int(index)]
                for index in prepared["window_indices"][candidate]
            ]
            window_evidence.append(
                [
                    chorus_evidence_for_candidate(
                        song_metadata=self.base.chorus_metadata[song_id],
                        reference_windows=metadata,
                        best_local_window=local,
                        padding_seconds=self.base.chorus_padding_seconds,
                    )
                    for local in range(len(metadata))
                ]
            )
        neural_occurrence = self._occurrence_components(
            neural_windows.cpu().numpy().astype(np.float64), window_evidence
        )
        alignment_occurrence = self._occurrence_components(
            alignment_windows.cpu().numpy().astype(np.float64), window_evidence
        )
        return _QueryComponents(
            prepared=prepared,
            neural=_MethodComponents(
                neural_base.cpu().numpy().astype(np.float64), *neural_occurrence
            ),
            alignment=_MethodComponents(
                alignment_base.cpu().numpy().astype(np.float64), *alignment_occurrence
            ),
        )

    @staticmethod
    def _rank_method(
        components: _MethodComponents,
        point_config: AdditiveChorusConfig,
        song_ids: tuple[str, ...],
    ) -> tuple[np.ndarray, list[int], np.ndarray, tuple[ChorusPointBreakdown, ...]]:
        points, breakdowns = award_additive_chorus_points(
            components.occurrence_scores,
            components.occurrence_coverages,
            point_config,
        )
        decision = standardize_candidate_scores(components.base_scores) + points
        order = sorted(
            range(len(song_ids)), key=lambda index: (-float(decision[index]), song_ids[index])
        )
        return decision, order, points, breakdowns

    @torch.inference_mode()
    def search(self, query_path: str | Path, top_k: int = 10) -> DualVocalRanking:
        """Return independent neural and circular rankings with fixed points."""

        components = self._score_components(query_path)
        song_ids = tuple(self.base.backend.candidate_song_ids)
        neural = self._rank_method(components.neural, self.point_config, song_ids)
        alignment = self._rank_method(components.alignment, self.point_config, song_ids)
        neural_decision, neural_order, neural_points, neural_rows = neural
        alignment_decision, alignment_order, alignment_points, alignment_rows = alignment
        neural_ranks = {candidate: rank for rank, candidate in enumerate(neural_order, 1)}
        alignment_ranks = {
            candidate: rank for rank, candidate in enumerate(alignment_order, 1)
        }
        limit = min(max(1, int(top_k)), self.maximum_top_k, len(song_ids))

        def hits(
            method: Literal["neural", "alignment"],
            method_components: _MethodComponents,
            decision: np.ndarray,
            order: list[int],
            points: np.ndarray,
            rows: tuple[ChorusPointBreakdown, ...],
            other_ranks: dict[int, int],
        ) -> tuple[VocalComparisonHit, ...]:
            values: list[VocalComparisonHit] = []
            for rank, candidate in enumerate(order[:limit], 1):
                row = rows[candidate]
                local_window = method_components.occurrence_windows[candidate][
                    row.best_occurrence
                ]
                evidence = method_components.occurrence_evidence[candidate][
                    row.best_occurrence
                ]
                hit = self.base._make_hit(
                    candidate_index=candidate,
                    rank=rank,
                    other_rank=other_ranks[candidate],
                    method=method,
                    score=float(decision[candidate]),
                    base_score=float(method_components.base_scores[candidate]),
                    chorus_bonus=float(points[candidate]),
                    evidence=evidence,
                    local_window=local_window,
                    prepared=components.prepared,
                )
                values.append(
                    replace(
                        hit,
                        chorus_strength_points=row.strength_points,
                        chorus_coverage_points=row.coverage_points,
                        chorus_repetition_points=(
                            row.repetition_points + row.all_occurrences_points
                        ),
                        chorus_matched_occurrences=row.matched_occurrences,
                        chorus_total_occurrences=row.total_occurrences,
                    )
                )
            return tuple(values)

        return DualVocalRanking(
            neural=hits(
                "neural",
                components.neural,
                neural_decision,
                neural_order,
                neural_points,
                neural_rows,
                alignment_ranks,
            ),
            alignment=hits(
                "alignment",
                components.alignment,
                alignment_decision,
                alignment_order,
                alignment_points,
                alignment_rows,
                neural_ranks,
            ),
        )


__all__ = [
    "AdditiveChorusRuntimeError",
    "AdditiveChorusRuntime",
]
