# This script creates the main project results.
"""Evaluate the final project settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qbh.additive_chorus_runtime import AdditiveChorusRuntime
from qbh.symbolic.additive_chorus import award_additive_chorus_points
from qbh.symbolic.audio_song_windows import (
    FullSongWindowChromaBackend,
    SongWindowChromaConfig,
    load_song_audio_sources,
)
from qbh.symbolic.chorus_prior import (
    ChorusEvidence,
    chorus_evidence_for_candidate,
    standardize_candidate_scores,
)
from qbh.symbolic.vocal_neural_data import prepare_vocal_neural_query

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(protocol_path: Path) -> list[dict[str, str]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = Path(protocol["recording_root"]).expanduser().resolve()
    speakers = protocol["speakers"]
    rows: list[dict[str, str]] = []
    for song in protocol["songs"]:
        for speaker in speakers:
            for take in range(1, int(speaker["takes"]) + 1):
                name = (
                    f"{song['filename_stem']} {speaker['speaker_id']} {take}"
                    f"{speaker['extension']}"
                )
                rows.append(
                    {
                        "file": name,
                        "path": str((root / name).resolve()),
                        "song_id": str(song["song_id"]),
                        "title": str(song["title"]),
                        "speaker": str(speaker["speaker_id"]),
                    }
                )
    rows.append(
        {
            "file": "u broke me first.wav",
            "path": str((root / "u broke me first.wav").resolve()),
            "song_id": "17f376ad3004644b",
            "title": "Tate McRae - you broke me first",
            "speaker": "unspecified",
        }
    )
    if not rows:
        raise ValueError("the evaluation set is empty")
    expected = {row["file"] for row in rows}
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"recording folder differs from protocol: missing={missing}, extra={extra}")
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        row["sha256"] = _sha256(path)
    return rows


def _metrics(ranks: list[int]) -> dict[str, Any]:
    values = np.asarray(ranks, dtype=np.int64)
    return {
        "top1": float(np.mean(values <= 1)),
        "top3": float(np.mean(values <= 3)),
        "top10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values.astype(np.float64))),
    }


def _order(scores: np.ndarray, song_ids: tuple[str, ...]) -> list[int]:
    return sorted(
        range(len(song_ids)), key=lambda index: (-float(scores[index]), song_ids[index])
    )


def _rank_of(order: list[int], song_ids: tuple[str, ...], target: str) -> int:
    index = song_ids.index(target)
    return order.index(index) + 1


def _chorus_groups(
    runtime: AdditiveChorusRuntime,
    backend: FullSongWindowChromaBackend,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[tuple[int, ...], ...]]]:
    priority: dict[str, tuple[int, ...]] = {}
    groups_by_song: dict[str, tuple[tuple[int, ...], ...]] = {}
    for song_id in backend.candidate_song_ids:
        global_indices = [
            index
            for index, metadata in enumerate(backend.window_metadata)
            if metadata["song_id"] == song_id
        ]
        metadata = [backend.window_metadata[index] for index in global_indices]
        groups: dict[tuple[float, float], list[int]] = defaultdict(list)
        for local, global_index in enumerate(global_indices):
            evidence = chorus_evidence_for_candidate(
                song_metadata=runtime.base.chorus_metadata[song_id],
                reference_windows=metadata,
                best_local_window=local,
                padding_seconds=runtime.base.chorus_padding_seconds,
            )
            if evidence.evidence <= 0.0:
                continue
            assert evidence.chorus_start_seconds is not None
            assert evidence.chorus_end_seconds is not None
            groups[(evidence.chorus_start_seconds, evidence.chorus_end_seconds)].append(
                global_index
            )
        if not groups:
            raise ValueError(f"full-song backend has no chorus windows for {song_id}")
        groups_by_song[song_id] = tuple(tuple(groups[key]) for key in sorted(groups))
        priority[song_id] = tuple(
            index for group in groups_by_song[song_id] for index in group
        )
    return priority, groups_by_song


@torch.inference_mode()
def _full_song_additive_order(
    runtime: AdditiveChorusRuntime,
    backend: FullSongWindowChromaBackend,
    query_path: Path,
    priority: dict[str, tuple[int, ...]],
    groups_by_song: dict[str, tuple[tuple[int, ...], ...]],
    query_cache: Path,
) -> list[int]:
    prepared = prepare_vocal_neural_query(
        query_path,
        backend=backend,
        cache_directory=query_cache,
        data_config=runtime.base.data_config,
        priority_window_indices_by_song=priority,
        priority_window_groups_by_song=groups_by_song,
        priority_windows_per_candidate=runtime.priority_windows_per_candidate,
    )
    query = torch.from_numpy(prepared["query"]).unsqueeze(0)
    reference = torch.from_numpy(prepared["vocal"]).unsqueeze(0)
    raw = torch.from_numpy(prepared["raw_similarity"]).unsqueeze(0)
    proximity = torch.from_numpy(prepared["key_proximity"]).unsqueeze(0)
    trained_windows = runtime.base.data_config.windows_per_candidate
    base = runtime.base.model(
        query[:, :, :trained_windows],
        reference[:, :, :trained_windows],
        raw[:, :, :trained_windows],
        proximity[:, :, :trained_windows],
    ).squeeze(0)
    candidates, windows = raw.shape[1:]
    window_scores = runtime.base.model.score_windows(
        query.reshape(candidates * windows, 12, query.shape[-1]),
        reference.reshape(candidates * windows, 12, reference.shape[-1]),
        raw.reshape(candidates * windows),
        proximity.reshape(candidates * windows),
    ).reshape(candidates, windows)

    evidence_rows: list[list[ChorusEvidence]] = []
    for candidate, song_id in enumerate(backend.candidate_song_ids):
        metadata = [
            backend.window_metadata[int(index)]
            for index in prepared["window_indices"][candidate]
        ]
        evidence_rows.append(
            [
                chorus_evidence_for_candidate(
                    song_metadata=runtime.base.chorus_metadata[song_id],
                    reference_windows=metadata,
                    best_local_window=local,
                    padding_seconds=runtime.base.chorus_padding_seconds,
                )
                for local in range(len(metadata))
            ]
        )
    occurrence = runtime._occurrence_components(
        window_scores.cpu().numpy().astype(np.float64), evidence_rows
    )
    points, _ = award_additive_chorus_points(
        occurrence[0], occurrence[1], runtime.point_config
    )
    decision = standardize_candidate_scores(
        base.cpu().numpy().astype(np.float64)
    ) + points
    return _order(decision, tuple(backend.candidate_song_ids))


@torch.inference_mode()
def _no_bonus_order(
    runtime: AdditiveChorusRuntime,
    query_path: Path,
) -> list[int]:
    components = runtime._score_components(query_path)
    prepared = components.prepared
    query = torch.from_numpy(prepared["query"]).unsqueeze(0)
    reference = torch.from_numpy(prepared["vocal"]).unsqueeze(0)
    raw = torch.from_numpy(prepared["raw_similarity"]).unsqueeze(0)
    proximity = torch.zeros_like(torch.from_numpy(prepared["key_proximity"])).unsqueeze(0)
    trained_windows = runtime.base.data_config.windows_per_candidate
    scores = runtime.base.model(
        query[:, :, :trained_windows],
        reference[:, :, :trained_windows],
        raw[:, :, :trained_windows],
        proximity[:, :, :trained_windows],
    ).squeeze(0)
    return _order(
        scores.cpu().numpy().astype(np.float64),
        tuple(runtime.base.backend.candidate_song_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=PROJECT_ROOT / "configs/additive_chorus_runtime.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "configs/catalog.json",
    )
    parser.add_argument(
        "--full-song-config",
        type=Path,
        default=PROJECT_ROOT / "configs/external_dev_song_windows_12_max.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/main_results.json",
    )
    args = parser.parse_args()

    records = _load_records(args.protocol.resolve())
    runtime = AdditiveChorusRuntime(args.runtime.resolve())
    raw_full_config = json.loads(args.full_song_config.read_text(encoding="utf-8"))
    full_backend = FullSongWindowChromaBackend(
        load_song_audio_sources(args.catalog.resolve()),
        PROJECT_ROOT / "artifacts/symbolic/main_full_song_reference_cache",
        SongWindowChromaConfig.from_mapping(raw_full_config.get("settings")),
        selection_metadata={
            "role": "main_results",
            "training_or_selection": False,
        },
    )
    if tuple(full_backend.candidate_song_ids) != tuple(
        runtime.base.backend.candidate_song_ids
    ):
        raise ValueError("full-song and vocal candidate orders differ")
    priority, groups_by_song = _chorus_groups(runtime, full_backend)
    song_ids = tuple(runtime.base.backend.candidate_song_ids)
    ranks: dict[str, list[int]] = {
        "current_model": [],
        "full_song_model": [],
        "model_without_bonuses": [],
    }

    for position, record in enumerate(records, 1):
        path = Path(record["path"])
        current_hits = runtime.search(path, top_k=runtime.maximum_top_k).neural
        current_rank = next(hit.rank for hit in current_hits if hit.song_id == record["song_id"])
        full_order = _full_song_additive_order(
            runtime,
            full_backend,
            path,
            priority,
            groups_by_song,
            PROJECT_ROOT / "artifacts/symbolic/main_full_song_query_cache",
        )
        full_rank = _rank_of(full_order, song_ids, record["song_id"])
        no_bonus = _no_bonus_order(runtime, path)
        no_bonus_rank = _rank_of(no_bonus, song_ids, record["song_id"])
        values = {
            "current_model": current_rank,
            "full_song_model": full_rank,
            "model_without_bonuses": no_bonus_rank,
        }
        for name, rank in values.items():
            ranks[name].append(rank)
        print(
            f"result {position}/{len(records)}: {record['file']} "
            f"current={current_rank} full={full_rank} no_bonus={no_bonus_rank}",
            flush=True,
        )

    report = {
        "metrics": {name: _metrics(values) for name, values in ranks.items()},
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, output)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    print(f"report={output}", flush=True)
    print(f"report_sha256={_sha256(output)}", flush=True)


if __name__ == "__main__":
    main()
