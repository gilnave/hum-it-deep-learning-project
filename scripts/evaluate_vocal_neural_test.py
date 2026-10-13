# This script checks the final model on the fixed CHAD test set.
"""Run the one-time locked CHAD test for the frozen runtime.

The evaluator opens only test queries whose target songs belong to the pinned
fixed catalog. It reports the independent neural and circular-alignment
rankings both before and after the frozen bounded additive chorus policy.  No
weights, thresholds, or policy values are selected in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qbh.additive_chorus_runtime import AdditiveChorusRuntime
from qbh.utils import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_QUERIES = 339
EXPECTED_TARGET_SONGS = 23


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_test_records(path: Path, allowed_song_ids: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid test JSONL line {line_number}") from error
            if record.get("split") == "test" and str(record.get("song_id")) in allowed_song_ids:
                records.append(record)
    records.sort(key=lambda row: (str(row["song_id"]), str(row["query_id"])))
    query_ids = [str(row["query_id"]) for row in records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("selected locked test contains duplicate query IDs")
    target_songs = {str(row["song_id"]) for row in records}
    if len(records) != EXPECTED_QUERIES or len(target_songs) != EXPECTED_TARGET_SONGS:
        raise ValueError(
            "locked test selection changed: "
            f"expected {EXPECTED_QUERIES} queries/{EXPECTED_TARGET_SONGS} songs, got "
            f"{len(records)} queries/{len(target_songs)} songs"
        )
    return records


def _order(scores: np.ndarray, song_ids: Sequence[str]) -> list[int]:
    if scores.shape != (len(song_ids),) or not np.isfinite(scores).all():
        raise ValueError("candidate scores must be one finite value per catalog song")
    return sorted(
        range(len(song_ids)), key=lambda index: (-float(scores[index]), song_ids[index])
    )


def _rank(scores: np.ndarray, target: int, song_ids: Sequence[str]) -> int:
    return _order(scores, song_ids).index(int(target)) + 1


def _metrics(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    ranks = np.asarray([int(row[rank_key]) for row in rows], dtype=np.int64)
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, rank in zip(rows, ranks, strict=True):
        grouped[str(row["song_id"])].append(int(rank))
    per_song = {
        song_id: {
            "queries": len(values),
            "top1": float(np.mean(np.asarray(values) <= 1)),
            "top3": float(np.mean(np.asarray(values) <= 3)),
            "top10": float(np.mean(np.asarray(values) <= 10)),
            "mrr": float(np.mean(1.0 / np.asarray(values, dtype=np.float64))),
            "mean_rank": float(np.mean(values)),
        }
        for song_id, values in sorted(grouped.items())
    }
    return {
        "queries": len(rows),
        "target_songs": len(grouped),
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks.astype(np.float64))),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "macro_song_mrr": float(np.mean([row["mrr"] for row in per_song.values()])),
        "worst_song_mrr": float(np.min([row["mrr"] for row in per_song.values()])),
        "failure_rate": 0.0,
        "per_song": per_song,
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=PROJECT_ROOT / "configs/additive_chorus_runtime.json",
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=PROJECT_ROOT / "artifacts/symbolic/chad_sequences/test.jsonl",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/chad/fragments",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/test_results.json",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_ROOT / "results/test_results.partial.json",
    )
    args = parser.parse_args()

    runtime_path = args.runtime.resolve()
    test_path = args.test_jsonl.resolve()
    audio_root = args.audio_root.resolve()
    output_path = args.output.resolve()
    state_path = args.state.resolve()
    runtime = AdditiveChorusRuntime(runtime_path)
    song_ids = tuple(runtime.base.backend.candidate_song_ids)
    song_index = {song_id: index for index, song_id in enumerate(song_ids)}
    test_song_ids = {
        song_id
        for song_id, song in runtime.songs.items()
        if str(song.get("split")) == "test"
    }
    records = _read_test_records(test_path, test_song_ids)

    identity = {
        "format_version": 1,
        "runtime": str(runtime_path),
        "runtime_sha256": file_sha256(runtime_path),
        "base_runtime": str(runtime.base.runtime_config_path),
        "base_runtime_sha256": file_sha256(runtime.base.runtime_config_path),
        "test_jsonl": str(test_path),
        "test_jsonl_sha256": file_sha256(test_path),
        "query_count": len(records),
        "target_song_count": len(test_song_ids),
        "catalog_song_ids": list(song_ids),
        "protocol": "one-time frozen CHAD test",
    }
    evaluation_fingerprint = _fingerprint(identity)
    rows: list[dict[str, Any]] = []
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("evaluation_fingerprint") != evaluation_fingerprint:
            raise ValueError("partial locked-test state belongs to another evaluation")
        rows = list(state.get("queries", []))
    completed = {str(row["query_id"]) for row in rows}

    for position, record in enumerate(records, 1):
        query_id = str(record["query_id"])
        if query_id in completed:
            continue
        song_id = str(record["song_id"])
        query_path = (
            audio_root
            / song_id
            / str(record["fragment_id"])
            / f"{query_id}.wav"
        ).resolve()
        if not query_path.is_file():
            raise FileNotFoundError(f"locked test audio is missing: {query_path}")

        components = runtime._score_components(query_path)
        neural_final = runtime._rank_method(components.neural, runtime.point_config, song_ids)
        alignment_final = runtime._rank_method(
            components.alignment, runtime.point_config, song_ids
        )
        target = song_index[song_id]
        row = {
            "query_id": query_id,
            "pair_id": str(record["pair_id"]),
            "song_id": song_id,
            "title": str(runtime.songs[song_id]["title"]),
            "query_sha256": file_sha256(query_path),
            "neural_base_rank": _rank(components.neural.base_scores, target, song_ids),
            "alignment_base_rank": _rank(components.alignment.base_scores, target, song_ids),
            "neural_additive_rank": neural_final[1].index(target) + 1,
            "alignment_additive_rank": alignment_final[1].index(target) + 1,
            "neural_chorus_points": float(neural_final[2][target]),
            "alignment_chorus_points": float(alignment_final[2][target]),
        }
        rows.append(row)
        completed.add(query_id)
        _atomic_json(
            {
                "format_version": 1,
                "status": "incomplete",
                "evaluation_fingerprint": evaluation_fingerprint,
                "identity": identity,
                "queries": rows,
            },
            state_path,
        )
        print(
            f"locked test {position}/{len(records)}: {query_path.name} "
            f"neural={row['neural_base_rank']} additive={row['neural_additive_rank']}",
            flush=True,
        )

    if len(rows) != len(records):
        raise RuntimeError(f"locked test incomplete: {len(rows)}/{len(records)}")
    rows.sort(key=lambda row: (str(row["song_id"]), str(row["query_id"])))
    metrics = {
        "neural_base": _metrics(rows, "neural_base_rank"),
        "alignment_base": _metrics(rows, "alignment_base_rank"),
        "neural_additive_chorus": _metrics(rows, "neural_additive_rank"),
        "alignment_additive_chorus": _metrics(rows, "alignment_additive_rank"),
    }
    report = {
        "format_version": 1,
        "status": "complete",
        "purpose": "one-time evaluation of the frozen system on locked CHAD test",
        "selection_split": "none; all model and policy choices frozen before this run",
        "test_queries_opened": True,
        "evaluation_fingerprint": evaluation_fingerprint,
        "identity": identity,
        "metrics": metrics,
        "queries": rows,
    }
    _atomic_json(report, output_path)
    state_path.unlink(missing_ok=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"report={output_path}", flush=True)
    print(f"report_sha256={file_sha256(output_path)}", flush=True)


if __name__ == "__main__":
    main()
