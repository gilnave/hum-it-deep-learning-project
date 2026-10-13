# This script trains the hum-to-vocal model.
"""Train a CPU-sized neural reranker on CHAD hums and Demucs vocal windows.

Only CHAD train queries update weights.  Checkpoint selection and early stopping
use song-disjoint CHAD validation queries against the complete configured catalog.
The script never opens the CHAD test JSONL or the users' personal recordings.
Content-addressed query shards and per-epoch checkpoints make both preprocessing
and optimization safely resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qbh.symbolic.audio_song_windows import SongWindowChromaConfig
from qbh.symbolic.audio_vocal_windows import (
    VocalStemWindowChromaBackend,
    load_vocal_stem_sources,
)
from qbh.symbolic.closed_catalog import load_closed_catalog_config
from qbh.symbolic.evaluation import canonical_sha256
from qbh.symbolic.sequence_train import seed_everything
from qbh.symbolic.vocal_neural import VocalNeuralModelConfig, VocalNeuralScorer
from qbh.symbolic.vocal_neural_data import (
    VocalNeuralDataConfig,
    build_vocal_neural_dataset,
)
from qbh.utils import file_sha256


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _move(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _ranking_metrics(
    scores: Tensor,
    targets: Tensor,
    *,
    target_song_ids: Sequence[str],
) -> dict[str, Any]:
    """Compute query and per-song retrieval metrics from complete candidate scores."""

    values = scores.detach().cpu()
    target_values = values.gather(1, targets.detach().cpu().unsqueeze(1))
    ranks = 1 + (values > target_values).sum(dim=1)
    ranks_numpy = ranks.numpy().astype(np.int64, copy=False)
    reciprocal = 1.0 / ranks_numpy.astype(np.float64)
    by_song: dict[str, list[int]] = defaultdict(list)
    for song_id, rank in zip(target_song_ids, ranks_numpy, strict=True):
        by_song[song_id].append(int(rank))
    per_song = {
        song_id: {
            "queries": len(song_ranks),
            "top1": float(np.mean(np.asarray(song_ranks) <= 1)),
            "top3": float(np.mean(np.asarray(song_ranks) <= 3)),
            "top10": float(np.mean(np.asarray(song_ranks) <= 10)),
            "mrr": float(np.mean(1.0 / np.asarray(song_ranks, dtype=np.float64))),
            "mean_rank": float(np.mean(song_ranks)),
        }
        for song_id, song_ranks in sorted(by_song.items())
    }
    song_mrrs = [float(metrics["mrr"]) for metrics in per_song.values()]
    return {
        "queries": len(ranks_numpy),
        "top1": float(np.mean(ranks_numpy <= 1)),
        "top3": float(np.mean(ranks_numpy <= 3)),
        "top10": float(np.mean(ranks_numpy <= 10)),
        "mrr": float(np.mean(reciprocal)),
        "failure_rate": 0.0,
        "mean_rank": float(np.mean(ranks_numpy)),
        "median_rank": float(np.median(ranks_numpy)),
        "macro_song_mrr": float(np.mean(song_mrrs)),
        "worst_song_mrr": float(np.min(song_mrrs)),
        "per_song": per_song,
        "ranks": ranks_numpy.tolist(),
    }


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"ranks"}}


@torch.inference_mode()
def _evaluate(
    model: VocalNeuralScorer,
    loader: DataLoader[Any],
    device: torch.device,
    target_song_ids: Sequence[str],
) -> tuple[dict[str, Any], Tensor]:
    model.eval()
    scores: list[Tensor] = []
    targets: list[Tensor] = []
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        scores.append(
            model(
                batch["query"],
                batch["vocal"],
                batch["raw_similarity"],
                batch["key_proximity"],
            ).cpu()
        )
        targets.append(batch["target_index"].cpu())
    joined_scores = torch.cat(scores)
    joined_targets = torch.cat(targets)
    return (
        _ranking_metrics(
            joined_scores,
            joined_targets,
            target_song_ids=target_song_ids,
        ),
        joined_scores,
    )


def _baseline_metrics(dataset: Any) -> dict[str, Any]:
    """Reproduce the manual browser's raw score plus maximum 0.03 key bonus."""

    raw = dataset.raw_similarity.float()
    proximity = dataset.key_proximity.float()
    best_window = raw.argmax(dim=2, keepdim=True)
    best_raw = raw.gather(2, best_window).squeeze(2)
    best_proximity = proximity.gather(2, best_window).squeeze(2)
    scores = best_raw + 0.03 * best_proximity
    return _ranking_metrics(
        scores,
        dataset.target_index,
        target_song_ids=dataset.target_song_ids,
    )


def _load_config(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("vocal neural training config must be a mapping")
    return raw


def _training_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve explicit data-size guards while preserving the original 30-song defaults.

    These checks are deliberately configuration data instead of being inferred from
    whatever happens to be on disk. A partially copied catalog, missing stem, or
    changed CHAD split therefore fails before optimization starts. Older 30-song
    configurations omit this section, so their audited 22/4/4 and 294/118 contract
    remains the backward-compatible default.
    """

    values = raw.get("contract", {})
    if not isinstance(values, Mapping):
        raise TypeError("contract settings must be a mapping")
    split_values = values.get(
        "expected_split_songs", {"train": 22, "validation": 4, "test": 4}
    )
    query_values = values.get("expected_queries", {"train": 294, "validation": 118})
    if not isinstance(split_values, Mapping) or not isinstance(query_values, Mapping):
        raise TypeError("contract split and query counts must be mappings")

    contract = {
        "expected_split_songs": {
            name: int(split_values[name]) for name in ("train", "validation", "test")
        },
        "expected_queries": {
            name: int(query_values[name]) for name in ("train", "validation")
        },
        "expected_windows_per_song": int(values.get("expected_windows_per_song", 24)),
    }
    flat_counts = [
        *contract["expected_split_songs"].values(),
        *contract["expected_queries"].values(),
        contract["expected_windows_per_song"],
    ]
    if any(count < 1 for count in flat_counts):
        raise ValueError("all neural training contract counts must be positive")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "vocal_neural.yaml",
    )
    arguments = parser.parse_args()
    config_path = arguments.config.expanduser().resolve()
    raw = _load_config(config_path)
    contract = _training_contract(raw)
    catalog_path = _resolve(PROJECT_ROOT, str(raw["data"]["catalog_config"]))
    sequence_root = _resolve(PROJECT_ROOT, str(raw["data"]["sequence_directory"]))
    audio_root = _resolve(PROJECT_ROOT, str(raw["data"]["chad_audio_root"]))
    vocal_manifest_path = _resolve(PROJECT_ROOT, str(raw["data"]["vocal_manifest"]))
    backend_config_path = _resolve(PROJECT_ROOT, str(raw["data"]["backend_config"]))
    feature_cache = _resolve(PROJECT_ROOT, str(raw["data"]["feature_cache"]))
    index_cache = _resolve(PROJECT_ROOT, str(raw["data"]["vocal_index_cache"]))
    output = _resolve(PROJECT_ROOT, str(raw["output"]["directory"]))

    catalog = load_closed_catalog_config(catalog_path)
    song_ids = tuple(sorted(str(song["song_id"]) for song in catalog["songs"]))
    split_songs = {
        split: tuple(
            sorted(
                str(song["song_id"])
                for song in catalog["songs"]
                if str(song["split"]) == split
            )
        )
        for split in ("train", "validation", "test")
    }
    actual_split_counts = {name: len(split_songs[name]) for name in split_songs}
    if actual_split_counts != contract["expected_split_songs"]:
        raise ValueError(
            "catalog split counts changed: "
            f"expected {contract['expected_split_songs']}, got {actual_split_counts}"
        )

    backend_raw = json.loads(backend_config_path.read_text(encoding="utf-8"))
    if (
        backend_raw.get("backend") != "vocal_stem_window_chroma"
        or backend_raw.get("selection_split") != "external_development_only"
    ):
        raise ValueError("neural pilot requires the pinned vocals-only window config")
    sources, vocal_manifest = load_vocal_stem_sources(
        vocal_manifest_path, expected_song_ids=song_ids
    )
    backend = VocalStemWindowChromaBackend(
        sources,
        index_cache,
        SongWindowChromaConfig.from_mapping(backend_raw.get("settings")),
        vocal_manifest=vocal_manifest,
        selection_metadata={
            "role": "neural_training_reference_bank",
            "query_labels_used": False,
            "test_split_used": False,
        },
    )
    expected_candidates = len(song_ids)
    expected_windows = expected_candidates * contract["expected_windows_per_song"]
    if (
        len(backend.candidate_song_ids) != expected_candidates
        or len(backend.window_metadata) != expected_windows
    ):
        raise ValueError(
            "vocal reference bank changed: "
            f"expected {expected_candidates} songs/{expected_windows} windows, got "
            f"{len(backend.candidate_song_ids)} songs/{len(backend.window_metadata)} windows"
        )

    data_config = VocalNeuralDataConfig.from_mapping(raw.get("dataset"))
    train_dataset, train_provenance = build_vocal_neural_dataset(
        sequence_path=sequence_root / "train.jsonl",
        audio_root=audio_root,
        allowed_target_song_ids=split_songs["train"],
        backend=backend,
        cache_directory=feature_cache,
        split="train",
        data_config=data_config,
    )
    validation_dataset, validation_provenance = build_vocal_neural_dataset(
        sequence_path=sequence_root / "validation.jsonl",
        audio_root=audio_root,
        allowed_target_song_ids=split_songs["validation"],
        backend=backend,
        cache_directory=feature_cache,
        split="validation",
        data_config=data_config,
    )
    actual_query_counts = {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
    }
    if actual_query_counts != contract["expected_queries"]:
        raise ValueError(
            "selected CHAD query counts changed: "
            f"expected {contract['expected_queries']}, got {actual_query_counts}"
        )

    training = raw.get("training", {})
    if not isinstance(training, Mapping):
        raise TypeError("training settings must be a mapping")
    epochs = int(training.get("epochs", 30))
    batch_size = int(training.get("batch_size", 6))
    learning_rate = float(training.get("learning_rate", 3e-4))
    weight_decay = float(training.get("weight_decay", 1e-4))
    patience = int(training.get("patience", 7))
    min_delta = float(training.get("min_delta", 1e-3))
    label_smoothing = float(training.get("label_smoothing", 0.03))
    seed = int(training.get("seed", 42))
    device = torch.device(str(training.get("device", "cpu")))
    resume = bool(training.get("resume", True))
    song_balanced_sampling = bool(training.get("song_balanced_sampling", False))
    selection_metric = str(training.get("selection_metric", "mrr"))
    if (
        min(epochs, batch_size, patience) < 1
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or min_delta < 0.0
        or not 0.0 <= label_smoothing < 0.2
        or device.type != "cpu"
        or selection_metric not in {"mrr", "macro_song_mrr"}
    ):
        raise ValueError("invalid CPU neural training settings")

    requested_threads = int(training.get("cpu_threads", 8))
    torch.set_num_threads(max(1, min(requested_threads, os.cpu_count() or 1)))
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    if song_balanced_sampling:
        song_counts = {
            song_id: train_dataset.target_song_ids.count(song_id)
            for song_id in set(train_dataset.target_song_ids)
        }
        sample_weights = torch.tensor(
            [1.0 / song_counts[song_id] for song_id in train_dataset.target_song_ids],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model_config = VocalNeuralModelConfig.from_mapping(raw.get("model"))
    model = VocalNeuralScorer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    output.mkdir(parents=True, exist_ok=True)
    experiment_identity = {
        "format_version": 1,
        "config_sha256": file_sha256(config_path),
        "catalog_sha256": file_sha256(catalog_path),
        "backend_config_sha256": file_sha256(backend_config_path),
        "vocal_manifest_sha256": file_sha256(vocal_manifest_path),
        "train_data": train_provenance,
        "validation_data": validation_provenance,
        "model_config": model_config.to_dict(),
        "training": dict(training),
        "contract": contract,
        "test_queries_opened": False,
        "personal_recordings_opened": False,
    }
    experiment_fingerprint = _fingerprint(experiment_identity)
    baseline = _baseline_metrics(validation_dataset)
    _atomic_json(
        {
            "experiment_fingerprint": experiment_fingerprint,
            "identity": experiment_identity,
            "validation_baseline": _compact(baseline),
        },
        output / "resolved_experiment.json",
    )
    print(
        "Validation baseline: "
        f"MRR={baseline['mrr']:.4f} Top-1={baseline['top1']:.4f} "
        f"Top-3={baseline['top3']:.4f} Top-10={baseline['top10']:.4f}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best_selection_score = float("-inf")
    best_validation_mrr = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    last_path = output / "last.pt"
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        if checkpoint.get("experiment_fingerprint") == experiment_fingerprint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            history = list(checkpoint.get("history", []))
            best_selection_score = float(
                checkpoint.get("best_selection_score", best_selection_score)
            )
            best_validation_mrr = float(
                checkpoint.get("best_validation_mrr", best_validation_mrr)
            )
            best_epoch = int(checkpoint.get("best_epoch", 0))
            stale_epochs = int(checkpoint.get("stale_epochs", 0))
            start_epoch = int(checkpoint["epoch"]) + 1
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            generator.set_state(checkpoint["loader_generator_state"].cpu())
            print(f"Resuming vocal neural training at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses: list[float] = []
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["query"],
                batch["vocal"],
                batch["raw_similarity"],
                batch["key_proximity"],
            )
            loss = torch.nn.functional.cross_entropy(
                logits,
                batch["target_index"],
                label_smoothing=label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation, validation_scores = _evaluate(
            model,
            validation_loader,
            device,
            validation_dataset.target_song_ids,
        )
        scheduler.step(validation["mrr"])
        training_loss = float(np.mean(losses))
        selection_score = float(validation[selection_metric])
        improved = selection_score > best_selection_score + min_delta
        if improved:
            best_selection_score = selection_score
            best_validation_mrr = float(validation["mrr"])
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation": _compact(validation),
            }
        )
        payload = {
            "format_version": 1,
            "backend": "neural_hum_to_vocal_late_interaction",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model_config.to_dict(),
            "data_config": data_config.to_dict(),
            "history": history,
            "selection_metric": selection_metric,
            "best_selection_score": best_selection_score,
            "best_validation_mrr": best_validation_mrr,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "selection_split": "public_CHAD_validation_only",
            "test_queries_opened": False,
            "personal_recordings_opened": False,
            "experiment_fingerprint": experiment_fingerprint,
            "experiment_identity_sha256": canonical_sha256(experiment_identity),
            "torch_rng_state": torch.get_rng_state(),
            "loader_generator_state": generator.get_state(),
        }
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, output / "best.pt")
            np.save(output / "best_validation_scores.npy", validation_scores.numpy())
            _atomic_json(_compact(validation), output / "best_validation_metrics.json")
        print(
            f"epoch={epoch} loss={training_loss:.5f} "
            f"validation_mrr={validation['mrr']:.5f} "
            f"macro_mrr={validation['macro_song_mrr']:.5f} "
            f"top1={validation['top1']:.4f} top3={validation['top3']:.4f} "
            f"top10={validation['top10']:.4f} stale={stale_epochs}/{patience}",
            flush=True,
        )
        if stale_epochs >= patience:
            break

    if best_epoch == 0 or not math.isfinite(best_selection_score):
        raise RuntimeError("vocal neural training did not produce a valid checkpoint")
    summary = {
        "backend": "neural_hum_to_vocal_late_interaction",
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_selection_score": best_selection_score,
        "best_validation_mrr": best_validation_mrr,
        "epochs_completed": len(history),
        "stopped_early": len(history) < epochs,
        "selection_split": "public_CHAD_validation_only",
        "validation_baseline": _compact(baseline),
        "best_validation": json.loads(
            (output / "best_validation_metrics.json").read_text(encoding="utf-8")
        ),
        "test_queries_opened": False,
        "personal_recordings_opened": False,
        "checkpoint_sha256": file_sha256(output / "best.pt"),
        "history": history,
    }
    _atomic_json(summary, output / "training_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
