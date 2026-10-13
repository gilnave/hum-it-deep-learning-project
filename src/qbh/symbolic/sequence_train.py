# These are shared training tools for repeatable experiments.
"""Reproducible training and public-CHAD validation for symbolic retrieval."""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from qbh.symbolic.chad_sequences import (
    CHADSymbolicPairDataset,
    SymbolicAugmentConfig,
    assert_symbolic_splits_song_disjoint,
    collate_symbolic_pairs,
)
from qbh.symbolic.sequence_model import (
    SymbolicSequenceEncoder,
    build_sequence_model,
    multi_positive_info_nce,
)


@dataclass(frozen=True)
class SequenceTrainConfig:
    epochs: int = 35
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    patience: int = 6
    min_delta: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> SequenceTrainConfig:
        values = values or {}
        result = cls(
            epochs=int(values.get("epochs", cls.epochs)),
            batch_size=int(values.get("batch_size", cls.batch_size)),
            learning_rate=float(values.get("learning_rate", cls.learning_rate)),
            weight_decay=float(values.get("weight_decay", cls.weight_decay)),
            temperature=float(values.get("temperature", cls.temperature)),
            patience=int(values.get("patience", cls.patience)),
            min_delta=float(values.get("min_delta", cls.min_delta)),
            gradient_clip_norm=float(values.get("gradient_clip_norm", cls.gradient_clip_norm)),
            seed=int(values.get("seed", cls.seed)),
            device=str(values.get("device", cls.device)),
            num_workers=int(values.get("num_workers", cls.num_workers)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 1:
            raise ValueError("epochs must be positive and batch_size must exceed one")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.temperature <= 0:
            raise ValueError("optimizer values and temperature are invalid")
        if self.patience < 1 or self.min_delta < 0 or self.gradient_clip_norm < 0:
            raise ValueError("early-stop or gradient-clip values are invalid")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _song_diverse_subset(dataset: CHADSymbolicPairDataset, maximum: int | None) -> None:
    """Take a deterministic round-robin subset instead of the first few songs."""

    if maximum is None or maximum >= len(dataset.records):
        return
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in dataset.records:
        buckets.setdefault(str(record["song_id"]), []).append(record)
    selected: list[dict[str, Any]] = []
    offset = 0
    song_ids = sorted(buckets)
    while len(selected) < maximum:
        added = False
        for song_id in song_ids:
            if offset < len(buckets[song_id]):
                selected.append(buckets[song_id][offset])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        offset += 1
    dataset.records = selected


def make_symbolic_loaders(
    sequence_dir: str | Path,
    train_config: SequenceTrainConfig,
    augmentation: SymbolicAugmentConfig | Mapping[str, Any] | None = None,
    max_pairs: Mapping[str, Any] | None = None,
) -> tuple[CHADSymbolicPairDataset, DataLoader[Any], DataLoader[Any]]:
    root = Path(sequence_dir).expanduser().resolve()
    paths = {split: root / f"{split}.jsonl" for split in ("train", "validation", "test")}
    assert_symbolic_splits_song_disjoint(paths)
    train_dataset = CHADSymbolicPairDataset(
        paths["train"], training=True, seed=train_config.seed, augmentation=augmentation
    )
    validation_dataset = CHADSymbolicPairDataset(paths["validation"], training=False)
    limits = max_pairs or {}
    _song_diverse_subset(train_dataset, int(limits["train"]) if "train" in limits else None)
    _song_diverse_subset(
        validation_dataset,
        int(limits["validation"]) if "validation" in limits else None,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=train_config.num_workers,
        collate_fn=collate_symbolic_pairs,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_symbolic_pairs,
        drop_last=False,
    )
    return train_dataset, train_loader, validation_loader


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def evaluate_song_retrieval(
    model: SymbolicSequenceEncoder,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, Any]:
    """Rank unique songs; any fragment from the correct song is a positive."""

    model.eval()
    query_embeddings: list[Tensor] = []
    query_song_ids: list[str] = []
    reference_by_id: dict[str, Tensor] = {}
    reference_song_by_id: dict[str, str] = {}
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        queries = model(batch["query"], batch["query_mask"]).cpu()
        references = model(batch["reference"], batch["reference_mask"]).cpu()
        query_embeddings.extend(queries)
        query_song_ids.extend(str(value) for value in batch["song_id"])
        for index, reference_id in enumerate(batch["reference_id"]):
            reference_by_id.setdefault(str(reference_id), references[index])
            reference_song_by_id.setdefault(str(reference_id), str(batch["song_id"][index]))
    if not query_embeddings or not reference_by_id:
        raise RuntimeError("validation produced no embeddings")
    reference_ids = sorted(reference_by_id)
    references = torch.stack([reference_by_id[value] for value in reference_ids])
    reference_songs = [reference_song_by_id[value] for value in reference_ids]
    candidate_songs = sorted(set(reference_songs))
    song_columns = {
        song_id: torch.tensor(
            [index for index, value in enumerate(reference_songs) if value == song_id],
            dtype=torch.long,
        )
        for song_id in candidate_songs
    }
    ranks: list[int] = []
    for query, target_song in zip(query_embeddings, query_song_ids, strict=True):
        similarities = query @ references.T
        song_scores = {
            song_id: float(similarities[indices].max()) for song_id, indices in song_columns.items()
        }
        ranked = sorted(candidate_songs, key=lambda value: (-song_scores[value], value))
        ranks.append(ranked.index(target_song) + 1)
    reciprocal = [1.0 / rank for rank in ranks]
    return {
        "queries": len(ranks),
        "candidate_songs": len(candidate_songs),
        "top1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "top5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "top10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(reciprocal) / len(reciprocal),
        "median_rank": float(np.median(ranks)),
        "ranks": ranks,
    }


def train_sequence_encoder(
    model: SymbolicSequenceEncoder,
    train_dataset: CHADSymbolicPairDataset,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    train_config: SequenceTrainConfig,
    output_dir: str | Path,
    *,
    experiment_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with checkpoint selection based only on public validation MRR."""

    seed_everything(train_config.seed)
    device = torch.device(train_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    output = Path(output_dir).expanduser().resolve()
    best_mrr = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, train_config.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            query = model(batch["query"], batch["query_mask"])
            reference = model(batch["reference"], batch["reference_mask"])
            # reference_id identifies one CHAD fragment.  Multiple people
            # humming that fragment are positives, never in-batch negatives.
            loss = multi_positive_info_nce(
                query,
                reference,
                batch["reference_id"],
                temperature=train_config.temperature,
            )
            loss.backward()
            if train_config.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_song_retrieval(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation": validation,
        }
        history.append(record)
        improved = validation["mrr"] > best_mrr + train_config.min_delta
        payload = {
            "format_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "train_config": asdict(train_config),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "selection_split": "public_CHAD_validation_only",
            "experiment_config": dict(experiment_config or {}),
        }
        _atomic_torch_save(payload, output / "last.pt")
        if improved:
            best_mrr = float(validation["mrr"])
            best_epoch = epoch
            stale_epochs = 0
            payload["best_validation_mrr"] = best_mrr
            _atomic_torch_save(payload, output / "best.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= train_config.patience:
            break
    if best_epoch == 0 or not math.isfinite(best_mrr):
        raise RuntimeError("training did not produce a finite validation checkpoint")
    result = {
        "best_epoch": best_epoch,
        "best_validation_mrr": best_mrr,
        "epochs_completed": len(history),
        "stopped_early": len(history) < train_config.epochs,
        "selection_split": "public_CHAD_validation_only",
        "history": history,
    }
    _atomic_json(result, output / "training_summary.json")
    return result


def run_sequence_training(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("symbolic training config must be a mapping")
    train_config = SequenceTrainConfig.from_mapping(raw.get("training"))
    project_root = path.parents[2]

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (
            candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
        )

    sequence_dir = resolve(str(raw["data"]["sequence_dir"]))
    output_dir = resolve(str(raw["output"]["directory"]))
    train_dataset, train_loader, validation_loader = make_symbolic_loaders(
        sequence_dir,
        train_config,
        raw.get("augmentation"),
        raw.get("data", {}).get("max_pairs_per_split"),
    )
    model = build_sequence_model(raw.get("model"))
    return train_sequence_encoder(
        model,
        train_dataset,
        train_loader,
        validation_loader,
        train_config,
        output_dir,
        experiment_config=raw,
    )


__all__ = [
    "SequenceTrainConfig",
    "evaluate_song_retrieval",
    "make_symbolic_loaders",
    "run_sequence_training",
    "seed_everything",
    "train_sequence_encoder",
]
