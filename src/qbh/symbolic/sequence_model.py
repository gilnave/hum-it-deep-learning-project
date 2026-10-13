# This is the small encoder used for melody sequence features.
"""Small CPU-friendly Siamese encoder for symbolic melody sequences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SequenceModelConfig:
    """Serializable architecture settings with conservative laptop defaults."""

    input_dim: int = 4
    hidden_dim: int = 96
    num_layers: int = 2
    embedding_dim: int = 96
    dropout: float = 0.15

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> SequenceModelConfig:
        values = values or {}
        result = cls(
            input_dim=int(values.get("input_dim", cls.input_dim)),
            hidden_dim=int(values.get("hidden_dim", cls.hidden_dim)),
            num_layers=int(values.get("num_layers", cls.num_layers)),
            embedding_dim=int(values.get("embedding_dim", cls.embedding_dim)),
            dropout=float(values.get("dropout", cls.dropout)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.num_layers, self.embedding_dim) <= 0:
            raise ValueError("all sequence model dimensions must be positive")
        if self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even for a bidirectional GRU")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
        }


class SymbolicSequenceEncoder(nn.Module):
    """Encode variable-length note features into unit-length embeddings.

    The same encoder is used for hum and song-reference sequences.  A small
    bidirectional GRU models local melodic context; masked attention pooling
    then lets distinctive turns matter more than long repeated notes.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 96,
        num_layers: int = 2,
        embedding_dim: int = 96,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.config = SequenceModelConfig(
            int(input_dim), int(hidden_dim), int(num_layers), int(embedding_dim), float(dropout)
        )
        self.config.validate()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self._reset_parameters()

    @classmethod
    def from_config(
        cls, config: SequenceModelConfig | Mapping[str, Any] | None
    ) -> SymbolicSequenceEncoder:
        parsed = (
            config
            if isinstance(config, SequenceModelConfig)
            else SequenceModelConfig.from_mapping(config)
        )
        return cls(**parsed.to_dict())

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, notes, feature_dim]")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected {self.config.input_dim} features, got {features.shape[-1]}")
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [batch, notes]")
        lengths = mask.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("every sequence must contain at least one unmasked note")

        # Put heterogeneous symbolic fields on comparable numerical scales.
        normalized = features.float().clone()
        normalized[..., 0] /= 6.0
        normalized[..., 1] /= 2.5
        projected = self.input_projection(normalized)
        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=features.shape[1],
        )
        attention_logits = self.attention(encoded).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, torch.finfo(encoded.dtype).min)
        weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(encoded * weights, dim=1)
        return F.normalize(self.projection(pooled), p=2, dim=-1, eps=1e-8)

    def encode(self, features: Tensor, mask: Tensor) -> Tensor:
        return self(features, mask)


def multi_positive_info_nce(
    query_embeddings: Tensor,
    reference_embeddings: Tensor,
    positive_ids: Sequence[str],
    *,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric InfoNCE without false negatives from repeated CHAD fragments."""

    if query_embeddings.shape != reference_embeddings.shape or query_embeddings.ndim != 2:
        raise ValueError("query and reference embeddings must share shape [batch, dim]")
    if len(positive_ids) != query_embeddings.shape[0]:
        raise ValueError("positive_ids length must equal batch size")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = tuple(str(value) for value in positive_ids)
    positive_mask = torch.tensor(
        [[left == right for right in ids] for left in ids],
        dtype=torch.bool,
        device=query_embeddings.device,
    )
    logits = query_embeddings @ reference_embeddings.T / temperature

    def direction_loss(scores: Tensor, positives: Tensor) -> Tensor:
        numerator = torch.logsumexp(scores.masked_fill(~positives, float("-inf")), dim=1)
        denominator = torch.logsumexp(scores, dim=1)
        return (denominator - numerator).mean()

    return 0.5 * (direction_loss(logits, positive_mask) + direction_loss(logits.T, positive_mask.T))


def build_sequence_model(
    config: SequenceModelConfig | Mapping[str, Any] | None = None,
) -> SymbolicSequenceEncoder:
    if isinstance(config, Mapping) and isinstance(config.get("model"), Mapping):
        config = config["model"]
    return SymbolicSequenceEncoder.from_config(config)


__all__ = [
    "SequenceModelConfig",
    "SymbolicSequenceEncoder",
    "build_sequence_model",
    "multi_positive_info_nce",
]
