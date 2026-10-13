# This file has the neural network used for final song scoring.
"""Compact neural late interaction for humming-to-vocal chroma matching.

The model is intentionally small enough for CPU training.  It does not collapse
an entire recording into one vector.  Instead, separate query and vocal encoders
produce contextual frame tokens, a two-dimensional interaction network examines
their complete local similarity matrix, and a multiple-instance aggregation
selects the strongest vocal window for each candidate song.

The learned branch is a residual correction to the validated chroma similarity.
Its final layer starts at zero, so an untrained model exactly preserves the
existing matcher and the circular key-distance preference requested for manual
testing.  This is safer than asking a small dataset to relearn a useful baseline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VocalNeuralModelConfig:
    """Architecture and fixed baseline-residual settings."""

    input_channels: int = 12
    hidden_channels: int = 32
    token_channels: int = 24
    interaction_channels: int = 8
    head_channels: int = 32
    dropout: float = 0.10
    window_temperature: float = 0.05
    raw_similarity_logit_scale: float = 5.0
    key_proximity_logit_bonus: float = 0.15
    learn_key_proximity: bool = True
    maximum_residual_logit: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> VocalNeuralModelConfig:
        values = values or {}
        result = cls(
            input_channels=int(values.get("input_channels", cls.input_channels)),
            hidden_channels=int(values.get("hidden_channels", cls.hidden_channels)),
            token_channels=int(values.get("token_channels", cls.token_channels)),
            interaction_channels=int(
                values.get("interaction_channels", cls.interaction_channels)
            ),
            head_channels=int(values.get("head_channels", cls.head_channels)),
            dropout=float(values.get("dropout", cls.dropout)),
            window_temperature=float(
                values.get("window_temperature", cls.window_temperature)
            ),
            raw_similarity_logit_scale=float(
                values.get("raw_similarity_logit_scale", cls.raw_similarity_logit_scale)
            ),
            key_proximity_logit_bonus=float(
                values.get("key_proximity_logit_bonus", cls.key_proximity_logit_bonus)
            ),
            learn_key_proximity=bool(
                values.get("learn_key_proximity", cls.learn_key_proximity)
            ),
            maximum_residual_logit=float(
                values.get("maximum_residual_logit", cls.maximum_residual_logit)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        dimensions = (
            self.input_channels,
            self.hidden_channels,
            self.token_channels,
            self.interaction_channels,
            self.head_channels,
        )
        if min(dimensions) < 1 or self.input_channels != 12:
            raise ValueError("vocal neural dimensions must be positive with 12 chroma inputs")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("vocal neural dropout must lie in [0, 1)")
        if self.window_temperature <= 0.0:
            raise ValueError("window aggregation temperature must be positive")
        if self.raw_similarity_logit_scale <= 0.0:
            raise ValueError("raw similarity logit scale must be positive")
        if not 0.0 <= self.key_proximity_logit_bonus <= 1.0:
            raise ValueError("key proximity logit bonus must lie in [0, 1]")
        if not 0.0 <= self.maximum_residual_logit <= 5.0:
            raise ValueError("maximum residual logit must lie in [0, 5]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _TemporalChromaEncoder(nn.Module):
    """Turn chroma frames into normalized local-context tokens."""

    def __init__(self, config: VocalNeuralModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(config.input_channels, config.hidden_channels, kernel_size=5, padding=2),
            nn.GroupNorm(4, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_channels, config.token_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, chroma: Tensor) -> Tensor:
        if chroma.ndim != 3 or chroma.shape[1] != 12:
            raise ValueError("chroma encoder expects [pairs, 12, frames]")
        tokens = self.network(chroma.float()).transpose(1, 2)
        return F.normalize(tokens, dim=-1, eps=1e-8)


class VocalNeuralScorer(nn.Module):
    """Score local hum/vocal pairs and aggregate windows into song logits."""

    def __init__(self, config: VocalNeuralModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or VocalNeuralModelConfig()
        self.config.validate()

        # Hums and separated vocals have very different timbre and noise, so
        # they use independent encoders while meeting in one token space.
        self.query_encoder = _TemporalChromaEncoder(self.config)
        self.vocal_encoder = _TemporalChromaEncoder(self.config)
        channels = self.config.interaction_channels
        self.interaction = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        # Pooled interaction maps, six directional MaxSim statistics, global
        # cosine, raw chroma similarity, and circular key proximity.
        feature_width = 2 * channels + 6 + 1 + 2
        self.residual_head = nn.Sequential(
            nn.Linear(feature_width, self.config.head_channels),
            nn.LayerNorm(self.config.head_channels),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_channels, 1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Starting at a zero residual makes epoch zero reproduce the existing
        # scorer exactly instead of beginning with arbitrary neural rankings.
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):  # pragma: no cover - architecture invariant
            raise TypeError("vocal residual head must end with a linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _maxsim_statistics(values: Tensor, dimension: int) -> tuple[Tensor, Tensor, Tensor]:
        maxima = values.max(dim=dimension).values
        return maxima.mean(dim=1), maxima.std(dim=1, unbiased=False), maxima.min(dim=1).values

    def score_windows(
        self,
        query: Tensor,
        vocal: Tensor,
        raw_similarity: Tensor,
        key_proximity: Tensor,
    ) -> Tensor:
        """Return one logit per already key-aligned hum/vocal window pair."""

        features, baseline = self.extract_window_features(
            query, vocal, raw_similarity, key_proximity
        )
        residual = self.residual_head(features).squeeze(1)
        if self.config.maximum_residual_logit > 0.0:
            maximum = self.config.maximum_residual_logit
            residual = maximum * torch.tanh(residual / maximum)
        return baseline + residual

    def extract_window_features(
        self,
        query: Tensor,
        vocal: Tensor,
        raw_similarity: Tensor,
        key_proximity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Expose frozen pair features for lightweight scorer adaptation.

        A small personal collection cannot safely update both convolutional
        encoders.  Callers may freeze them, cache this deterministic feature
        vector, and train only ``residual_head``.  ``score_windows`` delegates
        here, so the normal runtime and adapted experiments remain bit-for-bit
        consistent before the head is changed.
        """

        if query.shape != vocal.shape or query.ndim != 3 or query.shape[1] != 12:
            raise ValueError("query and vocal pairs must share shape [pairs, 12, frames]")
        if raw_similarity.shape != key_proximity.shape or raw_similarity.ndim != 1:
            raise ValueError("raw similarity and key proximity must be pair-length vectors")
        if len(raw_similarity) != len(query):
            raise ValueError("pair feature count differs from chroma pair count")

        query_tokens = self.query_encoder(query)
        vocal_tokens = self.vocal_encoder(vocal)
        similarities = torch.einsum("bqd,brd->bqr", query_tokens, vocal_tokens)
        maps = self.interaction(similarities.unsqueeze(1))
        pooled = torch.cat(
            (
                F.adaptive_avg_pool2d(maps, 1).flatten(1),
                F.adaptive_max_pool2d(maps, 1).flatten(1),
            ),
            dim=1,
        )
        query_stats = self._maxsim_statistics(similarities, 2)
        vocal_stats = self._maxsim_statistics(similarities, 1)
        query_global = query_tokens.mean(dim=1)
        vocal_global = vocal_tokens.mean(dim=1)
        global_cosine = F.cosine_similarity(query_global, vocal_global, dim=1)
        learned_key_proximity = (
            key_proximity.float()
            if self.config.learn_key_proximity
            else torch.zeros_like(key_proximity, dtype=torch.float32)
        )
        explicit = torch.stack((raw_similarity.float(), learned_key_proximity), dim=1)
        features = torch.cat(
            (pooled, *[value.unsqueeze(1) for value in query_stats + vocal_stats],
             global_cosine.unsqueeze(1), explicit),
            dim=1,
        )
        baseline = (
            self.config.raw_similarity_logit_scale * (raw_similarity.float() - 0.5)
            + self.config.key_proximity_logit_bonus * key_proximity.float()
        )
        return features, baseline

    def forward(
        self,
        query: Tensor,
        vocal: Tensor,
        raw_similarity: Tensor,
        key_proximity: Tensor,
    ) -> Tensor:
        """Return song logits for tensors shaped [batch, candidates, windows, ...]."""

        if query.ndim != 5 or query.shape != vocal.shape:
            raise ValueError("vocal bags must have shape [batch, candidates, windows, 12, frames]")
        batch, candidates, windows, chroma, frames = query.shape
        flattened = batch * candidates * windows
        window_logits = self.score_windows(
            query.reshape(flattened, chroma, frames),
            vocal.reshape(flattened, chroma, frames),
            raw_similarity.reshape(flattened),
            key_proximity.reshape(flattened),
        ).reshape(batch, candidates, windows)
        temperature = self.config.window_temperature
        # Log-mean-exp is a smooth maximum. Subtracting log(window count) keeps
        # the score invariant if a future catalog uses more equal-count windows.
        return temperature * (
            torch.logsumexp(window_logits / temperature, dim=2) - math.log(windows)
        )


__all__ = ["VocalNeuralModelConfig", "VocalNeuralScorer"]
