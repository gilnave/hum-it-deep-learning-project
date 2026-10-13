# This file makes the same windows from Demucs vocals.
"""Full-song local retrieval over reproducibly separated vocal stems.

The separator is deliberately kept outside the online search path. Catalog
songs are separated once, verified against a content-addressed manifest, and
then indexed by the same key-invariant local-window scorer used for the clean
mixture-versus-vocals experiment. This keeps the comparison honest: only the
reference audio source changes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qbh.symbolic.audio_song_windows import (
    FullSongWindowChromaBackend,
    SongAudioSource,
    SongWindowChromaConfig,
)
from qbh.symbolic.evaluation import EvaluationError, canonical_sha256
from qbh.utils import file_sha256


def load_vocal_stem_sources(
    manifest_path: str | Path,
    *,
    expected_song_ids: Sequence[str] | None = None,
) -> tuple[list[SongAudioSource], dict[str, Any]]:
    """Load a complete vocal manifest and reverify every stem's bytes.

    A partial Demucs run is useful for resume, but it must never silently make
    a smaller retrieval catalog. Therefore this loader rejects even one pending
    song and, when supplied, enforces the exact expected catalog identities.
    """

    path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"could not read vocal-stem manifest: {path}") from error
    if (
        int(payload.get("format_version", -1)) != 1
        or payload.get("manifest_kind") != "content_addressed_vocal_stems"
    ):
        raise EvaluationError("unsupported vocal-stem manifest")
    rows = payload.get("songs")
    summary = payload.get("summary")
    if not isinstance(rows, list) or not isinstance(summary, Mapping) or not rows:
        raise EvaluationError("vocal-stem manifest is missing songs/summary")
    if int(summary.get("pending", -1)) != 0:
        raise EvaluationError("vocal-stem manifest is incomplete")

    sources: list[SongAudioSource] = []
    verified_rows: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            raise EvaluationError("vocal-stem manifest contains an incomplete row")
        song_id = str(row.get("song_id", "")).strip()
        title = str(row.get("title", "")).strip()
        expected_hash = str(row.get("vocals_sha256", "")).lower()
        audio_path = Path(str(row.get("vocals_path", ""))).expanduser().resolve()
        if not song_id or not title or len(expected_hash) != 64 or not audio_path.is_file():
            raise EvaluationError(f"invalid vocal-stem row for {song_id or '<unknown>'}")
        actual_hash = file_sha256(audio_path)
        if actual_hash != expected_hash:
            raise EvaluationError(f"vocal stem changed after ingestion: {song_id}")
        sources.append(
            SongAudioSource(
                song_id=song_id,
                title=title,
                audio_path=audio_path,
                source_sha256=actual_hash,
            )
        )
        verified_rows.append({"song_id": song_id, "vocals_sha256": actual_hash})

    actual_ids = [source.song_id for source in sources]
    if len(actual_ids) != len(set(actual_ids)):
        raise EvaluationError("vocal-stem manifest contains duplicate song IDs")
    if expected_song_ids is not None and set(actual_ids) != set(expected_song_ids):
        raise EvaluationError("vocal-stem manifest differs from the expected catalog")
    verified = {
        "path": str(path),
        "sha256": file_sha256(path),
        "content_sha256": canonical_sha256(payload),
        "verified_stems_sha256": canonical_sha256(
            sorted(verified_rows, key=lambda row: row["song_id"])
        ),
        "songs": len(sources),
        "separator": payload.get("separator"),
        "separator_sha256": payload.get("separator_sha256"),
    }
    return sorted(sources, key=lambda source: source.song_id), verified


class VocalStemWindowChromaBackend(FullSongWindowChromaBackend):
    """Rank hums against equal-count local windows from vocals only."""

    name = "vocal_stem_window_chroma"

    def __init__(
        self,
        sources: Sequence[SongAudioSource],
        cache_directory: str | Path,
        config: SongWindowChromaConfig,
        *,
        vocal_manifest: Mapping[str, Any],
        selection_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.vocal_manifest = dict(vocal_manifest)
        super().__init__(
            sources,
            cache_directory,
            config,
            selection_metadata=selection_metadata,
        )

    def provenance(self) -> Mapping[str, Any]:
        provenance = dict(super().provenance())
        provenance.update(
            {
                "backend": self.name,
                "reference_source": "demucs_vocal_stems",
                "vocal_manifest": self.vocal_manifest,
            }
        )
        return provenance


__all__ = ["VocalStemWindowChromaBackend", "load_vocal_stem_sources"]
