# This file runs Demucs and saves a list of the new stems.
"""Content-addressed, resumable vocal-stem ingestion with Demucs.

The separator is an offline catalog-building step, never part of an interactive
search request.  Every output is bound to the original song bytes, the exact
Demucs package/model, and all inference settings.  A stopped run can therefore
reuse completed stems without confusing outputs from different experiments.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf


def _file_sha256(path: str | Path) -> str:
    """Hash a file without importing the model/training dependency graph."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible provenance with a stable serialization."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_dump(value: Mapping[str, Any], destination: str | Path) -> None:
    """Publish JSON atomically so an interrupted run never leaves a partial file."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_catalog(path: str | Path) -> dict[str, Any]:
    """Load only the catalog fields required by the separation job.

    Keeping this ingestion utility independent of training modules avoids loading
    PyTorch just to plan cached files. The stricter runtime catalog validation
    remains unchanged and is still used by retrieval/evaluation.
    """

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    songs = payload.get("songs")
    if int(payload.get("schema_version", -1)) != 1 or not isinstance(songs, list):
        raise ValueError("closed catalog must use schema_version 1 and list songs")
    normalized: list[dict[str, str]] = []
    for raw in songs:
        if not isinstance(raw, Mapping):
            raise TypeError("each catalog song must be an object")
        song = {
            "song_id": str(raw.get("song_id", "")).strip(),
            "title": str(raw.get("title", "")).strip(),
            "audio": str(raw.get("audio", "")).strip(),
        }
        if not all(song.values()):
            raise ValueError("catalog songs require song_id, title, and audio")
        normalized.append(song)
    song_ids = [song["song_id"] for song in normalized]
    if len(song_ids) != len(set(song_ids)):
        raise ValueError("catalog song IDs must be unique")
    catalog_root = Path(str(payload.get("catalog_root", ""))).expanduser()
    if not catalog_root.is_absolute():
        catalog_root = resolved.parent / catalog_root
    return {"catalog_root": catalog_root.resolve(), "songs": normalized}


@dataclass(frozen=True)
class VocalStemIngestionConfig:
    """Exact separator identity and storage policy for one ingestion run."""

    model_name: str
    model_weight_path: Path
    model_weight_sha256: str
    demucs_version: str
    device: str = "cpu"
    stem: str = "vocals"
    shifts: int = 1
    overlap: float = 0.25
    segment_seconds: int | None = None
    jobs: int = 0
    output_format: str = "wav"

    def __post_init__(self) -> None:
        if not self.model_name or self.stem != "vocals":
            raise ValueError("the pilot requires a named two-stem vocal model")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Demucs device must be cpu or cuda")
        if self.shifts < 0 or not 0.0 <= self.overlap < 1.0 or self.jobs < 0:
            raise ValueError("invalid Demucs shifts/overlap/jobs settings")
        if self.segment_seconds is not None and self.segment_seconds < 1:
            raise ValueError("Demucs segment_seconds must be positive")
        if self.output_format != "wav":
            raise ValueError("the reproducible pilot currently stores lossless WAV stems")
        digest = self.model_weight_sha256.lower()
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("model_weight_sha256 must be a hexadecimal SHA-256")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any], *, project_root: str | Path
    ) -> VocalStemIngestionConfig:
        root = Path(project_root).expanduser().resolve()
        weight = Path(str(values["model_weight_path"])).expanduser()
        weight = weight.resolve() if weight.is_absolute() else (root / weight).resolve()
        config = cls(
            model_name=str(values["model_name"]),
            model_weight_path=weight,
            model_weight_sha256=str(values["model_weight_sha256"]).lower(),
            demucs_version=str(values["demucs_version"]),
            device=str(values.get("device", "cpu")),
            stem=str(values.get("stem", "vocals")),
            shifts=int(values.get("shifts", 1)),
            overlap=float(values.get("overlap", 0.25)),
            segment_seconds=(
                int(values["segment_seconds"])
                if values.get("segment_seconds") is not None
                else None
            ),
            jobs=int(values.get("jobs", 0)),
            output_format=str(values.get("output_format", "wav")),
        )
        if not config.model_weight_path.is_file():
            raise FileNotFoundError(f"cached Demucs weight is missing: {weight}")
        if _file_sha256(config.model_weight_path) != config.model_weight_sha256:
            raise ValueError("cached Demucs model weight checksum changed")
        installed = importlib.metadata.version("demucs")
        if installed != config.demucs_version:
            raise ValueError(
                f"installed Demucs {installed} differs from pinned {config.demucs_version}"
            )
        return config

    def identity_payload(self) -> dict[str, Any]:
        """Fields that can change the separated waveform."""

        return {
            "engine": "demucs",
            "wav_writer": "soundfile_pcm_v1",
            "model_name": self.model_name,
            "model_weight_sha256": self.model_weight_sha256,
            "demucs_version": self.demucs_version,
            "device": self.device,
            "stem": self.stem,
            "shifts": self.shifts,
            "overlap": self.overlap,
            "segment_seconds": self.segment_seconds,
            "jobs": self.jobs,
            "output_format": self.output_format,
        }


@dataclass(frozen=True)
class PlannedVocalStem:
    """One catalog song and its deterministic vocal-stem destination."""

    song_id: str
    title: str
    source_path: Path
    source_sha256: str
    source_duration_seconds: float
    fingerprint: str
    output_path: Path
    metadata_path: Path
    staging_input_path: Path
    staging_vocal_path: Path
    staging_track_directory: Path


def _audio_info(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    duration = float(info.frames / info.samplerate) if info.samplerate else 0.0
    return {
        "duration_seconds": duration,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def load_ingestion_settings(
    config_path: str | Path,
) -> tuple[dict[str, Any], VocalStemIngestionConfig, Path, Path, Path]:
    """Load paths and validate the exact installed separator artifact."""

    path = Path(config_path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(raw.get("format_version", -1)) != 1
        or raw.get("pipeline") != "demucs_vocal_stem_ingestion"
    ):
        raise ValueError("unsupported vocal-stem ingestion configuration")
    project_root = path.parents[2]

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (
            candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
        )

    catalog_path = resolve(str(raw["catalog_config"]))
    output_root = resolve(str(raw["output_root"]))
    manifest_path = resolve(str(raw["manifest_path"]))
    config = VocalStemIngestionConfig.from_mapping(raw["separator"], project_root=project_root)
    return raw, config, catalog_path, output_root, manifest_path


def plan_vocal_stems(
    catalog_config: str | Path,
    output_root: str | Path,
    config: VocalStemIngestionConfig,
) -> list[PlannedVocalStem]:
    """Hash the catalog and map every song to a content-addressed output."""

    catalog = _load_catalog(catalog_config)
    root = Path(output_root).expanduser().resolve()
    separator_sha = _canonical_sha256(config.identity_payload())
    staging = root / ".staging" / separator_sha
    plans: list[PlannedVocalStem] = []
    for song in sorted(catalog["songs"], key=lambda row: str(row["song_id"])):
        song_id = str(song["song_id"])
        source = (catalog["catalog_root"] / song["audio"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"catalog song is missing: {source}")
        source_sha = _file_sha256(source)
        fingerprint = _canonical_sha256(
            {
                "format_version": 1,
                "source_sha256": source_sha,
                "separator_sha256": separator_sha,
            }
        )
        destination = root / song_id / fingerprint
        alias = f"{song_id}-{source_sha[:12]}"
        track_directory = staging / "separated" / config.model_name / alias
        plans.append(
            PlannedVocalStem(
                song_id=song_id,
                title=str(song["title"]),
                source_path=source,
                source_sha256=source_sha,
                source_duration_seconds=float(_audio_info(source)["duration_seconds"]),
                fingerprint=fingerprint,
                output_path=destination / "vocals.wav",
                metadata_path=destination / "metadata.json",
                staging_input_path=staging / "inputs" / f"{alias}{source.suffix.lower()}",
                staging_vocal_path=track_directory / "vocals.wav",
                staging_track_directory=track_directory,
            )
        )
    return plans


def _valid_completed_metadata(
    plan: PlannedVocalStem, config: VocalStemIngestionConfig
) -> dict[str, Any] | None:
    if not plan.output_path.is_file() or not plan.metadata_path.is_file():
        return None
    try:
        metadata = json.loads(plan.metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("fingerprint") != plan.fingerprint
            or metadata.get("source", {}).get("sha256") != plan.source_sha256
            or metadata.get("separator") != config.identity_payload()
            or metadata.get("vocals", {}).get("sha256") != _file_sha256(plan.output_path)
        ):
            return None
        info = _audio_info(plan.output_path)
        if info["duration_seconds"] < 30.0 or info["sample_rate"] != 44_100:
            return None
        return metadata
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError):
        return None


def _promote_staged_vocal(
    plan: PlannedVocalStem,
    config: VocalStemIngestionConfig,
) -> dict[str, Any]:
    info = _audio_info(plan.staging_vocal_path)
    if info["duration_seconds"] < 30.0 or info["sample_rate"] != 44_100:
        raise ValueError(f"Demucs produced an invalid vocal stem for {plan.song_id}")
    if abs(info["duration_seconds"] - plan.source_duration_seconds) > 2.0:
        raise ValueError(f"vocal/source durations disagree for {plan.song_id}")
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.output_path.exists():
        raise FileExistsError(f"refusing to replace an unverified vocal stem: {plan.output_path}")
    os.replace(plan.staging_vocal_path, plan.output_path)
    metadata = {
        "format_version": 1,
        "song_id": plan.song_id,
        "title": plan.title,
        "fingerprint": plan.fingerprint,
        "completed_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(plan.source_path),
            "sha256": plan.source_sha256,
            "duration_seconds": plan.source_duration_seconds,
        },
        "separator": config.identity_payload(),
        "vocals": {
            "path": str(plan.output_path),
            "sha256": _file_sha256(plan.output_path),
            **info,
        },
    }
    _atomic_json_dump(metadata, plan.metadata_path)
    # This directory contains only Demucs-generated staging files for this
    # exact song/fingerprint.  Removing the unused no_vocals stem bounds disk
    # use and cannot touch the user's source MP3.
    if plan.staging_track_directory.is_dir():
        shutil.rmtree(plan.staging_track_directory)
    plan.staging_input_path.unlink(missing_ok=True)
    return metadata


def _prepare_staging_input(plan: PlannedVocalStem) -> None:
    plan.staging_input_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.staging_input_path.is_file():
        if _file_sha256(plan.staging_input_path) == plan.source_sha256:
            return
        plan.staging_input_path.unlink()
    try:
        os.link(plan.source_path, plan.staging_input_path)
    except OSError:
        # Hard links are preferred because the source and project normally
        # share a volume.  Copying is a safe fallback for a catalog elsewhere.
        shutil.copy2(plan.source_path, plan.staging_input_path)


def demucs_command(
    config: VocalStemIngestionConfig,
    plans: Sequence[PlannedVocalStem],
    staging_output_root: Path,
) -> list[str]:
    """Build the shell-free command passed directly to subprocess."""

    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("demucs_soundfile.py")),
        "--name",
        config.model_name,
        "--device",
        config.device,
        "--shifts",
        str(config.shifts),
        "--overlap",
        str(config.overlap),
        "--two-stems",
        config.stem,
        "--out",
        str(staging_output_root),
    ]
    if config.segment_seconds is not None:
        command.extend(("--segment", str(config.segment_seconds)))
    if config.jobs:
        command.extend(("--jobs", str(config.jobs)))
    command.extend(str(plan.staging_input_path) for plan in plans)
    return command


def write_vocal_manifest(
    plans: Sequence[PlannedVocalStem],
    config: VocalStemIngestionConfig,
    manifest_path: str | Path,
    *,
    config_path: str | Path,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a complete catalog-status manifest after any partial run."""

    records: list[dict[str, Any]] = []
    complete = 0
    for plan in plans:
        metadata = _valid_completed_metadata(plan, config)
        status = "complete" if metadata is not None else "pending"
        complete += int(status == "complete")
        records.append(
            {
                "song_id": plan.song_id,
                "title": plan.title,
                "status": status,
                "fingerprint": plan.fingerprint,
                "source_path": str(plan.source_path),
                "source_sha256": plan.source_sha256,
                "vocals_path": str(plan.output_path),
                "vocals_sha256": (metadata["vocals"]["sha256"] if metadata is not None else None),
                "metadata_path": str(plan.metadata_path),
            }
        )
    payload: dict[str, Any] = {
        "format_version": 1,
        "manifest_kind": "content_addressed_vocal_stems",
        "ingestion_config": str(Path(config_path).expanduser().resolve()),
        "ingestion_config_sha256": _file_sha256(config_path),
        "separator": config.identity_payload(),
        "separator_sha256": _canonical_sha256(config.identity_payload()),
        "summary": {
            "pending": len(records) - complete,
        },
        "last_log": str(Path(log_path).resolve()) if log_path is not None else None,
        "songs": records,
    }
    payload["manifest_content_sha256"] = _canonical_sha256(payload)
    _atomic_json_dump(payload, manifest_path)
    return payload


def run_vocal_ingestion(
    config_path: str | Path,
    *,
    song_ids: Sequence[str] | None = None,
    maximum_songs: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resume available outputs, run Demucs once, then atomically promote stems."""

    raw, config, catalog_path, output_root, manifest_path = load_ingestion_settings(config_path)
    plans = plan_vocal_stems(catalog_path, output_root, config)
    requested = set(song_ids or [])
    unknown = sorted(requested - {plan.song_id for plan in plans})
    if unknown:
        raise ValueError("unknown requested song IDs: " + ", ".join(unknown))
    selected = [plan for plan in plans if not requested or plan.song_id in requested]
    if maximum_songs is not None:
        if maximum_songs < 1:
            raise ValueError("maximum_songs must be positive")
        selected = selected[:maximum_songs]

    # Recover a finished staging output from an interrupted earlier command.
    for plan in selected:
        if _valid_completed_metadata(plan, config) is None and plan.staging_vocal_path.is_file():
            _promote_staged_vocal(plan, config)
    pending = [plan for plan in selected if _valid_completed_metadata(plan, config) is None]
    if dry_run or not pending:
        return write_vocal_manifest(
            plans,
            config,
            manifest_path,
            config_path=config_path,
        )

    for plan in pending:
        _prepare_staging_input(plan)
    separator_sha = _canonical_sha256(config.identity_payload())
    staging_output = output_root / ".staging" / separator_sha / "separated"
    log_directory = output_root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"demucs-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
    command = demucs_command(config, pending, staging_output)
    print(f"Separating {len(pending)} song(s) with {config.model_name} on {config.device}.")
    print(f"Progress log: {log_path}")
    sys.stdout.flush()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path(config_path).expanduser().resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        write_vocal_manifest(
            plans,
            config,
            manifest_path,
            config_path=config_path,
            log_path=log_path,
        )
        raise RuntimeError(f"Demucs failed with exit code {completed.returncode}; see {log_path}")
    missing = [plan.song_id for plan in pending if not plan.staging_vocal_path.is_file()]
    if missing:
        raise RuntimeError("Demucs completed without expected vocal stems: " + ", ".join(missing))
    for plan in pending:
        _promote_staged_vocal(plan, config)
    return write_vocal_manifest(
        plans,
        config,
        manifest_path,
        config_path=config_path,
        log_path=log_path,
    )


__all__ = [
    "PlannedVocalStem",
    "VocalStemIngestionConfig",
    "demucs_command",
    "load_ingestion_settings",
    "plan_vocal_stems",
    "run_vocal_ingestion",
    "write_vocal_manifest",
]
