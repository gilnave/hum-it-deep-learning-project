# Small tools for hashes, files, devices, and random seeds.
"""Small shared utilities for deterministic and traceable experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_TORCH_ACCESS_DENIED = re.compile(
    r"open file failed with error code:\s*5",
    flags=re.IGNORECASE,
)


def _is_transient_access_error(error: BaseException) -> bool:
    """Return whether DriveFS may succeed if this exact write is retried.

    PyTorch's zip writer reports Windows access denial as RuntimeError instead
    of PermissionError. Matching only that narrow message keeps real
    serialization errors (bad objects, full disks, and so on) loud.
    """

    if isinstance(error, PermissionError):
        return True
    if isinstance(error, OSError):
        return getattr(error, "winerror", None) in {5, 32, 33}
    return isinstance(error, RuntimeError) and bool(_TORCH_ACCESS_DENIED.search(str(error)))


def _remove_temporary_file(path: Path) -> None:
    """Best-effort cleanup for a partial sibling left by a failed writer."""

    delay = 0.05
    for attempt in range(3):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as error:
            if not _is_transient_access_error(error) or attempt == 2:
                return
            time.sleep(delay)
            delay *= 2


def atomic_write_path(
    destination: str | Path,
    writer: Callable[[Path], None],
    *,
    attempts: int = 8,
) -> None:
    """Write a complete sibling and atomically replace the destination.

    Google Drive for desktop can briefly lock either a temporary file or its
    destination while synchronizing. Every writer retry therefore receives a
    new same-directory path, whereas a replacement retry reuses the already
    complete file. The old destination remains untouched until os.replace
    succeeds.
    """

    if attempts < 1:
        raise ValueError("attempts must be at least one")

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer_delay = 0.1

    for writer_attempt in range(attempts):
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            writer(temporary_path)
        except BaseException as error:
            if temporary_path is not None:
                _remove_temporary_file(temporary_path)
            if not _is_transient_access_error(error) or writer_attempt == attempts - 1:
                raise
            time.sleep(writer_delay)
            writer_delay = min(writer_delay * 2, 2.0)
            continue

        try:
            replace_delay = 0.1
            for replace_attempt in range(attempts):
                try:
                    os.replace(temporary_path, output_path)
                    return
                except OSError as error:
                    if not _is_transient_access_error(error) or replace_attempt == attempts - 1:
                        raise
                    time.sleep(replace_delay)
                    replace_delay = min(replace_delay * 2, 2.0)
        finally:
            _remove_temporary_file(temporary_path)

    raise RuntimeError("atomic writer exhausted without returning or raising")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without claiming impossible determinism."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    """Choose the requested device, falling back to CPU for ``auto``."""

    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-capable PyTorch device is available.")
    return torch.device(requested)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a stable file fingerprint without loading the entire file into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(payload: dict[str, Any]) -> str:
    """Return a short fingerprint for a JSON-serializable configuration."""

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def make_run_directory(root: str | Path, label: str) -> Path:
    """Create a timestamped result folder using Windows-safe characters."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{timestamp}_{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    """Write JSON through a temporary sibling so partial files are not mistaken as results."""

    def write_json(temporary_path: Path) -> None:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)

    atomic_write_path(path, write_json)
