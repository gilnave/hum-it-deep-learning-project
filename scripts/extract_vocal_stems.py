# This script uses Demucs to keep only the vocal part of each song.
"""Build or resume content-addressed Demucs vocal stems for the song catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qbh.vocal_stem_ingestion import run_vocal_ingestion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "vocal_stems.json",
    )
    parser.add_argument(
        "--song-id",
        action="append",
        default=None,
        help="Process only this song ID; repeat for multiple songs.",
    )
    parser.add_argument("--maximum-songs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    manifest = run_vocal_ingestion(
        arguments.config,
        song_ids=arguments.song_id,
        maximum_songs=arguments.maximum_songs,
        dry_run=arguments.dry_run,
    )
    print(json.dumps(manifest["summary"], indent=2))
    print(
        f"Manifest: {arguments.config.resolve().parents[1] / 'artifacts/vocal_stems/htdemucs/manifest.json'}"
    )


if __name__ == "__main__":
    main()
