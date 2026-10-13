# This file starts the final demo website.
"""Launch the neural-versus-alignment browser."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "additive_chorus_runtime.json"
        ),
    )
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    arguments = build_parser().parse_args(argv)
    from qbh.vocal_neural_demo import launch_vocal_neural_demo

    return launch_vocal_neural_demo(
        arguments.runtime_config,
        arguments.port,
        inbrowser=not arguments.no_browser,
    )


if __name__ == "__main__":
    main()
