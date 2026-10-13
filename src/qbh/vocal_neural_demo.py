# This file builds the simple Gradio demo page.
"""Browser UI for the single general neural and alignment rankings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qbh.additive_chorus_runtime import AdditiveChorusRuntime
from qbh.vocal_neural_runtime import VocalComparisonHit, VocalNeuralRuntime

RESULT_HEADERS = [
    "Rank",
    "Song",
    "Decision score",
    "Original score",
    "Chorus points / match",
    "Rank in other list",
    "Raw similarity",
    "Matched vocal window",
    "Key shift",
    "Circular distance / bonus",
    "Vocal stem",
]


class VocalNeuralDemoController:
    """UI-independent adapter exposing two intentionally unfused rankings."""

    def __init__(self, runtime_config: str | Path) -> None:
        path = Path(runtime_config).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backend") == "bounded_additive_chorus_points":
            self.runtime = AdditiveChorusRuntime(path)
        else:
            self.runtime = VocalNeuralRuntime(path)

    def status_markdown(self) -> str:
        return (
            "**Catalog:** verified Demucs vocal stems  \n"
            "**Model:** one general vocal model  \n"
            "**Decision:** general neural list plus an independent alignment list"
        )

    def product_rule_markdown(self) -> str:
        """Return the detailed scoring rule shown on demand in the browser."""
        if isinstance(self.runtime, AdditiveChorusRuntime):
            points = self.runtime.point_config
            return (
                "Bounded additive chorus points, capped at "
                f"**{points.maximum_points:.1f} points**. A strong chorus-melody match "
                f"earns up to {points.top_one_points:.1f}; covering at least "
                f"{points.full_coverage_threshold:.0%} of its matched window adds "
                f"{points.full_coverage_points:.1f}; matching distinct repeated chorus "
                "occurrences adds a small fixed award. The award never multiplies a "
                "score, never subtracts points, and timestamp overlap alone earns zero."
            )
        return (
            "Each candidate is compared specifically against its chorus windows; that "
            "chorus-restricted melody score has "
            f"{self.runtime.neural_chorus_reward:.0f}× the weight of its global score. "
            "Merely landing inside a chorus earns nothing."
        )

    def catalog_markdown(self) -> str:
        titles = sorted(str(song["title"]) for song in self.runtime.songs.values())
        return "## Searchable songs\n\n" + "\n".join(f"- {title}" for title in titles)

    @staticmethod
    def _rows(hits: tuple[VocalComparisonHit, ...]) -> list[list[Any]]:
        def chorus_text(hit: VocalComparisonHit) -> str:
            if hit.chorus_total_occurrences > 0:
                assert hit.chorus_start_seconds is not None
                assert hit.chorus_end_seconds is not None
                return (
                    f"{hit.chorus_bonus:+.3f} points; strength "
                    f"+{hit.chorus_strength_points:.1f}; coverage "
                    f"+{hit.chorus_coverage_points:.1f}; repetition "
                    f"+{hit.chorus_repetition_points:.1f}; "
                    f"{hit.chorus_matched_occurrences}/"
                    f"{hit.chorus_total_occurrences} distinct chorus occurrences; "
                    f"{hit.chorus_coverage:.0%} best-window overlap; chorus "
                    f"{hit.chorus_start_seconds:.1f}–{hit.chorus_end_seconds:.1f} s"
                )
            if hit.chorus_evidence <= 0.0:
                return "No bonus"
            assert hit.chorus_start_seconds is not None
            assert hit.chorus_end_seconds is not None
            return (
                f"{hit.chorus_bonus:+.3f}; {hit.chorus_coverage:.0%} overlap; "
                f"{hit.chorus_supporting_windows} window(s); chorus "
                f"{hit.chorus_start_seconds:.1f}–{hit.chorus_end_seconds:.1f} s"
            )

        return [
            [
                hit.rank,
                hit.title,
                round(hit.score, 4),
                round(hit.base_score, 4),
                chorus_text(hit),
                hit.other_rank,
                round(hit.raw_similarity, 4),
                f"{hit.window_start_seconds:.1f}–{hit.window_end_seconds:.1f} s",
                f"{hit.transposition_semitones:+d} semitones",
                f"{hit.circular_key_distance} semitones / +{hit.key_distance_bonus:.3f}",
                hit.vocal_path,
            ]
            for hit in hits
        ]

    def search(
        self, query_path: str | None, top_k: int
    ) -> tuple[list[list[Any]], list[list[Any]], str]:
        if not query_path:
            return [], [], "Record or upload a humming query first."
        try:
            result = self.runtime.search(query_path, top_k)
        except Exception as error:  # noqa: BLE001 - browser callback boundary
            return [], [], f"**Search failed:** {error}"
        if isinstance(self.runtime, AdditiveChorusRuntime):
            explanation = (
                "The general model and bounded additive chorus points "
                "are active. The neural and alignment lists remain independent and "
                "their scores are never fused."
            )
        else:
            explanation = (
                "The general model is active. The neural and alignment "
                "lists remain independent and their scores are never fused."
            )
        return (
            self._rows(result.neural),
            self._rows(result.alignment),
            explanation,
        )


def create_vocal_neural_demo(controller: VocalNeuralDemoController) -> Any:
    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install the project requirements to use the browser demo") from error

    with gr.Blocks(title="Humming-Based Song Retrival Model") as application:
        gr.Markdown(
            "# Humming-Based Song Retrival Model\n"
            "Record one hum and inspect two separate result lists for the same "
            "the vocals-only candidates."
        )
        gr.Markdown(controller.status_markdown())
        with gr.Accordion("Product rule", open=False):
            gr.Markdown(controller.product_rule_markdown())
        with gr.Tab("Hum-It"):
            query = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Humming query"
            )
            top_k = gr.Slider(1, 10, value=10, step=1, label="Results per list")
            button = gr.Button("Go", variant="primary")
            gr.Markdown("## Neural Model Results")
            neural_results = gr.Dataframe(
                headers=RESULT_HEADERS,
                datatype=[
                    "number",
                    "str",
                    "number",
                    "number",
                    "str",
                    "number",
                    "number",
                    "str",
                    "str",
                    "str",
                    "str",
                ],
                interactive=False,
                wrap=True,
                show_label=False,
            )
            # Alignment is useful for manual comparison but is intentionally
            # secondary to the neural ranking. Start it collapsed so the main
            # result list stays compact; clicking the heading reveals the table.
            with gr.Accordion(
                "Melody Alignment Only",
                open=False,
            ):
                alignment_results = gr.Dataframe(
                    headers=RESULT_HEADERS,
                    datatype=[
                        "number",
                        "str",
                        "number",
                        "number",
                        "str",
                        "number",
                        "number",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    interactive=False,
                    wrap=True,
                    show_label=False,
                )
            message = gr.Markdown()
            button.click(
                controller.search,
                inputs=[query, top_k],
                outputs=[neural_results, alignment_results, message],
            )
        with gr.Tab("Song list"):
            gr.Markdown(controller.catalog_markdown())
    return application


def launch_vocal_neural_demo(
    runtime_config: str | Path,
    port: int = 7866,
    *,
    inbrowser: bool = True,
) -> Any:
    if not 1 <= int(port) <= 65_535:
        raise ValueError("port must be between 1 and 65535")

    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install the project requirements to use the browser demo") from error

    # Reuse the original browser's exact black, white, and light-blue palette so
    # switching retrieval backends does not also change the product's visual
    # identity. Gradio 6 applies themes at launch time, rather than when Blocks
    # are constructed.
    from qbh.demo import DEMO_CSS, _demo_theme

    controller = VocalNeuralDemoController(runtime_config)
    return create_vocal_neural_demo(controller).launch(
        server_name="127.0.0.1",
        server_port=int(port),
        share=False,
        inbrowser=inbrowser,
        theme=_demo_theme(gr),
        css=DEMO_CSS,
    )


__all__ = [
    "VocalNeuralDemoController",
    "create_vocal_neural_demo",
    "launch_vocal_neural_demo",
]
