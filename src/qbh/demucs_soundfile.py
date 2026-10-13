# This helper saves the Demucs output as a WAV file.
"""Run Demucs while writing WAV stems through libsndfile.

Demucs 4 delegates WAV output to Torchaudio. Newer Torchaudio releases route
that call through TorchCodec, whose native DLL must exactly match the installed
PyTorch/FFmpeg combination. Separation itself does not depend on TorchCodec.
This tiny adapter preserves Demucs's own clipping behavior and writes the final
PCM samples with SoundFile, which is already the project's verified decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import soundfile as sf
from demucs import separate
from demucs.audio import prevent_clip


def save_wav_with_soundfile(
    waveform: Any,
    path: str | Path,
    samplerate: int,
    bitrate: int = 320,
    clip: str = "rescale",
    bits_per_sample: int = 16,
    as_float: bool = False,
    preset: int = 2,
) -> None:
    """Match Demucs's WAV writer contract without importing TorchCodec."""

    del bitrate, preset  # MP3-only options retained for signature compatibility.
    destination = Path(path)
    if destination.suffix.lower() != ".wav":
        raise ValueError("the reproducible vocal-stem pipeline only permits WAV output")
    clipped = prevent_clip(waveform, mode=clip)
    samples = clipped.detach().cpu().transpose(0, 1).contiguous().numpy()
    if as_float:
        subtype = "FLOAT"
    else:
        subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(bits_per_sample)
        if subtype is None:
            raise ValueError(f"unsupported PCM bit depth: {bits_per_sample}")
    sf.write(destination, samples, samplerate, format="WAV", subtype=subtype)


def main() -> None:
    # `demucs.separate` imported save_audio into its own module namespace, so
    # replacing that exact reference is sufficient and leaves inference intact.
    separate.save_audio = save_wav_with_soundfile
    separate.main(sys.argv[1:])


if __name__ == "__main__":
    main()
