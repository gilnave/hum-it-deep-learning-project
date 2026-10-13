# These are shared tools for loading and changing audio files.
"""Audio loading and log-mel feature extraction for query-by-humming.

The learned system receives two different kinds of recordings: a person's hum
and a studio recording of the corresponding song fragment.  This module keeps
their signal processing identical except for one deliberate difference: leading
and trailing silence is removed from humming queries.

All public functions operate on PyTorch tensors shaped ``[channels, samples]``.
The final model input is always ``[1, n_mels, frames]``.  Keeping those shapes
explicit avoids a surprisingly common source of silent broadcasting bugs.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile
import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional

# Streamed drives can briefly report an existing file as absent while fetching
# its metadata. Valid local paths still return immediately; only an initially
# absent path waits for this bounded, exponentially backed-off retry window.
_FILE_VISIBILITY_RETRY_DELAYS_SECONDS = (0.1, 0.2, 0.4, 0.8, 1.0)

# Manifest paths are provenance-bearing and must stay byte-identical when the
# project is moved. Only the portion below a literal ``data/raw`` directory is
# portable; the preceding directories identify the machine that prepared it.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _relocated_raw_audio_path(path: Path) -> Path | None:
    """Return this project's equivalent for a path below literal ``data/raw``."""

    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("data", "raw"):
            return _PROJECT_ROOT.joinpath(*parts[index:])
    return None


def _config_value(config: Any, name: str, default: Any) -> Any:
    """Read one value from either a dictionary or a namespace-like object."""

    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _ordered_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    """Convert a scalar or two-item sequence to an ordered numeric range."""

    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if len(value) != 2:
        raise ValueError(f"Expected a two-item range, received {value!r}")
    lower, upper = float(value[0]), float(value[1])
    return (lower, upper) if lower <= upper else (upper, lower)


def _symmetric_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    """Interpret a scalar gain as ``[-gain, +gain]`` for convenient YAML use."""

    if isinstance(value, (int, float)):
        magnitude = abs(float(value))
        return -magnitude, magnitude
    return _ordered_range(value, default)


@dataclass(frozen=True)
class AudioSettings:
    """Small, serializable description of the complete audio front end.

    ``from_config`` intentionally accepts dictionaries and ``SimpleNamespace``
    objects.  The project YAML loader returns namespaces, while tests and small
    notebooks often use plain dictionaries.
    """

    sample_rate: int = 16_000
    duration_seconds: float = 5.0
    n_mels: int = 80
    n_fft: int = 1_024
    win_length: int = 1_024
    hop_length: int = 256
    f_min: float = 50.0
    f_max: float | None = 7_600.0
    trim_top_db: float = 35.0
    trim_frame_length: int = 1_024
    trim_hop_length: int = 256
    normalize_features: bool = True

    # These deliberately modest augmentations are inexpensive on a laptop CPU.
    augmentation_enabled: bool = False
    gain_db: tuple[float, float] = (-4.0, 4.0)
    noise_snr_db: tuple[float, float] = (20.0, 40.0)
    # A fast resampling perturbation changes both musical key and singing speed.
    # Applying it independently to the hum and reference prevents the encoder
    # from treating absolute pitch as a song identity shortcut.
    pitch_speed_semitones: tuple[float, float] = (0.0, 0.0)
    pitch_speed_probability: float = 0.0
    # Shifting the normalized log-mel image is a cheap pitch-like perturbation
    # that does not also alter tempo.  It is especially useful for conservative
    # fine-tuning of a model that already learned strong song discrimination.
    maximum_frequency_shift_bins: int = 0
    frequency_shift_probability: float = 0.0
    time_mask_frames: int = 20
    frequency_mask_bins: int = 8

    def __post_init__(self) -> None:
        """Reject invalid settings early, before a long training run begins."""

        if self.sample_rate <= 0 or self.duration_seconds <= 0:
            raise ValueError("sample_rate and duration_seconds must be positive")
        if self.n_mels <= 0 or self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_mels, n_fft, and hop_length must be positive")
        if self.win_length <= 0 or self.win_length > self.n_fft:
            raise ValueError("win_length must be positive and no larger than n_fft")
        if self.f_max is not None and self.f_max > self.sample_rate / 2:
            raise ValueError("f_max cannot exceed the Nyquist frequency")
        if self.trim_top_db <= 0:
            raise ValueError("trim_top_db must be positive")
        if not 0.0 <= self.pitch_speed_probability <= 1.0:
            raise ValueError("pitch_speed_probability must be between zero and one")
        if max(abs(value) for value in self.pitch_speed_semitones) > 12.0:
            raise ValueError("pitch_speed_semitones must remain within one octave")
        if self.maximum_frequency_shift_bins < 0:
            raise ValueError("maximum_frequency_shift_bins cannot be negative")
        if not 0.0 <= self.frequency_shift_probability <= 1.0:
            raise ValueError("frequency_shift_probability must be between zero and one")

    @property
    def target_samples(self) -> int:
        """Exact number of waveform samples presented to the spectrogram."""

        return round(self.sample_rate * self.duration_seconds)

    @classmethod
    def from_config(cls, config: Any | None) -> AudioSettings:
        """Create settings from the project's flexible YAML configuration.

        ``duration_seconds`` is the preferred name.  ``clip_seconds`` remains an
        accepted alias because it is natural in ad-hoc experiment configs.
        Augmentation values may be nested below ``audio.augmentation``.
        """

        augmentation = _config_value(config, "augmentation", None)
        duration = _config_value(
            config,
            "duration_seconds",
            _config_value(config, "clip_seconds", cls.duration_seconds),
        )
        gain = _config_value(augmentation, "gain_db", cls.gain_db)
        noise = _config_value(augmentation, "noise_snr_db", cls.noise_snr_db)
        pitch_speed = _config_value(
            augmentation,
            "pitch_speed_semitones",
            cls.pitch_speed_semitones,
        )

        return cls(
            sample_rate=int(_config_value(config, "sample_rate", cls.sample_rate)),
            duration_seconds=float(duration),
            n_mels=int(_config_value(config, "n_mels", cls.n_mels)),
            n_fft=int(_config_value(config, "n_fft", cls.n_fft)),
            win_length=int(_config_value(config, "win_length", cls.win_length)),
            hop_length=int(_config_value(config, "hop_length", cls.hop_length)),
            f_min=float(_config_value(config, "f_min", cls.f_min)),
            f_max=_optional_float(_config_value(config, "f_max", cls.f_max)),
            trim_top_db=float(_config_value(config, "trim_top_db", cls.trim_top_db)),
            trim_frame_length=int(
                _config_value(config, "trim_frame_length", cls.trim_frame_length)
            ),
            trim_hop_length=int(_config_value(config, "trim_hop_length", cls.trim_hop_length)),
            normalize_features=bool(
                _config_value(config, "normalize_features", cls.normalize_features)
            ),
            augmentation_enabled=bool(
                _config_value(augmentation, "enabled", cls.augmentation_enabled)
            ),
            gain_db=_symmetric_range(gain, cls.gain_db),
            noise_snr_db=_ordered_range(noise, cls.noise_snr_db),
            pitch_speed_semitones=_symmetric_range(
                pitch_speed,
                cls.pitch_speed_semitones,
            ),
            pitch_speed_probability=float(
                _config_value(
                    augmentation,
                    "pitch_speed_probability",
                    cls.pitch_speed_probability,
                )
            ),
            maximum_frequency_shift_bins=int(
                _config_value(
                    augmentation,
                    "maximum_frequency_shift_bins",
                    cls.maximum_frequency_shift_bins,
                )
            ),
            frequency_shift_probability=float(
                _config_value(
                    augmentation,
                    "frequency_shift_probability",
                    cls.frequency_shift_probability,
                )
            ),
            time_mask_frames=int(
                _config_value(augmentation, "time_mask_frames", cls.time_mask_frames)
            ),
            frequency_mask_bins=int(
                _config_value(
                    augmentation,
                    "frequency_mask_bins",
                    cls.frequency_mask_bins,
                )
            ),
        )


def _optional_float(value: Any) -> float | None:
    """Convert a YAML number to float while preserving an explicit ``null``."""

    return None if value is None else float(value)


def _decode_with_soundfile(path: Path) -> tuple[Tensor, int]:
    """Decode through libsndfile and convert [samples, channels] to torch order."""

    samples, source_rate = soundfile.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    # soundfile returns time-major NumPy arrays. The explicit copy produces
    # contiguous channel-major memory owned by the tensor, independent of the
    # decoder's temporary buffer.
    waveform = torch.from_numpy(samples.transpose(1, 0).copy())
    return waveform, int(source_rate)


def _decode_with_ffmpeg(path: Path) -> tuple[Tensor, int]:
    """Convert browser-oriented containers to a temporary WAV using bundled FFmpeg."""

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "imageio-ffmpeg is required to decode this browser-upload format"
        ) from error

    ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="qbh_decode_") as temporary_directory:
        converted_path = Path(temporary_directory) / "decoded.wav"
        command = [
            ffmpeg_executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-map_metadata",
            "-1",
            "-c:a",
            "pcm_f32le",
            str(converted_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "FFmpeg returned no diagnostic text"
            raise RuntimeError(f"FFmpeg conversion failed: {detail}")
        if not converted_path.is_file():
            raise RuntimeError("FFmpeg reported success but did not create decoded.wav")
        return _decode_with_soundfile(converted_path)


def _resolve_audio_file(path: Path) -> Path | None:
    """Find ``path``, including a narrowly scoped moved-project fallback.

    The original location is always preferred. If it is initially absent and
    contains adjacent ``data/raw`` path components, try the same suffix below
    the current project root immediately. Both locations then share the bounded
    retry window needed by streamed filesystems.
    """

    if path.is_file():
        return path

    relocated_path = _relocated_raw_audio_path(path)
    if relocated_path == path:
        relocated_path = None
    if relocated_path is not None and relocated_path.is_file():
        return relocated_path

    for delay_seconds in _FILE_VISIBILITY_RETRY_DELAYS_SECONDS:
        time.sleep(delay_seconds)
        if path.is_file():
            return path
        if relocated_path is not None and relocated_path.is_file():
            return relocated_path

    return None


def load_audio(
    path: str | Path,
    sample_rate: int = 16_000,
    *,
    mono: bool = True,
) -> Tensor:
    """Load an audio file as finite ``float32`` samples at one sample rate.

    Resampling here means every later component can assume 16 kHz.  Stereo is
    averaged rather than selecting the left channel, which is safer for music
    recordings whose lead melody may not be centered perfectly.
    """

    requested_path = Path(path).expanduser()
    audio_path = _resolve_audio_file(requested_path)
    if audio_path is None:
        raise FileNotFoundError(f"Audio file does not exist: {requested_path}")

    try:
        # WAV/FLAC and many MP3 builds decode here without relying on TorchCodec.
        waveform, source_rate = _decode_with_soundfile(audio_path)
    except (soundfile.SoundFileError, OSError, RuntimeError, ValueError) as soundfile_error:
        try:
            waveform, source_rate = _decode_with_ffmpeg(audio_path)
        except Exception as ffmpeg_error:
            raise RuntimeError(
                f"Could not decode audio file {audio_path}. "
                f"soundfile: {soundfile_error}; FFmpeg fallback: {ffmpeg_error}"
            ) from ffmpeg_error

    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError(f"Expected non-empty [channels, samples] audio: {audio_path}")

    waveform = waveform.to(dtype=torch.float32)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)

    # A corrupt decoder should not quietly inject NaNs into the model.
    return torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0).contiguous()


def trim_silence(
    waveform: Tensor,
    *,
    top_db: float = 35.0,
    frame_length: int = 1_024,
    hop_length: int = 256,
) -> Tensor:
    """Remove quiet leading/trailing frames with a small torch-only RMS detector.

    The threshold is relative to the loudest frame, so the function works for
    both soft and loud microphones.  Completely silent input is returned intact;
    downstream fixed-length padding can then handle it predictably.
    """

    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, samples], received {tuple(waveform.shape)}")
    if waveform.shape[-1] == 0:
        return waveform
    if top_db <= 0 or frame_length <= 0 or hop_length <= 0:
        raise ValueError("top_db, frame_length, and hop_length must be positive")

    # Average channel energy before framing.  Padding permits very short clips.
    mono_energy = waveform.square().mean(dim=0, keepdim=True).unsqueeze(0)
    required_padding = max(0, frame_length - mono_energy.shape[-1])
    if required_padding:
        mono_energy = functional.pad(mono_energy, (0, required_padding))

    frame_energy = functional.avg_pool1d(
        mono_energy,
        kernel_size=frame_length,
        stride=hop_length,
        ceil_mode=True,
    ).flatten()
    peak_energy = frame_energy.max()
    if peak_energy <= torch.finfo(frame_energy.dtype).eps:
        return waveform

    threshold = peak_energy * (10.0 ** (-top_db / 10.0))
    active_frames = torch.nonzero(frame_energy >= threshold, as_tuple=False).flatten()
    if active_frames.numel() == 0:
        return waveform

    # Include half a frame of context so initial consonants are not clipped.
    first = int(active_frames[0].item())
    last = int(active_frames[-1].item())
    start = max(0, first * hop_length - frame_length // 2)
    end = min(waveform.shape[-1], last * hop_length + frame_length + frame_length // 2)
    return waveform[..., start:end].contiguous()


def pad_or_crop(
    waveform: Tensor,
    target_samples: int,
    *,
    random_crop: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return exactly ``target_samples`` using center padding/cropping by default."""

    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, samples], received {tuple(waveform.shape)}")
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")

    current_samples = waveform.shape[-1]
    if current_samples == target_samples:
        return waveform
    if current_samples < target_samples:
        missing = target_samples - current_samples
        left = missing // 2
        return functional.pad(waveform, (left, missing - left))

    maximum_start = current_samples - target_samples
    if random_crop and maximum_start > 0:
        start = int(
            torch.randint(
                maximum_start + 1,
                size=(1,),
                generator=generator,
            ).item()
        )
    else:
        start = maximum_start // 2
    return waveform[..., start : start + target_samples].contiguous()


def _uniform(
    lower: float,
    upper: float,
    *,
    generator: torch.Generator | None,
) -> float:
    """Draw one CPU scalar while respecting a DataLoader-specific generator."""

    if lower == upper:
        return lower
    draw = torch.rand((), generator=generator).item()
    return lower + (upper - lower) * draw


def _augment_waveform(
    waveform: Tensor,
    settings: AudioSettings,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """Apply gain and white-noise perturbations without expensive resampling."""

    gain_db = _uniform(*settings.gain_db, generator=generator)
    augmented = waveform * (10.0 ** (gain_db / 20.0))

    signal_rms = augmented.square().mean().sqrt()
    if signal_rms > torch.finfo(augmented.dtype).eps:
        snr_db = _uniform(*settings.noise_snr_db, generator=generator)
        noise = torch.randn(
            augmented.shape,
            dtype=augmented.dtype,
            device=augmented.device,
            generator=generator,
        )
        noise_rms = noise.square().mean().sqrt().clamp_min(1e-12)
        desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        augmented = augmented + noise * (desired_noise_rms / noise_rms)

    # Audio decoders conventionally expect this range.  Clipping here also stops
    # an unusual metadata record from producing huge spectrogram values.
    return augmented.clamp(-1.0, 1.0)


def _augment_pitch_and_speed(
    waveform: Tensor,
    settings: AudioSettings,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """Resample once to create an inexpensive joint key/tempo perturbation.

    A positive semitone draw shortens the waveform and raises its pitch, like
    playing tape faster; a negative draw does the reverse.  Linear interpolation
    is deliberate here: this is training-only corruption rather than a fidelity
    conversion, and it is dramatically cheaper than rebuilding a band-limited
    resampler for every random factor on CPU.
    """

    probability = settings.pitch_speed_probability
    lower, upper = settings.pitch_speed_semitones
    if probability <= 0.0 or (lower == 0.0 and upper == 0.0):
        return waveform
    if torch.rand((), generator=generator).item() >= probability:
        return waveform

    semitones = _uniform(lower, upper, generator=generator)
    speed_factor = 2.0 ** (semitones / 12.0)
    output_samples = max(2, round(waveform.shape[-1] / speed_factor))
    if output_samples == waveform.shape[-1]:
        return waveform
    return functional.interpolate(
        waveform.unsqueeze(0),
        size=output_samples,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _mask_axis(
    features: Tensor,
    *,
    axis: int,
    maximum_width: int,
    generator: torch.Generator | None,
) -> Tensor:
    """Mask one random contiguous frequency or time band in-place on a clone."""

    axis_size = features.shape[axis]
    maximum_width = min(maximum_width, axis_size)
    if maximum_width <= 0:
        return features

    width = int(torch.randint(maximum_width + 1, (1,), generator=generator).item())
    if width == 0:
        return features
    start = int(torch.randint(axis_size - width + 1, (1,), generator=generator).item())

    masked = features.clone()
    slices = [slice(None)] * masked.ndim
    slices[axis] = slice(start, start + width)
    masked[tuple(slices)] = 0.0
    return masked


def _shift_frequency_axis(
    features: Tensor,
    settings: AudioSettings,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """Translate log-mel energy up or down without wrapping at the edges."""

    maximum = min(settings.maximum_frequency_shift_bins, features.shape[-2] - 1)
    probability = settings.frequency_shift_probability
    if maximum <= 0 or probability <= 0.0:
        return features
    if torch.rand((), generator=generator).item() >= probability:
        return features
    shift = int(torch.randint(-maximum, maximum + 1, (1,), generator=generator).item())
    if shift == 0:
        return features

    shifted = torch.roll(features, shifts=shift, dims=-2)
    shifted = shifted.clone()
    if shift > 0:
        shifted[..., :shift, :] = 0.0
    else:
        shifted[..., shift:, :] = 0.0
    return shifted


class LogMelExtractor(nn.Module):
    """Turn fixed-rate waveforms into normalized 80-bin log-mel tensors."""

    def __init__(self, settings: AudioSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or AudioSettings()
        f_max = self.settings.f_max or self.settings.sample_rate / 2
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.settings.sample_rate,
            n_fft=self.settings.n_fft,
            win_length=self.settings.win_length,
            hop_length=self.settings.hop_length,
            f_min=self.settings.f_min,
            f_max=f_max,
            n_mels=self.settings.n_mels,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            center=True,
        )
        self.to_decibels = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(
        self,
        waveform: Tensor,
        *,
        training: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Extract one fixed-size feature tensor shaped ``[1, mel, frames]``."""

        if waveform.ndim != 2:
            raise ValueError(
                f"Expected waveform [channels, samples], received {tuple(waveform.shape)}"
            )
        if waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        working = waveform
        if training and self.settings.augmentation_enabled:
            # Run this before the fixed crop.  Long source clips therefore keep
            # real neighboring audio after a speed-up instead of being padded.
            working = _augment_pitch_and_speed(
                working,
                self.settings,
                generator=generator,
            )

        fixed = pad_or_crop(
            working,
            self.settings.target_samples,
            random_crop=training,
            generator=generator,
        )
        if training and self.settings.augmentation_enabled:
            fixed = _augment_waveform(fixed, self.settings, generator=generator)

        features = self.to_decibels(self.mel_spectrogram(fixed))
        if self.settings.normalize_features:
            mean = features.mean(dim=(-2, -1), keepdim=True)
            standard_deviation = features.std(dim=(-2, -1), keepdim=True, unbiased=False)
            features = (features - mean) / standard_deviation.clamp_min(1e-5)

        if training and self.settings.augmentation_enabled:
            features = _shift_frequency_axis(
                features,
                self.settings,
                generator=generator,
            )
            features = _mask_axis(
                features,
                axis=-1,
                maximum_width=self.settings.time_mask_frames,
                generator=generator,
            )
            features = _mask_axis(
                features,
                axis=-2,
                maximum_width=self.settings.frequency_mask_bins,
                generator=generator,
            )

        expected_shape = (1, self.settings.n_mels)
        if tuple(features.shape[:2]) != expected_shape:
            raise RuntimeError(
                f"Unexpected log-mel shape {tuple(features.shape)}; expected [1, "
                f"{self.settings.n_mels}, frames]"
            )
        return features.contiguous()


def extract_log_mel(
    path: str | Path,
    settings: AudioSettings | None = None,
    *,
    is_humming: bool = False,
    training: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Convenience wrapper for loading, optional hum trimming, and extraction."""

    resolved_settings = settings or AudioSettings()
    waveform = load_audio(path, resolved_settings.sample_rate, mono=True)
    if is_humming:
        waveform = trim_silence(
            waveform,
            top_db=resolved_settings.trim_top_db,
            frame_length=resolved_settings.trim_frame_length,
            hop_length=resolved_settings.trim_hop_length,
        )
    extractor = LogMelExtractor(resolved_settings)
    return extractor(waveform, training=training, generator=generator)


__all__ = [
    "AudioSettings",
    "LogMelExtractor",
    "extract_log_mel",
    "load_audio",
    "pad_or_crop",
    "trim_silence",
]
