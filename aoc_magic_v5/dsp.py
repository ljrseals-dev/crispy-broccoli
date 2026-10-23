"""Frozen DSP chain from AOC Magic V2 — copied verbatim.

Do not refactor math. Do not change parameter defaults.
Do not change the order of operations in process_character_dsp.

Order:
  highpass(70 Hz)
  → pitch_shift(cfg.pitch)
  → time_stretch(cfg.speed)
  → compressor(cfg.comp_threshold, cfg.comp_ratio)
  → soft_saturate(cfg.drive_db)
  → lowpass(12 kHz)
"""

from __future__ import annotations

import logging

import numpy as np

from aoc_magic_v3.errors import DspError

logger = logging.getLogger(__name__)


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def peak_normalize(y: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 0.0:
        return y.astype(np.float32, copy=False)

    target = db_to_linear(ceiling_db)
    return (y * (target / peak)).astype(np.float32)


def highpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return y

    try:
        from scipy.signal import butter, sosfilt

        sos = butter(
            2,
            cutoff_hz,
            btype="highpass",
            fs=sr,
            output="sos",
        )
        return sosfilt(sos, y).astype(np.float32)
    except ImportError:
        return y


def lowpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return y

    try:
        from scipy.signal import butter, sosfilt

        sos = butter(
            2,
            cutoff_hz,
            btype="lowpass",
            fs=sr,
            output="sos",
        )
        return sosfilt(sos, y).astype(np.float32)
    except ImportError:
        return y


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-6:
        return y
    import librosa
    return librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=float(semitones),
    ).astype(np.float32)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """
    rate > 1.0 = faster
    rate < 1.0 = slower
    """
    if rate <= 0:
        raise ValueError("Speed must be greater than zero.")

    if abs(rate - 1.0) < 1e-6:
        return y

    import librosa
    return librosa.effects.time_stretch(
        y=y,
        rate=float(rate),
    ).astype(np.float32)


def compressor(
    y: np.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
) -> np.ndarray:
    """
    Lightweight sample-domain compressor.

    This avoids pydub's integer/float PCM ambiguity and keeps the pipeline
    entirely NumPy-based. Ratios below 1.0 would expand rather than compress,
    so they are clamped to 1.0.
    """
    ratio = max(float(ratio), 1.0)

    x = y.astype(np.float32, copy=True)

    threshold = db_to_linear(threshold_db)

    mag = np.abs(x)
    over = mag > threshold

    if np.any(over):
        compressed_mag = threshold + (
            (mag[over] - threshold) / ratio
        )
        x[over] = np.sign(x[over]) * compressed_mag

    return x


def soft_saturate(y: np.ndarray, drive_db: float) -> np.ndarray:
    if abs(drive_db) < 1e-6:
        return y

    gain = db_to_linear(drive_db)
    return np.tanh(y * gain).astype(np.float32)


def limiter(y: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """
    Soft-knee safety limiter.

    Values well below the ceiling pass through essentially unchanged;
    values above it are smoothly folded back toward the ceiling. This is
    applied before the final peak-normalize so no sample can overshoot.
    """
    ceiling = db_to_linear(ceiling_db)
    ceiling = max(ceiling, 1e-6)
    return (np.tanh(y / ceiling) * ceiling).astype(np.float32)


def process_character_dsp(
    audio: np.ndarray,
    sr: int,
    *,
    pitch: float = 2.0,
    speed: float = 1.15,
    comp_threshold: float = -20.0,
    comp_ratio: float = 4.0,
    drive_db: float = 1.5,
    highpass_hz: float = 70.0,
    lowpass_hz: float = 12000.0,
) -> np.ndarray:
    """Apply the frozen V2 character DSP chain to an in-memory array.

    highpass → pitch_shift → time_stretch → compressor → soft_saturate → lowpass
    """
    try:
        y = np.asarray(audio, dtype=np.float32)
        y = highpass(y, sr, highpass_hz)
        y = pitch_shift(y, sr, pitch)
        y = time_stretch(y, speed)
        y = compressor(
            y,
            threshold_db=comp_threshold,
            ratio=comp_ratio,
        )
        y = soft_saturate(y, drive_db)
        y = lowpass(y, sr, lowpass_hz)
        return y.astype(np.float32)
    except Exception as exc:
        raise DspError(f"process_character_dsp failed: {exc}") from exc
