#!/usr/bin/env python3
"""
AOC Magic Voice Pipeline v2
===========================

Offline text-to-speech -> optional kNN-VC timbre conversion -> character DSP.

Important:
- Use voice references only when you have permission to use them.
- pyttsx3 depends on an installed local OS speech engine.
- kNN-VC model/repository files may be downloaded by torch.hub on first use;
  subsequent execution can use the cached copy.
- The program always has a DSP-only fallback when VC is unavailable.
- This is not cloud TTS. After the first torch.hub download it does not
  require an internet connection to run.

Install (required):
    pip install numpy soundfile librosa pyttsx3

Install (optional, enables kNN-VC voice conversion):
    pip install torch torchaudio

Install (optional, enables highpass/lowpass filter stages):
    pip install scipy

Usage:
    python aoc_magic.py "Your text here" --out result.wav
    python aoc_magic.py "Your text here" --ref ref1.wav ref2.wav --out result.wav

Examples:
    python aoc_magic.py "Bold action starts now." --pitch 2 --speed 1.15
    python aoc_magic.py "Your text here" --ref my_voice.wav --topk 4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import torch
except ImportError:
    torch = None

try:
    import torchaudio
except ImportError:
    torchaudio = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    tts_rate: int = 260
    tts_volume: float = 1.0

    sample_rate: int = 44100
    pitch: float = 2.0
    speed: float = 1.15

    comp_threshold: float = -20.0
    comp_ratio: float = 4.0

    drive_db: float = 1.5
    highpass_hz: float = 70.0
    lowpass_hz: float = 12000.0

    limiter_ceiling_db: float = -1.0

    topk: int = 4


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def require_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    return p


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


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


def load_mono(path: str | Path, sr: int) -> tuple[np.ndarray, int]:
    y, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), actual_sr


def pick_device() -> str:
    """
    Return the best available torch device string.

    Falls back to CPU if torch is missing or the CUDA probe raises, which
    can happen on some ROCm / MPS / headless builds.
    """
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def synthesize_speech(
    text: str,
    output_file: str | Path,
    rate: int = 260,
    volume: float = 1.0,
) -> Path:
    if pyttsx3 is None:
        raise RuntimeError(
            "pyttsx3 is not installed. Install it with: pip install pyttsx3"
        )

    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    rate = int(np.clip(rate, 50, 500))
    volume = float(np.clip(volume, 0.0, 1.0))

    output_file = ensure_parent(output_file)

    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        engine.save_to_file(text, str(output_file))
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass

    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError(
            "TTS did not produce an audio file. "
            "Check that a local pyttsx3 speech engine is installed."
        )

    print(f"[TTS] {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# DSP
# ---------------------------------------------------------------------------

def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-6:
        return y
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
        # SciPy is optional. If unavailable, leave signal unchanged.
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
    input_path: str | Path,
    output_path: str | Path,
    cfg: Config,
) -> Path:
    print(
        "[DSP] "
        f"pitch={cfg.pitch:+.2f} st, "
        f"speed={cfg.speed:.3f}x, "
        f"compression={cfg.comp_ratio:.2f}:1"
    )

    y, sr = load_mono(input_path, cfg.sample_rate)

    y = highpass(y, sr, cfg.highpass_hz)
    y = pitch_shift(y, sr, cfg.pitch)
    y = time_stretch(y, cfg.speed)
    y = compressor(
        y,
        threshold_db=cfg.comp_threshold,
        ratio=cfg.comp_ratio,
    )
    y = soft_saturate(y, cfg.drive_db)
    y = lowpass(y, sr, cfg.lowpass_hz)

    output_path = ensure_parent(output_path)
    sf.write(str(output_path), y, sr, subtype="PCM_16")

    print(f"[DSP] {output_path}")
    return Path(output_path)


# ---------------------------------------------------------------------------
# kNN-VC
# ---------------------------------------------------------------------------

def load_knn_vc(device: str):
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install torch and torchaudio "
            "to enable kNN-VC."
        )

    print(f"[VC] Loading kNN-VC on {device}...")

    model = torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        prematched=True,
        trust_repo=True,
        pretrained=True,
        device=device,
    )

    return model


def convert_voice(
    knn_vc,
    source_path: str | Path,
    reference_paths: list[str | Path],
    output_path: str | Path,
    topk: int = 4,
) -> Path:
    if torchaudio is None:
        raise RuntimeError(
            "torchaudio is not installed. Install it to enable kNN-VC."
        )

    source_path = require_file(source_path)
    refs = [require_file(p) for p in reference_paths]

    if not refs:
        raise ValueError("At least one reference audio file is required.")

    if topk < 1:
        raise ValueError("topk must be >= 1.")

    print(f"[VC] Source: {source_path}")
    print(f"[VC] References: {len(refs)}")
    print(f"[VC] top-k: {topk}")

    query_seq = knn_vc.get_features(str(source_path))
    matching_set = knn_vc.get_matching_set([str(p) for p in refs])

    out_wav = knn_vc.match(
        query_seq,
        matching_set,
        topk=topk,
    )

    if out_wav.ndim == 1:
        out_wav = out_wav.unsqueeze(0)

    output_path = ensure_parent(output_path)
    torchaudio.save(
        str(output_path),
        out_wav.detach().cpu(),
        16000,
    )

    print(f"[VC] {output_path}")
    return Path(output_path)


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------

def finalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 44100,
    ceiling_db: float = -1.0,
) -> Path:
    y, sr = librosa.load(
        str(input_path),
        sr=sample_rate,
        mono=True,
    )

    y = limiter(y, ceiling_db)
    y = peak_normalize(y, ceiling_db)

    output_path = ensure_parent(output_path)

    sf.write(
        str(output_path),
        y,
        sr,
        subtype="PCM_16",
    )

    return Path(output_path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    text: str,
    output_path: str,
    references: list[str] | None,
    cfg: Config,
    use_dsp: bool = True,
) -> Path:

    with tempfile.TemporaryDirectory(prefix="aoc_magic_") as temp_dir:
        temp_dir = Path(temp_dir)

        tts_file = temp_dir / "tts.wav"

        print("\n=== AOC MAGIC V2 ===")
        print("[1/4] Synthesizing speech...")

        synthesize_speech(
            text,
            tts_file,
            rate=cfg.tts_rate,
            volume=cfg.tts_volume,
        )

        current = tts_file

        # ---------------------------------------------------------------
        # Source conditioning
        # ---------------------------------------------------------------
        if use_dsp:
            conditioned = temp_dir / "conditioned.wav"

            print("[2/4] Conditioning source audio...")

            process_character_dsp(
                current,
                conditioned,
                cfg,
            )

            current = conditioned
        else:
            print("[2/4] DSP conditioning disabled.")

        # ---------------------------------------------------------------
        # Optional voice conversion
        # ---------------------------------------------------------------
        if references:
            print("[3/4] Voice conversion...")

            device = pick_device()

            try:
                model = load_knn_vc(device)

                converted = temp_dir / "converted.wav"

                convert_voice(
                    model,
                    current,
                    references,
                    converted,
                    topk=cfg.topk,
                )

                current = converted

                print("[VC] Voice conversion succeeded.")

            except Exception as exc:
                print(
                    "[VC] WARNING: voice conversion failed.\n"
                    f"      {exc}\n"
                    "      Falling back to conditioned audio."
                )
        else:
            print("[3/4] No references supplied; skipping VC.")

        # ---------------------------------------------------------------
        # Final output
        # ---------------------------------------------------------------
        print("[4/4] Finalizing output...")

        finalize_audio(
            current,
            output_path,
            sample_rate=cfg.sample_rate,
            ceiling_db=cfg.limiter_ceiling_db,
        )

        print(f"\nMagic complete: {output_path}")

        return Path(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AOC Magic V2 offline TTS + optional kNN-VC + DSP pipeline."
    )

    parser.add_argument(
        "text",
        help="Text to synthesize.",
    )

    parser.add_argument(
        "--ref",
        nargs="+",
        help="Reference WAV files for permitted voice conversion.",
    )

    parser.add_argument(
        "--out",
        default="aoc_style_final.wav",
        help="Output WAV path.",
    )

    parser.add_argument(
        "--no-dsp",
        action="store_true",
        help="Disable source conditioning DSP.",
    )

    parser.add_argument(
        "--pitch",
        type=float,
        default=2.0,
        help="Pitch shift in semitones.",
    )

    parser.add_argument(
        "--speed",
        "--stretch",
        dest="speed",
        type=float,
        default=1.15,
        help="Playback rate. >1 is faster, <1 is slower.",
    )

    parser.add_argument(
        "--tts-rate",
        type=int,
        default=260,
        help="pyttsx3 speech rate.",
    )

    parser.add_argument(
        "--tts-volume",
        type=float,
        default=1.0,
        help="pyttsx3 volume, 0.0-1.0.",
    )

    parser.add_argument(
        "--comp-threshold",
        type=float,
        default=-20.0,
        help="Compressor threshold in dBFS.",
    )

    parser.add_argument(
        "--comp-ratio",
        type=float,
        default=4.0,
        help="Compressor ratio (>= 1.0).",
    )

    parser.add_argument(
        "--drive",
        type=float,
        default=1.5,
        help="Soft saturation input drive in dB.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="kNN-VC nearest-neighbor count.",
    )

    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        choices=(16000, 22050, 32000, 44100, 48000),
        help="Final output sample rate.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        output = Path(args.out)

        cfg = Config(
            tts_rate=args.tts_rate,
            tts_volume=float(np.clip(args.tts_volume, 0.0, 1.0)),
            sample_rate=args.sr,
            pitch=args.pitch,
            speed=args.speed,
            comp_threshold=args.comp_threshold,
            comp_ratio=max(args.comp_ratio, 1.0),
            drive_db=args.drive,
            topk=args.topk,
        )

        run_pipeline(
            text=args.text,
            output_path=str(output),
            references=args.ref,
            cfg=cfg,
            use_dsp=not args.no_dsp,
        )

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
