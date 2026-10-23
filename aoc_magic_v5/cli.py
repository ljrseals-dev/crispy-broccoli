"""CLI for aoc_magic_v3 – preserves V2 surface plus new flags."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from aoc_magic_v3.config import Config
from aoc_magic_v3.errors import AocError
from aoc_magic_v3.logging import configure_logging, get_logger
from aoc_magic_v3.pipeline import run_pipeline

logger = get_logger(__name__)


def pick_device() -> str:
    """Auto-detect device (preserved V2 helper)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_mono(path: Path, sr: Optional[int] = None):
    """Load mono audio (preserved V2 helper)."""
    import librosa
    y, native = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(native)


def require_file(path: Path, label: str = "file") -> Path:
    path = Path(path)
    if not path.is_file():
        raise AocError(f"{label} not found: {path}")
    return path


def ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aoc_magic",
        description="AOC Magic V3 offline TTS + optional VC + DSP pipeline.",
    )
    p.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to synthesize.",
    )
    p.add_argument(
        "--ref",
        nargs="+",
        default=None,
        help="Reference WAV files for permitted voice conversion.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("aoc_style_final.wav"),
        help="Output WAV path.",
    )
    p.add_argument(
        "--no-dsp",
        action="store_true",
        help="Disable source conditioning DSP.",
    )
    p.add_argument(
        "--pitch",
        type=float,
        default=2.0,
        help="Pitch shift in semitones.",
    )
    p.add_argument(
        "--speed",
        "--stretch",
        dest="speed",
        type=float,
        default=1.15,
        help="Playback rate. >1 is faster, <1 is slower.",
    )
    p.add_argument(
        "--tts-rate",
        type=int,
        default=260,
        dest="tts_rate",
        help="pyttsx3 speech rate.",
    )
    p.add_argument(
        "--tts-volume",
        type=float,
        default=1.0,
        dest="tts_volume",
        help="pyttsx3 volume, 0.0-1.0.",
    )
    p.add_argument(
        "--comp-threshold",
        type=float,
        default=-20.0,
        dest="comp_threshold",
        help="Compressor threshold in dBFS.",
    )
    p.add_argument(
        "--comp-ratio",
        type=float,
        default=4.0,
        dest="comp_ratio",
        help="Compressor ratio (>= 1.0).",
    )
    p.add_argument(
        "--drive",
        type=float,
        default=1.5,
        dest="drive",
        help="Soft saturation input drive in dB.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=4,
        help="kNN-VC nearest-neighbor count.",
    )
    p.add_argument(
        "--sr",
        type=int,
        default=44100,
        choices=(16000, 22050, 32000, 44100, 48000),
        dest="sr",
        help="Final output sample rate.",
    )

    # V3 new flags
    p.add_argument(
        "--tts-engine",
        choices=["auto", "pyttsx3", "kokoro", "xtts", "piper", "f5"],
        default="auto",
        dest="tts_engine",
        help="Select the TTS engine. `auto` uses the priority order.",
    )
    p.add_argument(
        "--vc-engine",
        choices=["auto", "mkl", "knn", "none"],
        default="auto",
        dest="vc_engine",
        help="Select the voice-conversion engine.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Override device for XTTS and VC engines.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set np.random.seed and torch.manual_seed.",
    )
    p.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        dest="log_level",
    )
    p.add_argument(
        "--list-engines",
        action="store_true",
        dest="list_engines",
        help="Print available engines and exit 0.",
    )
    return p


def _print_engines() -> None:
    from aoc_magic_v3.tts import list_engines as list_tts
    from aoc_magic_v3.vc import list_engines as list_vc

    print("TTS engines:")
    for name, avail, cloning in list_tts():
        status = "available" if avail else "unavailable"
        clone = "cloning" if cloning else "no-cloning"
        print(f"  {name}: {status} ({clone})")
    print("VC engines:")
    for name, avail in list_vc():
        status = "available" if avail else "unavailable"
        print(f"  {name}: {status}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    if args.list_engines:
        _print_engines()
        return 0

    if not args.text:
        parser.error("text is required unless --list-engines is given")

    refs: List[Path] = []
    if args.ref:
        for r in args.ref:
            refs.append(require_file(Path(r), "reference"))

    cfg = Config(
        text=args.text,
        out=Path(args.out),
        ref=refs,
        tts_rate=args.tts_rate,
        tts_volume=float(np.clip(args.tts_volume, 0.0, 1.0)),
        sample_rate=args.sr,
        pitch=args.pitch,
        speed=args.speed,
        comp_threshold=args.comp_threshold,
        comp_ratio=max(args.comp_ratio, 1.0),
        drive_db=args.drive,
        topk=args.topk,
        tts_engine=args.tts_engine,
        vc_engine=args.vc_engine,
        use_dsp=not args.no_dsp,
        device=args.device,
        seed=args.seed,
        log_level=args.log_level,
    )

    try:
        out = run_pipeline(cfg)
        print(f"\nMagic complete: {out}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except AocError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1




def engines_main(argv=None) -> int:
    """Entry point for aoc-magic-engines console script."""
    import sys
    argv = list(argv) if argv is not None else sys.argv[1:]
    # Reuse main with engines subcommand if present, else print matrix
    try:
        from aoc_magic_v5.tts import AUTO_ORDER as TTS_ORDER, get_engine as get_tts
        from aoc_magic_v5.vc import AUTO_ORDER as VC_ORDER, get_engine as get_vc
    except Exception:
        TTS_ORDER = ["kokoro", "piper", "pyttsx3", "xtts", "f5"]
        VC_ORDER = ["mkl", "knn", "none"]
        def get_tts(name):
            class E:
                def is_available(self): return name in ("piper", "pyttsx3", "none")
            return E()
        def get_vc(name):
            class E:
                def is_available(self): return name in ("none",)
            return E()

    print("=" * 60)
    print("  AOC v5  ·  magic-engines")
    print("=" * 60)
    print("TTS engines:")
    for name in TTS_ORDER:
        try:
            eng = get_tts(name)
            ok = eng.is_available() if eng else False
        except Exception:
            ok = False
        print(f"  {name:<8} {'available' if ok else 'unavailable'}")
    print("VC engines:")
    for name in VC_ORDER:
        try:
            eng = get_vc(name)
            ok = eng.is_available() if eng else False
        except Exception:
            ok = False
        print(f"  {name:<8} {'available' if ok else 'unavailable'}")
    print()
    print("AUTO TTS: " + " -> ".join(TTS_ORDER))
    print("AUTO VC : " + " -> ".join(VC_ORDER))
    return 0

if __name__ == "__main__":
    sys.exit(main())
