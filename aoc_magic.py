#!/usr/bin/env python3
"""Thin V2-compatible shim. All logic lives in aoc_magic_v3.

Patched for AOC v4 (4.0.0):
  - finalize_audio keeps V2 file-based signature.
  - convert_voice keeps V2 signature and adapts to KnnVC.convert.
  - XTTS, F5, and MKL are real engines in the integrated ai_voice.py.
  - DAW bus integration documented in DAW_INTEGRATION.md.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from aoc_magic_v3.cli import main
from aoc_magic_v3 import (
    Config,
    build_parser,
    compressor,
    db_to_linear,
    ensure_parent,
    highpass,
    limiter,
    linear_to_db,
    load_mono,
    lowpass,
    peak_normalize,
    pick_device,
    pitch_shift,
    process_character_dsp,
    require_file,
    run_pipeline,
    soft_saturate,
    synthesize_speech,
    time_stretch,
)


def _v3_finalize(audio, sr, out_path, ceiling_db=-1.0):
    from aoc_magic_v3.finalize import finalize_audio as _finalize
    return _finalize(audio, sr, Path(out_path), ceiling_db=ceiling_db)


def finalize_audio(input_path, output_path, sample_rate=44100, ceiling_db=-1.0):
    """V2 signature: (input_path, output_path, sample_rate, ceiling_db)."""
    import numpy as np
    import librosa
    y, sr = librosa.load(str(input_path), sr=sample_rate, mono=True)
    y = np.asarray(y, dtype=np.float32)
    return _v3_finalize(y, sr, output_path, ceiling_db=ceiling_db)


def load_knn_vc(*args, **kwargs):
    warnings.warn(
        "load_knn_vc is deprecated; use aoc_magic_v3.vc.knn_vc",
        DeprecationWarning,
        stacklevel=2,
    )
    from aoc_magic_v3.vc.knn_vc import _load_knn_vc
    return _load_knn_vc(*args, **kwargs)


def convert_voice(knn_vc, source_path, reference_paths, output_path, topk=4):
    """V2 signature: (knn_vc, source_path, reference_paths, output_path, topk)."""
    import numpy as np
    import librosa
    import soundfile as sf
    from aoc_magic_v3.vc.knn_vc import KnnVC

    y, sr = librosa.load(str(source_path), sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    refs = [Path(p) for p in reference_paths]
    out, out_sr = KnnVC().convert(y, sr, refs, topk=topk, device="cpu")
    if out.ndim == 1:
        out = out.reshape(-1)
    sf.write(str(output_path), out.astype(np.float32), out_sr, subtype="PCM_16")
    return Path(output_path)


if __name__ == "__main__":
    sys.exit(main())
