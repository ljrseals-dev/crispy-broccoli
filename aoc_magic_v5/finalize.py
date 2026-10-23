"""Finalization stage: limiter → peak_normalize → write PCM_16."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from aoc_magic_v3.dsp import limiter, peak_normalize
from aoc_magic_v3.errors import AocError

logger = logging.getLogger(__name__)


def finalize_audio(
    audio: np.ndarray,
    sr: int,
    out_path: Path,
    ceiling_db: float = -1.0,
) -> Path:
    """Apply soft-knee limiter + peak normalize and write PCM_16 WAV."""
    try:
        y = np.asarray(audio, dtype=np.float32)
        y = limiter(y, ceiling_db)
        y = peak_normalize(y, ceiling_db)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), y, sr, subtype="PCM_16")
        logger.info("Wrote %s (%.2fs @ %d Hz)", out_path, len(y) / max(sr, 1), sr)
        return out_path
    except Exception as exc:
        raise AocError(f"finalize_audio failed: {exc}") from exc
