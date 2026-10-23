"""XTTS-v2 TTS engine with native voice cloning.

Reference: Coqui XTTS-v2, CPML license (non-commercial).
Supports cross-lingual cloning from a ~6-second reference clip.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from aoc_magic_v3.errors import EngineUnavailable, TTSError
from aoc_magic_v3.tts.base import TTSEngine, TTSResult

logger = logging.getLogger(__name__)

_XTTS_AVAILABLE = False
try:
    import torch  # type: ignore  # noqa: F401
    import torchaudio  # type: ignore  # noqa: F401
    from TTS.api import TTS as CoquiTTS  # type: ignore
    _XTTS_AVAILABLE = True
except ImportError:
    CoquiTTS = None  # type: ignore

_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_NATIVE_SR = 24000
_CPML_WARNED = False


class XTTSEngine(TTSEngine):
    name = "xtts"

    _instance = None
    _device = None

    def is_available(self) -> bool:
        return bool(_XTTS_AVAILABLE)

    def supports_cloning(self) -> bool:
        return True

    def synthesize(
        self,
        text: str,
        *,
        rate: int,
        volume: float,
        reference_paths: Optional[List[Path]] = None,
        device: str = "cpu",
    ) -> TTSResult:
        global _CPML_WARNED
        if not self.is_available():
            raise EngineUnavailable("XTTS-v2 dependencies not installed.")

        if not reference_paths:
            raise TTSError("XTTS requires at least one reference clip.")
        for p in reference_paths:
            if not Path(p).is_file():
                raise TTSError(f"XTTS reference not found: {p}")

        text = text.strip()
        if not text:
            raise TTSError("XTTS received empty text.")

        if not _CPML_WARNED:
            logger.warning(
                "[XTTS] Weights are under the Coqui Public Model License "
                "(CPML) — non-commercial use only."
            )
            _CPML_WARNED = True

        speed = max(0.5, min(2.0, float(rate) / 260.0))
        volume = max(0.0, min(1.0, float(volume)))

        try:
            if XTTSEngine._instance is None or XTTSEngine._device != device:
                logger.info("[XTTS] loading model on %s", device)
                XTTSEngine._instance = CoquiTTS(_MODEL_NAME).to(device)
                XTTSEngine._device = device

            wav = XTTSEngine._instance.tts(
                text=text,
                speaker_wav=[str(p) for p in reference_paths],
                language="en",
                speed=speed,
            )
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            if volume < 1.0:
                audio = audio * volume
            if audio.size == 0:
                raise TTSError("XTTS returned empty audio.")

            logger.info("[XTTS] synthesized %d samples", audio.size)
            return TTSResult(audio=audio, sample_rate=_NATIVE_SR, engine="xtts")

        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"XTTS synthesis failed: {exc}") from exc
