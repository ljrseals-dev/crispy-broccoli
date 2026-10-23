"""F5-TTS engine — flow-matching zero-shot voice cloning.

Reference: F5-TTS (DiT + flow matching). Requires f5-tts package and
~1.2 GB model. GPU strongly recommended; CPU is supported but slow.

Not part of the `auto` order. Opt in with --tts-engine f5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from aoc_magic_v3.errors import EngineUnavailable, TTSError
from aoc_magic_v3.tts.base import TTSEngine, TTSResult

logger = logging.getLogger(__name__)

_F5_AVAILABLE = False
try:
    import torch  # type: ignore  # noqa: F401
    from f5_tts.api import F5TTS  # type: ignore
    _F5_AVAILABLE = True
except ImportError:
    F5TTS = None  # type: ignore

_NATIVE_SR = 24000
_MODEL = "F5TTS_v1_Base"


class F5TTSEngine(TTSEngine):
    name = "f5"

    _instance = None
    _device = None

    def is_available(self) -> bool:
        return bool(_F5_AVAILABLE)

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
        if not self.is_available():
            raise EngineUnavailable("F5-TTS not installed.")

        if not reference_paths:
            raise TTSError("F5-TTS requires at least one reference clip.")
        for p in reference_paths:
            if not Path(p).is_file():
                raise TTSError(f"F5-TTS reference not found: {p}")

        text = text.strip()
        if not text:
            raise TTSError("F5-TTS received empty text.")

        try:
            if F5TTSEngine._instance is None or F5TTSEngine._device != device:
                logger.info("[F5-TTS] loading model on %s", device)
                F5TTSEngine._instance = F5TTS(
                    model=_MODEL, device=device,
                )
                F5TTSEngine._device = device

            wav, sr, _ = F5TTSEngine._instance.infer(
                ref_file=str(reference_paths[0]),
                ref_text="",
                gen_text=text,
                speed=max(0.5, min(2.0, float(rate) / 260.0)),
            )
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            volume = max(0.0, min(1.0, float(volume)))
            if volume < 1.0:
                audio = audio * volume
            if audio.size == 0:
                raise TTSError("F5-TTS returned empty audio.")

            return TTSResult(
                audio=audio,
                sample_rate=int(sr or _NATIVE_SR),
                engine="f5",
            )

        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"F5-TTS synthesis failed: {exc}") from exc
