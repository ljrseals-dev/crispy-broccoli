"""Kokoro-82M TTS engine.

Uses the ONNX Runtime path (soniqo/Kokoro-82M-ONNX) for bit-for-bit
output parity with the PyTorch reference, without a torch dependency.

Reference: hexgrad/Kokoro-82M (StyleTTS2 + iSTFTNet), Apache-2.0.
ONNX mirror: soniqo/Kokoro-82M-ONNX.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from aoc_magic_v3.errors import EngineUnavailable, TTSError
from aoc_magic_v3.tts.base import TTSEngine, TTSResult

logger = logging.getLogger(__name__)

_KOKORO_AVAILABLE = False
try:
    import kokoro_onnx  # type: ignore  # noqa: F401
    import onnxruntime  # type: ignore  # noqa: F401
    _KOKORO_AVAILABLE = True
except ImportError:
    kokoro_onnx = None  # type: ignore

_DEFAULT_VOICE = "af_bella"
_DEFAULT_LANG = "en-us"

_CACHE_DIR = Path.home() / ".cache" / "aoc_magic" / "kokoro"
_MODEL_PATH = _CACHE_DIR / "kokoro-v1.0.onnx"
_VOICES_PATH = _CACHE_DIR / "voices-v1.0.bin"


class KokoroEngine(TTSEngine):
    name = "kokoro"

    _instance = None

    def is_available(self) -> bool:
        if not _KOKORO_AVAILABLE:
            return False
        return _MODEL_PATH.is_file() and _VOICES_PATH.is_file()

    def supports_cloning(self) -> bool:
        return False

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
            raise EngineUnavailable(
                f"Kokoro model files not found at {_CACHE_DIR}. "
                "Download kokoro-v1.0.onnx and voices-v1.0.bin from "
                "https://huggingface.co/soniqo/Kokoro-82M-ONNX and place "
                "them in that directory."
            )

        text = text.strip()
        if not text:
            raise TTSError("Kokoro received empty text.")

        speed = max(0.5, min(2.0, float(rate) / 260.0))
        volume = max(0.0, min(1.0, float(volume)))

        try:
            if KokoroEngine._instance is None:
                logger.info("[Kokoro] loading model from %s", _CACHE_DIR)
                KokoroEngine._instance = kokoro_onnx.Kokoro(
                    str(_MODEL_PATH), str(_VOICES_PATH),
                )

            samples, sr = KokoroEngine._instance.create(
                text,
                voice=_DEFAULT_VOICE,
                speed=speed,
                lang=_DEFAULT_LANG,
            )
            audio = np.asarray(samples, dtype=np.float32).reshape(-1)
            if volume < 1.0:
                audio = audio * volume
            if audio.size == 0:
                raise TTSError("Kokoro returned empty audio.")

            logger.info(
                "[Kokoro] synthesized %d samples @ %d Hz", audio.size, int(sr),
            )
            return TTSResult(audio=audio, sample_rate=int(sr), engine="kokoro")

        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Kokoro synthesis failed: {exc}") from exc
