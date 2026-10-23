"""Piper TTS engine.

VITS-based, non-autoregressive, CPU-only. MIT-licensed engine.
Voices are downloaded separately and stored under ~/.cache/aoc_magic/piper/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from aoc_magic_v3.errors import EngineUnavailable, TTSError
from aoc_magic_v3.tts.base import TTSEngine, TTSResult

logger = logging.getLogger(__name__)

_PIPER_AVAILABLE = False
try:
    import piper  # type: ignore  # noqa: F401
    import onnxruntime  # type: ignore  # noqa: F401
    _PIPER_AVAILABLE = True
except ImportError:
    piper = None  # type: ignore

_CACHE_DIR = Path.home() / ".cache" / "aoc_magic" / "piper"


class PiperEngine(TTSEngine):
    name = "piper"

    _voice = None
    _voice_path = None

    def is_available(self) -> bool:
        if not _PIPER_AVAILABLE:
            return False
        return _CACHE_DIR.is_dir() and any(_CACHE_DIR.glob("*.onnx"))

    def supports_cloning(self) -> bool:
        return False

    @staticmethod
    def _select_voice() -> Path:
        candidates = sorted(_CACHE_DIR.glob("*.onnx"))
        if not candidates:
            raise EngineUnavailable(
                f"No Piper voice found under {_CACHE_DIR}. "
                "Download a voice from https://huggingface.co/rhasspy/piper-voices "
                "and place the .onnx and .onnx.json files there."
            )
        return candidates[0]

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
            raise EngineUnavailable("Piper voice or runtime not available.")

        text = text.strip()
        if not text:
            raise TTSError("Piper received empty text.")

        volume = max(0.0, min(1.0, float(volume)))

        try:
            voice_path = self._select_voice()
            if PiperEngine._voice is None or PiperEngine._voice_path != voice_path:
                PiperEngine._voice = piper.PiperVoice.load(str(voice_path))
                PiperEngine._voice_path = voice_path
                logger.info("[Piper] loaded voice %s", voice_path.name)

            chunks = []
            native_sr = None
            for chunk in PiperEngine._voice.synthesize(text):
                if hasattr(chunk, "audio_int16_bytes"):
                    arr = np.frombuffer(
                        chunk.audio_int16_bytes, dtype=np.int16,
                    ).astype(np.float32) / 32768.0
                    native_sr = getattr(chunk, "sample_rate", None)
                elif hasattr(chunk, "audio_float_array"):
                    arr = np.asarray(chunk.audio_float_array, dtype=np.float32)
                    native_sr = getattr(chunk, "sample_rate", None)
                else:
                    arr = np.asarray(chunk, dtype=np.float32)
                chunks.append(arr)

            if not chunks:
                raise TTSError("Piper produced no audio chunks.")

            audio = np.concatenate(chunks).astype(np.float32)
            if native_sr is None:
                native_sr = 22050
            if volume < 1.0:
                audio = audio * volume

            logger.info(
                "[Piper] synthesized %d samples @ %d Hz",
                audio.size, int(native_sr),
            )
            return TTSResult(audio=audio, sample_rate=int(native_sr), engine="piper")

        except (EngineUnavailable, TTSError):
            raise
        except Exception as exc:
            raise TTSError(f"Piper synthesis failed: {exc}") from exc
