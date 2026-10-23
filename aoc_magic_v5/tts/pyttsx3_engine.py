"""Pyttsx3 TTS engine – final offline fallback."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from aoc_magic_v3.errors import EngineUnavailable, TTSError
from aoc_magic_v3.tts.base import TTSEngine, TTSResult

logger = logging.getLogger(__name__)

_AVAILABLE = False
try:
    import pyttsx3  # noqa: F401
    _AVAILABLE = True
except ImportError:
    pyttsx3 = None  # type: ignore


class Pyttsx3Engine(TTSEngine):
    name = "pyttsx3"

    def is_available(self) -> bool:
        return _AVAILABLE and pyttsx3 is not None

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
            raise EngineUnavailable("pyttsx3 is not installed")

        rate = int(max(50, min(500, rate)))
        volume = float(max(0.0, min(1.0, volume)))

        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                engine.save_to_file(text, str(tmp_path))
                engine.runAndWait()

                if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                    raise TTSError("pyttsx3 produced empty or missing temp file")

                import librosa
                audio, sr = librosa.load(str(tmp_path), sr=None, mono=True)
                audio = np.asarray(audio, dtype=np.float32)
                return TTSResult(audio=audio, sample_rate=int(sr), engine=self.name)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"pyttsx3 synthesis failed: {exc}") from exc
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
