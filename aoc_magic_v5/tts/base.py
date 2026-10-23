"""TTS engine interface contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class TTSResult:
    audio: np.ndarray  # float32, mono, shape (n,)
    sample_rate: int  # actual rate of `audio`
    engine: str  # engine name, e.g. "kokoro"


class TTSEngine(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this engine can run in the current environment.
        Must not raise. Must not perform network I/O."""

    @abstractmethod
    def supports_cloning(self) -> bool:
        """Return True if this engine accepts a reference for voice cloning."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        rate: int,
        volume: float,
        reference_paths: Optional[List[Path]] = None,
        device: str = "cpu",
    ) -> TTSResult:
        """Synthesize `text`. Must raise EngineUnavailable if not available.
        Must raise TTSError on synthesis failure.
        Must not write to disk. Must return in-memory audio."""
