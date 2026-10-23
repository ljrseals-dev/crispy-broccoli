"""Voice-conversion engine interface contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class VCEngine(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this VC can run. Must not raise. No network I/O."""

    @abstractmethod
    def convert(
        self,
        source: np.ndarray,
        source_sr: int,
        reference_paths: List[Path],
        *,
        topk: int = 4,
        device: str = "cpu",
    ) -> tuple:
        """Return (audio, sample_rate). Must raise VcError on failure.
        Must not write to disk."""
