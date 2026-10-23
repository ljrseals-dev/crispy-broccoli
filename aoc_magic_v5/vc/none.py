"""No-op voice conversion engine."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from aoc_magic_v3.vc.base import VCEngine


class NoneVC(VCEngine):
    name = "none"

    def is_available(self) -> bool:
        return True

    def convert(
        self,
        source: np.ndarray,
        source_sr: int,
        reference_paths: List[Path],
        *,
        topk: int = 4,
        device: str = "cpu",
    ) -> tuple:
        return np.asarray(source, dtype=np.float32), int(source_sr)
