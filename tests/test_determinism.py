"""Same seed → identical bytes (for deterministic stages)."""

from __future__ import annotations

import numpy as np

from aoc_magic_v3.dsp import process_character_dsp


def test_dsp_determinism():
    rng = np.random.RandomState(0)
    y = rng.randn(8000).astype(np.float32) * 0.1
    a = process_character_dsp(y, 44100, pitch=0.0, speed=1.0, drive_db=0.0)
    b = process_character_dsp(y, 44100, pitch=0.0, speed=1.0, drive_db=0.0)
    assert np.array_equal(a, b)
