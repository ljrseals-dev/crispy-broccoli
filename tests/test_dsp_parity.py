"""V3 DSP == V2 DSP, bit-identical (§6 / §14).

Loads the authoritative V2 source from aoc_magic_v2_ref.py (copy of the
V2 single-file script) and asserts np.array_equal on pure DSP functions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load V2 reference module
# ---------------------------------------------------------------------------

_V2_PATH = Path(__file__).resolve().parent.parent / "aoc_magic_v2_ref.py"


def _load_v2():
    if not _V2_PATH.is_file():
        pytest.skip(f"V2 reference not found at {_V2_PATH}")
    name = "aoc_magic_v2_ref"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _V2_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


v2 = _load_v2()

from aoc_magic_v3 import dsp as v3  # noqa: E402


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    return rng.standard_normal(8192).astype(np.float32)


def test_compressor_parity(signal):
    assert np.array_equal(
        v2.compressor(signal.copy(), threshold_db=-20.0, ratio=4.0),
        v3.compressor(signal.copy(), threshold_db=-20.0, ratio=4.0),
    )


def test_limiter_parity(signal):
    assert np.array_equal(
        v2.limiter(signal.copy(), ceiling_db=-1.0),
        v3.limiter(signal.copy(), ceiling_db=-1.0),
    )


def test_peak_normalize_parity(signal):
    assert np.array_equal(
        v2.peak_normalize(signal.copy(), ceiling_db=-1.0),
        v3.peak_normalize(signal.copy(), ceiling_db=-1.0),
    )


def test_soft_saturate_parity(signal):
    assert np.array_equal(
        v2.soft_saturate(signal.copy(), 1.5),
        v3.soft_saturate(signal.copy(), 1.5),
    )


def test_highpass_parity(signal):
    assert np.array_equal(
        v2.highpass(signal.copy(), 44100, 70.0),
        v3.highpass(signal.copy(), 44100, 70.0),
    )


def test_lowpass_parity(signal):
    assert np.array_equal(
        v2.lowpass(signal.copy(), 44100, 12000.0),
        v3.lowpass(signal.copy(), 44100, 12000.0),
    )


def test_db_helpers_parity():
    for db in (-12.0, -6.0, -1.0, 0.0, 3.0):
        assert v2.db_to_linear(db) == v3.db_to_linear(db)
    assert abs(v2.linear_to_db(0.5) - v3.linear_to_db(0.5)) < 1e-12


def test_process_character_dsp_parity(signal):
    """In-memory chain vs V2 file-based chain on identical audio content.

    V2 process_character_dsp reads/writes files; we extract the in-memory
    path by calling the same component functions in the same order.
    """
    sr = 44100
    # V2 chain order (from process_character_dsp body):
    y2 = signal.copy()
    y2 = v2.highpass(y2, sr, 70.0)
    y2 = v2.pitch_shift(y2, sr, 2.0)
    y2 = v2.time_stretch(y2, 1.15)
    y2 = v2.compressor(y2, threshold_db=-20.0, ratio=4.0)
    y2 = v2.soft_saturate(y2, 1.5)
    y2 = v2.lowpass(y2, sr, 12000.0)

    y3 = v3.process_character_dsp(
        signal.copy(),
        sr,
        pitch=2.0,
        speed=1.15,
        comp_threshold=-20.0,
        comp_ratio=4.0,
        drive_db=1.5,
        highpass_hz=70.0,
        lowpass_hz=12000.0,
    )
    assert np.array_equal(y2, y3)
