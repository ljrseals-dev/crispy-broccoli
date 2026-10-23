"""Each VC's is_available() does not raise."""

from __future__ import annotations

from aoc_magic_v3.vc.knn_vc import KnnVC
from aoc_magic_v3.vc.mkl_vc import MklVC
from aoc_magic_v3.vc.none import NoneVC


def test_none_vc():
    eng = NoneVC()
    assert eng.is_available() is True
    assert eng.name == "none"
    import numpy as np
    y = np.zeros(100, dtype=np.float32)
    out, sr = eng.convert(y, 22050, [])
    assert out.shape == y.shape
    assert sr == 22050


def test_knn_is_available():
    eng = KnnVC()
    assert isinstance(eng.is_available(), bool)
    assert eng.name == "knn"


def test_mkl_is_available():
    eng = MklVC()
    assert isinstance(eng.is_available(), bool)
    assert eng.name == "mkl"
