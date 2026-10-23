"""Voice-conversion engine registry."""

from __future__ import annotations

from typing import Dict, List, Type

from aoc_magic_v3.errors import ConfigError
from aoc_magic_v3.vc.base import VCEngine
from aoc_magic_v3.vc.knn_vc import KnnVC
from aoc_magic_v3.vc.mkl_vc import MklVC
from aoc_magic_v3.vc.none import NoneVC

__all__ = [
    "VCEngine",
    "MklVC",
    "KnnVC",
    "NoneVC",
    "get_engine",
    "list_engines",
    "AUTO_ORDER",
]

_REGISTRY: Dict[str, Type[VCEngine]] = {
    "mkl": MklVC,
    "knn": KnnVC,
    "none": NoneVC,
}

AUTO_ORDER = ["mkl", "knn", "none"]


def get_engine(name: str) -> VCEngine:
    if name not in _REGISTRY:
        raise ConfigError(f"Unknown VC engine: {name!r}")
    return _REGISTRY[name]()


def list_engines() -> List[tuple]:
    out = []
    for name, cls in _REGISTRY.items():
        try:
            avail = cls().is_available()
        except Exception:
            avail = False
        out.append((name, avail))
    return out
