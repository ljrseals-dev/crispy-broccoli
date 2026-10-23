"""TTS engine registry."""

from __future__ import annotations

from typing import Dict, List, Type

from aoc_magic_v3.errors import ConfigError
from aoc_magic_v3.tts.base import TTSEngine, TTSResult
from aoc_magic_v3.tts.f5_tts_engine import F5TTSEngine
from aoc_magic_v3.tts.kokoro_engine import KokoroEngine
from aoc_magic_v3.tts.piper_engine import PiperEngine
from aoc_magic_v3.tts.pyttsx3_engine import Pyttsx3Engine
from aoc_magic_v3.tts.xtts_engine import XTTSEngine

__all__ = [
    "TTSEngine",
    "TTSResult",
    "Pyttsx3Engine",
    "KokoroEngine",
    "XTTSEngine",
    "PiperEngine",
    "F5TTSEngine",
    "get_engine",
    "list_engines",
    "AUTO_ORDER",
]

_REGISTRY: Dict[str, Type[TTSEngine]] = {
    "pyttsx3": Pyttsx3Engine,
    "kokoro": KokoroEngine,
    "xtts": XTTSEngine,
    "piper": PiperEngine,
    "f5": F5TTSEngine,
}

# `auto` priority order per §7.5. F5 is intentionally excluded.
AUTO_ORDER = ["xtts", "kokoro", "piper", "pyttsx3"]


def get_engine(name: str) -> TTSEngine:
    if name not in _REGISTRY:
        raise ConfigError(f"Unknown TTS engine: {name!r}")
    return _REGISTRY[name]()


def list_engines() -> List[tuple]:
    out = []
    for name, cls in _REGISTRY.items():
        try:
            eng = cls()
            avail = eng.is_available()
            cloning = eng.supports_cloning()
        except Exception:
            avail = False
            cloning = False
        out.append((name, avail, cloning))
    return out
