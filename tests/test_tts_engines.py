"""Each engine's is_available() does not raise."""

from __future__ import annotations

from aoc_magic_v3.tts.kokoro_engine import KokoroEngine
from aoc_magic_v3.tts.piper_engine import PiperEngine
from aoc_magic_v3.tts.pyttsx3_engine import Pyttsx3Engine
from aoc_magic_v3.tts.xtts_engine import XTTSEngine


def test_pyttsx3_is_available():
    eng = Pyttsx3Engine()
    assert isinstance(eng.is_available(), bool)
    assert eng.supports_cloning() is False
    assert eng.name == "pyttsx3"


def test_kokoro_is_available():
    eng = KokoroEngine()
    assert isinstance(eng.is_available(), bool)
    assert eng.supports_cloning() is False
    assert eng.name == "kokoro"


def test_xtts_is_available():
    eng = XTTSEngine()
    assert isinstance(eng.is_available(), bool)
    assert eng.supports_cloning() is True
    assert eng.name == "xtts"


def test_piper_is_available():
    eng = PiperEngine()
    assert isinstance(eng.is_available(), bool)
    assert eng.supports_cloning() is False
    assert eng.name == "piper"


def test_f5_is_available():
    from aoc_magic_v3.tts.f5_tts_engine import F5TTSEngine
    eng = F5TTSEngine()
    assert isinstance(eng.is_available(), bool)
    assert eng.supports_cloning() is True
    assert eng.name == "f5"
