"""Exercise every fallback path in §8 / §14.

§14 requires monkeypatching optional *imports* to raise ImportError and
asserting the pipeline still returns a WAV.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from aoc_magic_v3.config import Config
from aoc_magic_v3.errors import EngineUnavailable, VcError
from aoc_magic_v3.pipeline import run_pipeline, synthesize_speech
from aoc_magic_v3.tts.base import TTSResult


_BLOCKED = {"torch", "torchaudio", "TTS", "kokoro_onnx", "piper", "onnxruntime"}


def _fake_tts_result(sr: int = 44100) -> TTSResult:
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return TTSResult(audio=audio, sample_rate=sr, engine="mock")


@pytest.fixture
def out_path(tmp_path):
    return tmp_path / "out.wav"


def test_pipeline_survives_missing_optional_imports(monkeypatch, out_path):
    """§14: block optional deps at import boundary; pipeline still writes WAV."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in _BLOCKED:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Force pyttsx3 path with a deterministic in-memory result so we do not
    # depend on OS speech engines inside the import-block test.
    monkeypatch.setattr(
        "aoc_magic_v3.tts.pyttsx3_engine.Pyttsx3Engine.is_available",
        lambda self: True,
    )
    monkeypatch.setattr(
        "aoc_magic_v3.tts.pyttsx3_engine.Pyttsx3Engine.synthesize",
        lambda self, *a, **k: _fake_tts_result(),
    )

    cfg = Config(
        text="hello",
        out=out_path,
        tts_engine="auto",
        vc_engine="auto",
        ref=[],  # no refs → VC skipped
    )
    result = run_pipeline(cfg)
    assert Path(result).is_file()
    assert Path(result).stat().st_size > 44


def test_tts_auto_falls_back_when_engines_unavailable(out_path, monkeypatch):
    from aoc_magic_v3.tts import kokoro_engine, xtts_engine, piper_engine

    monkeypatch.setattr(kokoro_engine.KokoroEngine, "is_available", lambda self: False)
    monkeypatch.setattr(xtts_engine.XTTSEngine, "is_available", lambda self: False)
    monkeypatch.setattr(piper_engine.PiperEngine, "is_available", lambda self: False)
    monkeypatch.setattr(
        "aoc_magic_v3.tts.pyttsx3_engine.Pyttsx3Engine.is_available",
        lambda self: True,
    )
    monkeypatch.setattr(
        "aoc_magic_v3.tts.pyttsx3_engine.Pyttsx3Engine.synthesize",
        lambda self, *a, **k: _fake_tts_result(),
    )

    cfg = Config(text="hello", out=out_path, tts_engine="auto", vc_engine="none")
    result = run_pipeline(cfg)
    assert Path(result).is_file()


def test_explicit_engine_no_silent_fallback(monkeypatch):
    """§8: explicit --tts-engine kokoro must raise, not fall back."""
    from aoc_magic_v3.tts import kokoro_engine

    monkeypatch.setattr(kokoro_engine.KokoroEngine, "is_available", lambda self: False)
    cfg = Config(text="hello", tts_engine="kokoro", vc_engine="none")
    with pytest.raises(EngineUnavailable):
        synthesize_speech(cfg)


def test_vc_failure_skips_and_still_writes_wav(out_path, monkeypatch):
    monkeypatch.setattr(
        "aoc_magic_v3.pipeline.synthesize_speech",
        lambda cfg: _fake_tts_result(),
    )
    from aoc_magic_v3.vc import knn_vc, mkl_vc

    def boom(*a, **k):
        raise VcError("boom")

    monkeypatch.setattr(mkl_vc.MklVC, "is_available", lambda self: True)
    monkeypatch.setattr(mkl_vc.MklVC, "convert", boom)
    monkeypatch.setattr(knn_vc.KnnVC, "is_available", lambda self: True)
    monkeypatch.setattr(knn_vc.KnnVC, "convert", boom)

    cfg = Config(
        text="hello",
        out=out_path,
        ref=[Path("/tmp/fake_ref.wav")],
        tts_engine="auto",
        vc_engine="auto",
    )
    result = run_pipeline(cfg)
    assert Path(result).is_file()


def test_dsp_failure_skips_and_still_writes_wav(out_path, monkeypatch):
    monkeypatch.setattr(
        "aoc_magic_v3.pipeline.synthesize_speech",
        lambda cfg: _fake_tts_result(),
    )
    from aoc_magic_v3.errors import DspError

    def boom(*a, **k):
        raise DspError("dsp boom")

    monkeypatch.setattr("aoc_magic_v3.pipeline.process_character_dsp", boom)

    cfg = Config(text="hello", out=out_path, vc_engine="none")
    result = run_pipeline(cfg)
    assert Path(result).is_file()


def test_is_available_never_raises():
    from aoc_magic_v3.tts.kokoro_engine import KokoroEngine
    from aoc_magic_v3.tts.xtts_engine import XTTSEngine
    from aoc_magic_v3.tts.piper_engine import PiperEngine
    from aoc_magic_v3.vc.knn_vc import KnnVC
    from aoc_magic_v3.vc.mkl_vc import MklVC

    for cls in (KokoroEngine, XTTSEngine, PiperEngine, KnnVC, MklVC):
        assert cls().is_available() in (True, False)
