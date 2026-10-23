"""Integration tests for the unified AOC Magic v3 release (3.1.0).

Covers cross-project contracts. The new engines (xtts, f5, mkl) are
tested for *availability gating* only — they are opt-in and heavy, so CI
never imports their real dependencies.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_ai_voice():
    if "ai_voice" in sys.modules:
        return sys.modules["ai_voice"]
    spec = importlib.util.spec_from_file_location("ai_voice", ROOT / "ai_voice.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ai_voice"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fresh_cfg(tmp_path, monkeypatch, ai_voice):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(ai_voice, "CFG_PATH", str(cfg_path))
    monkeypatch.setattr(ai_voice, "BASE", str(tmp_path))
    return cfg_path


def test_magic_tts_falls_back_to_piper_when_kokoro_missing(monkeypatch):
    ai_voice = _load_ai_voice()
    monkeypatch.setattr(ai_voice._MagicKokoro, "is_available", lambda self: False)
    monkeypatch.setattr(ai_voice._MagicPiper, "is_available", lambda self: True)
    cfg = dict(ai_voice.DEFAULT_CFG["magic"])
    cfg["tts_engine"] = "auto"
    assert ai_voice.magic_pick_tts(cfg).name == "piper"


def test_knnvc_raises_magicerror_not_typeerror(monkeypatch):
    ai_voice = _load_ai_voice()
    knn = ai_voice._MagicKnnVC()
    monkeypatch.setattr(knn, "_load",
                        lambda device: (_ for _ in ()).throw(ai_voice.MagicError("boom")))
    with pytest.raises(ai_voice.MagicError):
        knn.convert(__import__("numpy").zeros(16000, dtype="float32"), 16000,
                    [Path("/tmp/fake.wav")], topk=4, device="cpu")


def test_magic_config_block_merged(tmp_path, monkeypatch):
    ai_voice = _load_ai_voice()
    cfg_path = _fresh_cfg(tmp_path, monkeypatch, ai_voice)
    cfg_path.write_text(json.dumps({
        "llm": {"url": "http://127.0.0.1:11434/v1/chat/completions"},
        "magic": {"enabled": True, "pitch": 3.0},
    }))
    merged = ai_voice.load_cfg()
    assert merged["whisper_model"] == "base"
    assert merged["llm"]["url"].endswith(":11434/v1/chat/completions")
    assert merged["magic"]["enabled"] is True
    assert merged["magic"]["pitch"] == 3.0
    assert merged["magic"]["vc_engine"] == "auto"


def test_v2_shim_finalize_signature(tmp_path):
    aoc = importlib.import_module("aoc_magic")
    import numpy as np
    import soundfile as sf
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sr = 22050
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    sf.write(str(src), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
             sr, subtype="PCM_16")
    out = aoc.finalize_audio(str(src), str(dst), sample_rate=sr, ceiling_db=-1.0)
    assert Path(out).exists() and Path(out).stat().st_size > 44


def test_defense_manifest_covers_kokoro_weights(tmp_path, monkeypatch):
    ai_voice = _load_ai_voice()
    kokoro = tmp_path / "kokoro"
    kokoro.mkdir()
    (kokoro / "kokoro-v1.0.onnx").write_bytes(b"\x00k")
    (kokoro / "voices-v1.0.bin").write_bytes(b"\x00v")
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    (voice_dir / "en_US-lessac-medium.onnx").write_bytes(b"\x00p")
    monkeypatch.setattr(ai_voice, "BASE", str(tmp_path))
    monkeypatch.setattr(ai_voice, "VOICE_DIR", str(voice_dir))
    monkeypatch.setattr(ai_voice, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(ai_voice, "KOKORO_CACHE", kokoro)
    monkeypatch.setattr(ai_voice, "PIPER_CACHE", tmp_path / "piper_empty")
    monkeypatch.setattr(ai_voice, "MANIFEST_PATH", str(tmp_path / "models.manifest"))
    keys = "\n".join(ai_voice.build_manifest().keys())
    assert "kokoro-v1.0.onnx" in keys
    assert "voices-v1.0.bin" in keys
    assert "en_US-lessac-medium.onnx" in keys


def test_mkl_in_auto_order_and_gated():
    ai_voice = _load_ai_voice()
    assert "mkl" in ai_voice._MAGIC_VC_AUTO
    assert ai_voice._MAGIC_VC_AUTO[0] == "mkl"
    mkl = ai_voice._MagicMklVC()
    assert mkl.is_available() is False


def test_xtts_not_in_auto_order():
    ai_voice = _load_ai_voice()
    assert "xtts" not in ai_voice._MAGIC_TTS_AUTO
    assert "f5" not in ai_voice._MAGIC_TTS_AUTO


def test_xtts_available_gating():
    ai_voice = _load_ai_voice()
    eng = ai_voice._MagicXTTS()
    assert isinstance(eng.is_available(), bool)


def test_f5_available_gating():
    ai_voice = _load_ai_voice()
    eng = ai_voice._MagicF5()
    assert isinstance(eng.is_available(), bool)


def test_xtts_explicit_engine_requires_refs(monkeypatch):
    ai_voice = _load_ai_voice()
    monkeypatch.setattr(ai_voice._MagicXTTS, "is_available", lambda self: True)
    eng = ai_voice._MagicXTTS()
    with pytest.raises(ai_voice.MagicError):
        eng.synthesize("hi", rate=260, volume=1.0, reference_paths=None)


def test_f5_explicit_engine_requires_refs(monkeypatch):
    ai_voice = _load_ai_voice()
    monkeypatch.setattr(ai_voice._MagicF5, "is_available", lambda self: True)
    eng = ai_voice._MagicF5()
    with pytest.raises(ai_voice.MagicError):
        eng.synthesize("hi", rate=260, volume=1.0, reference_paths=None)


def test_mkl_explicit_engine_requires_repo(monkeypatch):
    ai_voice = _load_ai_voice()
    monkeypatch.setattr(ai_voice._MagicMklVC, "is_available", lambda self: True)
    eng = ai_voice._MagicMklVC()
    with pytest.raises(ai_voice.MagicError):
        eng.convert(__import__("numpy").zeros(16000, dtype="float32"), 16000,
                    [], topk=4, device="cpu")
