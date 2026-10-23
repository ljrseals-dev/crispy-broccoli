"""Dataclass defaults match V2 expectations."""

from __future__ import annotations

from pathlib import Path

from aoc_magic_v3.config import Config


def test_defaults():
    c = Config()
    assert c.sample_rate == 44100
    assert c.pitch == 2.0
    assert c.speed == 1.15
    assert c.comp_threshold == -20.0
    assert c.comp_ratio == 4.0
    assert c.drive_db == 1.5
    assert c.limiter_ceiling_db == -1.0
    assert c.tts_rate == 260
    assert c.tts_volume == 1.0
    assert c.tts_engine == "auto"
    assert c.vc_engine == "auto"
    assert c.topk == 4
    assert c.device == "auto"
    assert c.seed == 0
    assert c.log_level == "info"
    assert c.out == Path("aoc_style_final.wav")
    assert c.ref == []
    assert c.highpass_hz == 70.0
    assert c.lowpass_hz == 12000.0


def test_resolved_device_cpu_without_torch():
    c = Config(device="auto")
    d = c.resolved_device()
    assert d in ("cpu", "cuda", "mps")
