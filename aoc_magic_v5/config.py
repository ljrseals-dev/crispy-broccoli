"""Configuration dataclass for aoc_magic_v3.

Field names and defaults match V2 for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Pipeline configuration. Field names aligned with V2 + V3 extensions."""

    text: str = ""
    out: Path = field(default_factory=lambda: Path("aoc_style_final.wav"))
    ref: List[Path] = field(default_factory=list)

    # TTS (V2 names)
    tts_rate: int = 260
    tts_volume: float = 1.0

    # DSP / character (V2 defaults)
    sample_rate: int = 44100
    pitch: float = 2.0
    speed: float = 1.15
    comp_threshold: float = -20.0
    comp_ratio: float = 4.0
    drive_db: float = 1.5
    highpass_hz: float = 70.0
    lowpass_hz: float = 12000.0
    limiter_ceiling_db: float = -1.0

    # VC
    topk: int = 4

    # V3 engine selection
    tts_engine: str = "auto"
    vc_engine: str = "auto"
    use_dsp: bool = True

    # Runtime
    device: str = "auto"
    seed: int = 0
    log_level: str = "info"

    @property
    def rate(self) -> int:
        return self.tts_rate

    @property
    def volume(self) -> float:
        return self.tts_volume

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


def load_cfg(path=None):
    """Load config from ~/ai-voice/config.json or defaults."""
    import json
    from pathlib import Path
    p = Path(path) if path else Path.home() / "ai-voice" / "config.json"
    defaults = {
        "whisper_model": "base",
        "whisper_language": "en",
        "sample_rate": 16000,
        "llm": {"url": "http://127.0.0.1:8080/v1/chat/completions", "model": "local"},
        "tts": {"voice": "en_US-lessac-medium", "speed": 1.0},
        "magic": {"enabled": False, "tts_engine": "auto", "vc_engine": "none",
                  "device": "cpu", "sample_rate": 44100, "limiter_ceiling_db": -1.0},
    }
    if p.exists():
        try:
            data = json.loads(p.read_text())
            # shallow merge
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(defaults.get(k), dict):
                    defaults[k] = {**defaults[k], **v}
                else:
                    defaults[k] = v
        except Exception:
            pass
    return defaults
