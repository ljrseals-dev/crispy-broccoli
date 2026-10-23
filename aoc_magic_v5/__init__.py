"""AOC Magic v5 — offline TTS → DSP → VC → finalize synthesis pipeline."""

__version__ = "5.0.0"

try:
    from aoc_magic_v5.pipeline import run_pipeline
except Exception:
    run_pipeline = None

try:
    from aoc_magic_v5.config import Config, load_cfg
except Exception:
    Config = None
    def load_cfg():
        import json, os
        from pathlib import Path
        p = Path.home() / "ai-voice" / "config.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}

__all__ = ["__version__", "run_pipeline", "Config", "load_cfg"]
