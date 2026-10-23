"""Main synthesis pipeline with fallback contract (§8).

Stage order matches V2 for backward compatibility when engines are auto:
  TTS → DSP (source conditioning) → VC → Finalize
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from aoc_magic_v3.config import Config
from aoc_magic_v3.dsp import process_character_dsp
from aoc_magic_v3.errors import (
    AocError,
    DspError,
    EngineUnavailable,
    TTSError,
    VcError,
)
from aoc_magic_v3.finalize import finalize_audio
from aoc_magic_v3.tts import AUTO_ORDER as TTS_AUTO_ORDER
from aoc_magic_v3.tts import get_engine as get_tts_engine
from aoc_magic_v3.tts.base import TTSResult
from aoc_magic_v3.vc import AUTO_ORDER as VC_AUTO_ORDER
from aoc_magic_v3.vc import get_engine as get_vc_engine

logger = logging.getLogger(__name__)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32)
    import librosa
    return librosa.resample(
        audio.astype(np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="soxr_hq",
    ).astype(np.float32)


def _select_tts_engines(cfg: Config) -> List[str]:
    if cfg.tts_engine != "auto":
        return [cfg.tts_engine]

    order = list(TTS_AUTO_ORDER)
    device = cfg.resolved_device()
    if not cfg.ref or device != "cuda":
        order = [e for e in order if e != "xtts"]
    return order


def _select_vc_engines(cfg: Config) -> List[str]:
    if cfg.vc_engine != "auto":
        return [cfg.vc_engine]
    return list(VC_AUTO_ORDER)


def synthesize_speech(cfg: Config) -> TTSResult:
    """Run TTS stage with fallback per §8."""
    engines = _select_tts_engines(cfg)
    last_err: Optional[Exception] = None
    device = cfg.resolved_device()

    for name in engines:
        try:
            eng = get_tts_engine(name)
            if not eng.is_available():
                if cfg.tts_engine != "auto":
                    raise EngineUnavailable(f"TTS engine '{name}' is not available")
                logger.warning("TTS engine '%s' unavailable, trying next", name)
                continue
            logger.info("[1/4] TTS via %s …", name)
            result = eng.synthesize(
                cfg.text,
                rate=cfg.tts_rate,
                volume=cfg.tts_volume,
                reference_paths=cfg.ref or None,
                device=device,
            )
            if result.sample_rate != cfg.sample_rate:
                audio = _resample(result.audio, result.sample_rate, cfg.sample_rate)
                result = TTSResult(
                    audio=audio, sample_rate=cfg.sample_rate, engine=result.engine
                )
            logger.info(
                "TTS done: engine=%s shape=%s sr=%d",
                result.engine,
                result.audio.shape,
                result.sample_rate,
            )
            return result
        except (EngineUnavailable, TTSError) as exc:
            last_err = exc
            if cfg.tts_engine != "auto":
                raise
            logger.warning(
                "TTS engine '%s' failed: %s — trying next",
                name,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            continue

    msg = (
        "No TTS engine available. Install pyttsx3 (pip install pyttsx3) or place a "
        "Kokoro/Piper/XTTS model in ~/.cache/aoc_magic/."
    )
    if last_err:
        raise EngineUnavailable(msg) from last_err
    raise EngineUnavailable(msg)


def run_dsp(audio: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    """Run DSP stage; on failure log warning and return input."""
    if not cfg.use_dsp:
        logger.info("[2/4] DSP conditioning disabled.")
        return audio
    try:
        logger.info("[2/4] DSP …")
        out = process_character_dsp(
            audio,
            sr,
            pitch=cfg.pitch,
            speed=cfg.speed,
            comp_threshold=cfg.comp_threshold,
            comp_ratio=cfg.comp_ratio,
            drive_db=cfg.drive_db,
            highpass_hz=cfg.highpass_hz,
            lowpass_hz=cfg.lowpass_hz,
        )
        return out
    except DspError as exc:
        logger.warning(
            "DSP failed: %s — using pre-DSP audio",
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return audio


def run_vc(
    audio: np.ndarray,
    sr: int,
    cfg: Config,
) -> tuple:
    """Run VC stage with fallback per §8. Returns (audio, sr)."""
    if not cfg.ref:
        logger.info("[3/4] No references supplied; skipping VC.")
        return audio, sr

    engines = _select_vc_engines(cfg)
    device = cfg.resolved_device()

    for name in engines:
        if name == "none":
            logger.info("[3/4] VC engine=none")
            return audio, sr
        try:
            eng = get_vc_engine(name)
            if not eng.is_available():
                if cfg.vc_engine != "auto":
                    raise VcError(f"VC engine '{name}' is not available")
                logger.warning("VC engine '%s' unavailable, trying next", name)
                continue
            logger.info("[3/4] VC via %s …", name)
            out, out_sr = eng.convert(
                audio,
                sr,
                cfg.ref,
                topk=cfg.topk,
                device=device,
            )
            if out_sr != sr:
                out = _resample(out, out_sr, sr)
                out_sr = sr
            logger.info("VC done: engine=%s shape=%s sr=%d", name, out.shape, out_sr)
            return out, out_sr
        except VcError as exc:
            if cfg.vc_engine != "auto":
                raise
            logger.warning(
                "VC engine '%s' failed: %s — skipping VC",
                name,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            continue

    logger.warning("All VC engines failed/skipped — using pre-VC audio")
    return audio, sr


def run_pipeline(cfg: Config) -> Path:
    """Full pipeline: TTS → DSP → VC → Finalize (V2 order)."""
    np.random.seed(cfg.seed)
    try:
        import torch
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    except ImportError:
        pass

    tts_result = synthesize_speech(cfg)
    audio, sr = tts_result.audio, tts_result.sample_rate

    # V2 order: DSP before VC (source conditioning)
    audio = run_dsp(audio, sr, cfg)

    audio, sr = run_vc(audio, sr, cfg)

    if sr != cfg.sample_rate:
        audio = _resample(audio, sr, cfg.sample_rate)
        sr = cfg.sample_rate

    logger.info("[4/4] Finalize …")
    return finalize_audio(audio, sr, cfg.out, ceiling_db=cfg.limiter_ceiling_db)
