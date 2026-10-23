# AOC v4 — Technology Overview

## Design goals (priority order)

1. Fully offline — no cloud, no API keys, no telemetry.
2. Termux-first, Linux-second.
3. Degrade, never crash — every layer has a defined fallback.
4. One program, one config, one manifest.
5. Host-agnostic DAW contract (filesystem bus, not plugin ABI).

## Dependency graph

Minimum: `numpy`, `soundfile`, `librosa`, `pyttsx3`.
Optional: scipy, kokoro-onnx, torch, TTS, f5-tts, piper-tts.

## Latency budget (Termux mid-range Android)

End-to-end voice loop: ~5 s typical, 10 s P95 (record + Whisper + LLM + TTS).

## Resource profile

| Config | Disk | RAM |
|--------|------|-----|
| Minimal (pyttsx3) | 40 MB | 120 MB |
| + Whisper base | 190 MB | 550 MB |
| + Piper | +60 MB | +40 MB |
| + Kokoro | +80 MB | +80 MB |
| + kNN-VC | +2.1 GB | +2.5 GB transient |
| + MKL-VC | +700 MB | +1.2 GB transient |
| + XTTS | +1.8 GB | +3.5 GB transient |
| + F5 | +1.2 GB | +2.5 GB transient |

## Offline guarantees

Voice control, magic (kokoro/piper/pyttsx3), DAW bus, defense audit: **no network**.
xtts/f5/kNN/MKL: network only on first cache warm.

## Why a single file?

Termux users often cannot `pip install -e .`. A single `python ai_voice.py` runs with no install. One config governs every behavior.

## Why a filesystem DAW bus?

REAPER (Lua), Ableton (Python remote script), FL Studio (MIDI) have incompatible APIs. A filesystem bus is the only common denominator that is debuggable with `ls`/`cat`, survives restarts, and works across machines/containers/SSH.
