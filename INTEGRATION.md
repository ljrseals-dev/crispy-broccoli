# hush-voice + AOC Magic v4 — Integration Guide

Merged stack: offline voice control (Whisper → LLM → HUSH bus) + AOC Magic synthesis pipeline (TTS → DSP → VC → finalize).

## Components

| Component | Role |
|-----------|------|
| `install.sh` / `ai-voice.sh` | Unified installer + launcher (6-step install, run/daemon/status/stop/magic) |
| `ai_voice.py` | Full CLI: magic, magic-engines, magic-gui, daw bus helpers |
| `aoc_magic_v3/` | Synthesis engines (piper, kokoro, pyttsx3, xtts, f5) + VC + DSP |
| `voice_daemon.py` | Background STT → LLM → bus → TTS daemon |
| `voice_client.py` / `voice_loop.py` | Socket client + interactive loop |
| `daw_bus.py` + `ops.py` | File-based command bus |
| `hush_bus_executor.cpp` | C++ executor (requires HUSH headers + hush_core) |

## Install

```bash
cd hush-voice
bash install.sh
# emits [1/6] … [6/6], ends with manifest: N files
bash ~/ai-voice/ai-voice.sh daemon
bash ~/ai-voice/ai-voice.sh run
```

## Magic pipeline

```bash
bash ~/ai-voice/ai-voice.sh magic "hello from the installer" --out /tmp/hush_test.wav
bash ~/ai-voice/ai-voice.sh magic-engines
```

Enable magic-routed speech for the daemon:

```bash
python3 - <<'PY'
import json, os
p = os.path.expanduser("~/ai-voice/config.json")
d = json.load(open(p))
d.setdefault("magic", {})["enabled"] = True
json.dump(d, open(p, "w"), indent=2)
print("magic.enabled = True")
PY
bash ~/ai-voice/ai-voice.sh stop
bash ~/ai-voice/ai-voice.sh daemon
```

## HUSH bus (optional)

The C++ executor needs your HUSH tree:

```bash
ln -s /path/to/hush/processor ~/ai-voice/processor
ln -s /path/to/hush/third_party/nlohmann ~/ai-voice/third_party/nlohmann
cd ~/ai-voice && cmake -B build -S . && cmake --build build -j
./build/hush_bus_executor &
python3 -c "import daw_bus; print(daw_bus.alive())"
```

Phone ↔ desktop bridge: prefer `sshfs user@desktop:~/daw_bus ~/daw_bus`.

## Verification checklist

See the full Installation Prompt for the 11-step end-to-end verification table (status before/after daemon, magic "hello", magic-engines, daw_bus.alive, magic.enabled flip, interactive run loop).
