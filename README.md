# AOC v5

Offline voice, synthesis, and HUSH command bus.

```bash
bash install.sh
aoc-magic-engines
aoc-magic "hello from AOC v5" --out /tmp/h5.wav
aoc-daemon &
aoc-voice
```

See INTEGRATION.md for full wiring and the Installation Prompt for verification steps.

# hush-voice + AOC Magic v4

Offline voice generation, voice control, HUSH command bus, and DAW integration in one project.

See **[INTEGRATION.md](INTEGRATION.md)** for the full wiring diagram, phone↔desktop bridging, and verification checklist.

```
text ──▶ TTS (kokoro / piper / pyttsx3 / xtts / f5)
     ──▶ DSP (frozen V2 chain)
     ──▶ VC (mkl / knn / none)
     ──▶ finalize (limiter + peak normalize) ──▶ WAV ──▶ DAW bus
```

- **Voice control** — microphone → Whisper → local LLM → TTS on Termux/Linux.
- **Synthesis** — the full offline pipeline, no extra package install required.
- **DAW integration** — host-agnostic filesystem bus; executors for REAPER,
  Ableton, and FL Studio ship as ~50-line snippets each.

See [`TECHNICAL.md`](TECHNICAL.md), [`TECHNOLOGY.md`](TECHNOLOGY.md), and
[`DAW_INTEGRATION.md`](DAW_INTEGRATION.md) for the full manual.

## Install

```bash
pip install -e .[dsp,test]        # minimal (pyttsx3-only TTS works out of the box)
pip install -e .[tts-kokoro]      # Kokoro-82M ONNX (CPU-friendly)
pip install -e .[tts-xtts]        # XTTS-v2 (CPML — non-commercial only)
pip install -e .[tts-f5]          # F5-TTS (flow-matching, CPU or GPU)
pip install -e .[vc]              # kNN-VC and MKL-VC (torch + gdown)
pip install -e .[all]             # everything
python ai_voice.py install        # Termux/Linux voice-control setup
```

### Entrypoint 1 — voice control

```bash
python ai_voice.py install
python ai_voice.py doctor
python ai_voice.py defense audit
python ai_voice.py daemon &
python ai_voice.py run
python ai_voice.py status
python ai_voice.py stop
```

### Entrypoint 2 — offline synthesis

```bash
python ai_voice.py magic "hello" --out out.wav
python ai_voice.py magic-gui
python ai_voice.py magic-engines
python ai_voice.py magic-passthrough "hello" --out x.wav
python -m aoc_magic_v3 "hello" --out x.wav
python aoc_magic.py "hello" --out x.wav
```

### Entrypoint 3 — DAW integration

```bash
python ai_voice.py daw --list
python ai_voice.py daw transport.play
python ai_voice.py daw transport.set_tempo '{"bpm": 128}'
python ai_voice.py daw track.set_volume '{"track": 1, "volume": 0.75}'
python ai_voice.py daw --exec &
```

### Engines

**TTS (magic):** `kokoro` → `piper` → `pyttsx3` (auto). `xtts` / `f5` opt-in via `--tts-engine`.

**VC (magic):** `mkl` → `knn` → `none` (auto, self-gating).

**Voice control `speak()`:** if `config.magic.enabled`, routes through magic pipeline first.

### One config

`~/ai-voice/config.json` — voice-control + `magic` block. Set `"magic.enabled": true` to route `speak()` through the synthesis pipeline.

### Consent

Voice conversion must only be performed on voices you own or have explicit permission to use.
