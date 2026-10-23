# AOC v4 — Technical Manual

Version 4.0.0. Authoritative reference for the AOC v4 runtime.

## 1. Architecture

Three surfaces sharing one config, one defense manifest, one log stream:

| Surface | Entry | Backend |
|---------|-------|---------|
| Voice control | `daemon` / `run` | mic → Whisper → LLM → TTS |
| Synthesis | `magic` / `magic-gui` | TTS → DSP → VC → finalize → WAV |
| DAW bus | `daw` | filesystem inbox/outbox JSON |

Shared paths: `~/ai-voice/{config.json,ctl.sock,daemon.pid,logs/,models.manifest,ratelimit.json}`, `~/daw_bus/`, `~/.cache/aoc_magic/`.

## 2. Config schema (`~/ai-voice/config.json`)

Key blocks: `whisper_*`, `llm`, `tts`, `defense`, `magic`. All fields optional; defaults in `DEFAULT_CFG`. Set `magic.enabled: true` to route `speak()` through the synthesis pipeline.

## 3. Engine matrix

| Engine | Kind | Auto | Cloning | Extra |
|--------|------|------|---------|-------|
| kokoro | TTS | yes | no | `[tts-kokoro]` |
| piper | TTS | yes | no | `[tts-piper]` |
| pyttsx3 | TTS | yes | no | built-in |
| xtts | TTS | no | yes | `[tts-xtts]` (CPML) |
| f5 | TTS | no | yes | `[tts-f5]` |
| mkl | VC | yes | — | `[vc]` + repo + checkpoints |
| knn | VC | yes | — | `[vc]` |
| none | VC | yes | — | built-in |

Auto TTS: `kokoro → piper → pyttsx3`. Auto VC: `mkl → knn → none`.

## 4. MKL-VC setup

```bash
git clone https://github.com/alobashev/mkl-vc ~/.cache/aoc_magic/mkl-vc
mkdir -p ~/.cache/aoc_magic/mkl && cd ~/.cache/aoc_magic/mkl
pip install gdown
gdown https://drive.google.com/uc?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU
wget https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt
wget https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt
```

## 5. Defense

1. SHA-256 manifest (covers voices, models, kokoro, piper caches)
2. Permission lock (0600 files / 0700 dirs)
3. Injection regex guard
4. Rate limit (default 20 asks / 60 s)

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Whisper OOM | `whisper_model: "tiny"` |
| LLM refused | start llama-server :8080 or ollama :11434 |
| mkl unavailable | run MKL setup above |
| Defense tamper | `python ai_voice.py defense build` |
| Rate limited | wait or `defense reset` |
