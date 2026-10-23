# AOC Magic V3

Offline voice-generation pipeline with pluggable TTS and voice-conversion engines.

```
text ──▶ TTS (auto/kokoro/xtts/piper/pyttsx3/f5)
       ──▶ DSP (frozen V2 chain)
       ──▶ VC (mkl / knn / none)
       ──▶ finalize (limiter + peak normalize) ──▶ WAV
```

## Quick start

```bash
pip install -e .
python -m aoc_magic_v3 "hello" --out out.wav
python aoc_magic.py "hello" --out out.wav   # V2 shim
python -m aoc_magic_v3 --list-engines
python -m aoc_magic_v3.gui
```

## Engines

### TTS (`--tts-engine auto` priority)

1. **XTTS-v2** — requires references + CUDA (cloning)
2. **Kokoro-82M** — ONNX, CPU-friendly
3. **Piper** — edge ONNX voices
4. **pyttsx3** — always-available OS fallback

Opt-in (not in auto): **F5-TTS** (`--tts-engine f5`) — flow-matching zero-shot, needs GPU ideally.

### VC (`--vc-engine auto` priority)

1. **MKL-VC** — factorized optimal transport (Interspeech 2025)
2. **kNN-VC** — legacy nearest-neighbor
3. **none** — pass-through

## Model setup

Models are **not** auto-downloaded in `is_available()`. Place them under `~/.cache/aoc_magic/`.

### Kokoro

```
~/.cache/aoc_magic/kokoro/kokoro-v1.0.onnx
~/.cache/aoc_magic/kokoro/voices-v1.0.bin
```

From https://huggingface.co/soniqo/Kokoro-82M-ONNX

### Piper

Place `*.onnx` (+ json) under `~/.cache/aoc_magic/piper/`.

### XTTS

Coqui TTS caches weights on first `synthesize()` (not during availability check).

### MKL / kNN

Loaded via `torch.hub` (`bshall/knn-vc`) on first convert. One-time network allowed.

## CLI (V2 flags preserved)

```
--out --ref --no-dsp --pitch --speed --tts-rate --tts-volume
--comp-threshold --comp-ratio --drive --topk --sr
--tts-engine {auto,pyttsx3,kokoro,xtts,piper,f5}
--vc-engine {auto,mkl,knn,none}
--device {auto,cpu,cuda,mps}
--seed --log-level --list-engines
```

## License Matrix

| Component | License | Commercial use |
|-----------|---------|----------------|
| aoc_magic_v3 code | MIT | Yes |
| Kokoro-82M weights | Apache 2.0 | Yes |
| XTTS-v2 weights | CPML | **No** |
| F5-TTS weights | check upstream | check upstream |
| Piper engine | MIT | Yes |
| Piper voices | per-voice | varies |
| MKL-VC (method) | check upstream | check upstream |
| kNN-VC weights | MIT | Yes |

## Consent

Voice conversion must only be performed on voices you own or have explicit permission to use.

## Research Basis

1. Baas, van Niekerk, Kamper. "Voice Conversion With Just Nearest Neighbors." Interspeech 2023. (kNN-VC)
2. Lobashev, Yermekova, Larchenko. "Training-Free Voice Conversion with Factorized Optimal Transport." Interspeech 2025. (MKL-VC)
3. Li et al. "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models." NeurIPS 2023. (Kokoro base)
4. Casanova et al. "XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model." Interspeech 2024. (XTTS-v2)
5. Chen et al. "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching." 2024. (F5-TTS)
6. "kNN-SVC: Robust Zero-Shot Singing Voice Conversion." IEEE 2025. (kNN failure modes)
