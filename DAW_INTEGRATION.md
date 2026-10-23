# AOC v4 — DAW Integration Guide

## Bus protocol

Client writes JSON to `~/daw_bus/inbox/<uuid>.json`. Executor reads, mutates DAW, writes reply to `~/daw_bus/outbox/<uuid>.json`. Heartbeat at `~/daw_bus/heartbeat`.

### Command schema

```json
{"id": "uuid", "op": "transport.play", "target": ["transport"], "args": {}, "ts": 1730000000.123}
```

Reply: `{"ok": true, "value": ...}` or `{"ok": false, "error": "..."}`.

### Canonical ops

- `midi.insert_notes` — target `["track", idx]`, args notes list
- `track.set_volume` — volume 0.0–1.0 linear
- `transport.set_tempo` — bpm float
- `transport.play` / `transport.stop`
- `session.query` — what: tracks | tempo | playing

### Client usage

```bash
python ai_voice.py daw --list
python ai_voice.py daw --alive
python ai_voice.py daw transport.play
python ai_voice.py daw transport.set_tempo '{"bpm": 128}'
python ai_voice.py daw --exec &   # minimal test executor
```

### REAPER / Ableton / FL

Full Lua (REAPER), Python remote-script (Ableton), and MIDI-bridge (FL Studio) executor snippets are in the package `executors/README.md` and the full guide shipped with this release. Drop the matching snippet into the DAW and confirm with `daw --alive`.

### Debugging

```bash
python ai_voice.py daw --list
python ai_voice.py daw --alive
python ai_voice.py daw --exec &
python ai_voice.py daw transport.play
watch -n 0.2 'ls -la ~/daw_bus/inbox ~/daw_bus/outbox'
```
