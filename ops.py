"""Canonical DAW ops (host-agnostic).

Each op has {target, args} contract:

  midi.insert_notes
    target: ["track", "<track_id>"]
    args:   {"notes": [{"pitch": int, "start": float, "length": float,
                        "velocity": int}]}
    result: {"ok": True, "inserted": int}

  track.set_volume
    target: ["track", "<track_id>"]
    args:   {"volume": float}          # 0.0 .. 1.0
    result: {"ok": True}

  transport.set_tempo
    target: ["transport"]
    args:   {"bpm": float}
    result: {"ok": True}

  transport.play
    target: ["transport"]
    args:   {}
    result: {"ok": True}

  transport.stop
    target: ["transport"]
    args:   {}
    result: {"ok": True}

  session.query
    target: ["session"]
    args:   {"what": "tracks" | "tempo" | "playing"}
    result: {"ok": True, "value": <any>}
"""

OPS = {
    "midi.insert_notes": {"target": ["track"], "args": ["notes"]},
    "track.set_volume": {"target": ["track"], "args": ["volume"]},
    "transport.set_tempo": {"target": ["transport"], "args": ["bpm"]},
    "transport.play": {"target": ["transport"], "args": []},
    "transport.stop": {"target": ["transport"], "args": []},
    "session.query": {"target": ["session"], "args": ["what"]},
}
