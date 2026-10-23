The wire format (`~/daw_bus/inbox` / `outbox` JSON files) and the op
contract in `ops.py` are host-agnostic — any tool that can poll a
directory and write a JSON reply can be a DAW executor. The dispatch
function itself, however, has to be re-implemented per host (~40–60
lines each for REAPER Lua, Ableton Python, FL Studio Python), because
each DAW exposes a completely different mutation API: REAPER gives you
`reaper.InsertMediaItem`/`reaper.SetMediaItemInfo_Value`, Ableton gives
you `Live.Song`/`Live.Clip` on the Python remote-script side, and FL
Studio exposes only its MIDI controller scripting surface. There is no
shared "set the volume of track 3" call to shim, so the dispatcher —
not the protocol — is what changes.

**REAPER (Lua) sketch**

```lua
function dispatch(op, target, args)
  if op == "transport.set_tempo" then
    reaper.SetCurrentBPM(0, args.bpm, true); return {ok=true}
  elseif op == "transport.play" then
    reaper.Main_OnCommand(1007, 0); return {ok=true}
  elseif op == "track.set_volume" then
    local tr = reaper.GetTrack(0, tonumber(target[2]))
    reaper.SetMediaTrackInfo_Value(tr, "D_VOL", args.volume); return {ok=true}
  elseif op == "midi.insert_notes" then
    local tr = reaper.GetTrack(0, tonumber(target[2]))
    for _, n in ipairs(args.notes) do
      local item = reaper.CreateNewMIDIItemInProj(tr, n.start, n.start+n.length)
      local take = reaper.GetActiveTake(item)
      reaper.MIDI_InsertNote(take, false, false, 0, n.length, 0, n.pitch, n.velocity, true)
    end
    return {ok=true, inserted=#args.notes}
  end
  return {ok=false, error="unknown op"}
end
```

**Ableton (Python, inside a Control Surface script)**

```python
def dispatch(op, target, args):
    song = self.song()
    if op == "transport.set_tempo":
        song.tempo = args["bpm"]; return {"ok": True}
    if op == "transport.play":
        song.start_playing(); return {"ok": True}
    if op == "transport.stop":
        song.stop_playing(); return {"ok": True}
    if op == "track.set_volume":
        track = song.tracks[int(target[1])]
        track.mixer_device.volume.value = args["volume"]; return {"ok": True}
    if op == "midi.insert_notes":
        track = song.tracks[int(target[1])]
        clip = track.create_midi_clip(0.0, 4.0)
        for n in args["notes"]:
            clip.set_notes((n["pitch"], n["start"], n["length"],
                            n["velocity"], False))
        return {"ok": True, "inserted": len(args["notes"])}
    return {"ok": False, "error": "unknown op"}
```
