// hush_engine_adapters.cpp — real-engine glue for hush_bus_executor.
#include "ForensicRack.h"
#include "AudioEngine.h"
#include "EngineBridge.h"
#include "json.hpp"

using json = nlohmann::json;

namespace hush {

static AudioEngine* eng() { return AudioEngine::instance(); }

static ForensicRack* rk(const std::string& track) {
    return eng()->find_rack(track);
}

void set_track_gain(const std::string& track, double gain_db) {
    eng()->set_track_gain_db(track, static_cast<float>(gain_db));
}

void set_track_mute(const std::string& track, bool muted) {
    eng()->set_track_mute(track, muted);
}

void set_forensic_params(const std::string& track, const json& args) {
    ForensicRack* r = rk(track);
    if (!r) return;
    ForensicParams fp{};
    fp.eq_low_db          = args.value("eq_low_db", 0.0f);
    fp.eq_mid_db          = args.value("eq_mid_db", 0.0f);
    fp.eq_high_db         = args.value("eq_high_db", 0.0f);
    fp.comp_ratio         = args.value("comp_ratio", 1.0f);
    fp.comp_threshold_db  = args.value("comp_threshold_db", 0.0f);
    fp.gate_threshold_db  = args.value("gate_threshold_db", -80.0f);
    fp.denoise_amount     = args.value("denoise_amount", 0.0f);
    r->setParams(fp);
}

int insert_midi_notes(const std::string& track, const json& args) {
    MidiClip clip{};
    clip.track        = track;
    clip.length_bars  = args.value("length_bars", 1);
    if (args.contains("notes") && args["notes"].is_array()) {
        for (const auto& n : args["notes"]) {
            MidiNote note{};
            note.pitch = n.value("pitch", 60);
            note.start = n.value("start", 0.0f);
            note.dur   = n.value("dur", 1.0f);
            note.vel   = n.value("vel", 100);
            clip.notes.push_back(note);
        }
    }
    eng()->insert_midi_clip(clip);
    return static_cast<int>(clip.notes.size());
}

void set_tempo(double bpm) {
    EngineBridge::instance()->setTempo(static_cast<float>(bpm));
}

void transport_play() {
    EngineBridge::instance()->transportPlay();
}

void transport_stop() {
    EngineBridge::instance()->transportStop();
}

json query_session() {
    json out;
    json tracks = json::array();
    for (const auto& name : eng()->track_names()) {
        tracks.push_back(name);
    }
    out["tracks"] = tracks;
    out["tempo"] = eng()->tempo();
    return out;
}

} // namespace hush
