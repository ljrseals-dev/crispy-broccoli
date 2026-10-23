// hush_bus_executor.cpp — file-based command bus executor for HUSH.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace hush {
    void set_track_gain(const std::string& track, double gain_db);
    void set_track_mute(const std::string& track, bool muted);
    void set_forensic_params(const std::string& track, const json& args);
    int  insert_midi_notes(const std::string& track, const json& args);
    void set_tempo(double bpm);
    void transport_play();
    void transport_stop();
    json query_session();
}

static std::atomic<bool> g_run{true};

struct Bus {
    fs::path root, inbox, outbox, processing, heartbeat;
    Bus() {
        const char* home = std::getenv("HOME");
        root = fs::path(home ? home : ".") / "daw_bus";
        inbox = root / "inbox";
        outbox = root / "outbox";
        processing = root / "processing";
        heartbeat = root / "heartbeat.json";
        fs::create_directories(inbox);
        fs::create_directories(outbox);
        fs::create_directories(processing);
    }
};

static void atomic_write(const fs::path& path, const json& j) {
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f(tmp);
        f << j.dump();
    }
    fs::rename(tmp, path);
}

static std::string target_track(const json& req) {
    if (req.contains("target") && req["target"].is_object()
        && req["target"].contains("track")) {
        return req["target"]["track"].get<std::string>();
    }
    return "";
}

static json dispatch(const json& req) {
    try {
        std::string op = req.value("op", "");
        json args = req.value("args", json::object());
        std::string track = target_track(req);

        if (op == "midi.insert_notes") {
            int n = hush::insert_midi_notes(track, args);
            return {{"ok", true}, {"result", {{"inserted", n}}}};
        }
        if (op == "track.set_volume") {
            double g = args.value("gain_db", 0.0);
            hush::set_track_gain(track, g);
            return {{"ok", true}, {"result", {{"track", track}, {"gain_db", g}}}};
        }
        if (op == "track.mute") {
            bool m = args.value("muted", false);
            hush::set_track_mute(track, m);
            return {{"ok", true}, {"result", {{"track", track}, {"muted", m}}}};
        }
        if (op == "forensic.set_params") {
            hush::set_forensic_params(track, args);
            return {{"ok", true}, {"result", {{"track", track}, {"applied", true}}}};
        }
        if (op == "transport.set_tempo") {
            double bpm = args.value("bpm", 120.0);
            hush::set_tempo(bpm);
            return {{"ok", true}, {"result", {{"bpm", bpm}}}};
        }
        if (op == "transport.play") {
            hush::transport_play();
            return {{"ok", true}, {"result", {{"playing", true}}}};
        }
        if (op == "transport.stop") {
            hush::transport_stop();
            return {{"ok", true}, {"result", {{"stopped", true}}}};
        }
        if (op == "session.query") {
            return {{"ok", true}, {"result", hush::query_session()}};
        }
        return {{"ok", false}, {"error", "unknown_op:" + op}};
    } catch (const std::exception& e) {
        return {{"ok", false}, {"error", std::string("exception:") + e.what()}};
    }
}

int main() {
    Bus bus;
    std::thread hb([&]() {
        while (g_run.load()) {
            atomic_write(bus.heartbeat, {{"ts", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}});
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    while (g_run.load()) {
        std::vector<fs::path> pending;
        for (auto& e : fs::directory_iterator(bus.inbox)) {
            if (e.path().extension() == ".json") pending.push_back(e.path());
        }
        if (pending.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        std::sort(pending.begin(), pending.end(), [](const fs::path& a, const fs::path& b) {
            return fs::last_write_time(a) < fs::last_write_time(b);
        });
        fs::path in = pending.front();
        fs::path proc = bus.processing / in.filename();
        try {
            fs::rename(in, proc);
        } catch (...) {
            continue;
        }
        json req;
        try {
            std::ifstream f(proc);
            f >> req;
        } catch (...) {
            fs::remove(proc);
            continue;
        }
        json resp = dispatch(req);
        std::string id = req.value("id", "");
        if (!id.empty()) {
            atomic_write(bus.outbox / (id + ".json"), resp);
        }
        fs::remove(proc);
    }

    hb.join();
    return 0;
}
