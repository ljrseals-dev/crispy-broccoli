#!/usr/bin/env python3
"""
Voice background daemon.
Owns mic, LLM client, TTS. Exposes a Unix socket at ~/ai-voice/ctl.sock.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
SOCK = BASE / "ctl.sock"
PIDFILE = BASE / "daemon.pid"
LOG = BASE / "logs" / "voice.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG = {
    "sample_rate": 16000,
    "record_seconds": 5,
    "whisper_model": "base",
    "whisper_language": "en",
    "llm": {
        "url": "http://127.0.0.1:8080/v1/chat/completions",
        "model": "local",
        "system_prompt": "You are a helpful, concise voice assistant. Keep answers under 3 sentences unless asked for detail.",
        "max_tokens": 200,
        "temperature": 0.7,
    },
    "tts": {
        "voice": "en_US-lessac-medium",
        "prefer_termux_tts": False,
    },
}
CFG_PATH = BASE / "config.json"
try:
    with open(CFG_PATH) as f:
        loaded = json.load(f)
        CFG = {**DEFAULT_CFG, **loaded}
        if "llm" in loaded:
            CFG["llm"] = {**DEFAULT_CFG["llm"], **loaded["llm"]}
        if "tts" in loaded:
            CFG["tts"] = {**DEFAULT_CFG["tts"], **loaded["tts"]}
except FileNotFoundError:
    CFG = DEFAULT_CFG

ON_TERMUX = os.path.exists("/data/data/com.termux")
VOICES_DIR = BASE / "voices"

_stop = threading.Event()
_worker_lock = threading.Lock()
_history = []
START_TIME = time.time()


def log(role, text):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{role}\t{text}\n")


# ── Mic ───────────────────────────────────────────────────────────────────
def record(seconds=5):
    """Record via termux-microphone-record (Termux) or return None."""
    if not ON_TERMUX or not shutil.which("termux-microphone-record"):
        return None
    path = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["termux-microphone-record", "-f", path, "-l", str(seconds)],
        check=False,
        capture_output=True,
    )
    last = -1
    for _ in range(int((seconds + 2) * 10)):
        time.sleep(0.1)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > 1000 and size == last:
            break
        last = size
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    try:
        os.unlink(path)
    except OSError:
        pass
    return None


def transcribe(wav_path):
    try:
        import whisper

        model = whisper.load_model(CFG.get("whisper_model", "base"))
        r = model.transcribe(
            wav_path,
            language=CFG.get("whisper_language") or None,
            fp16=False,
        )
        return r.get("text", "").strip()
    except Exception as e:
        log("ERROR", f"transcribe: {e}")
        return ""


# ── LLM ───────────────────────────────────────────────────────────────────
def ask_llm(text, history=None):
    llm = CFG["llm"]
    messages = [{"role": "system", "content": llm["system_prompt"]}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": text})
    try:
        r = requests.post(
            llm["url"],
            json={
                "model": llm.get("model", "local"),
                "messages": messages,
                "max_tokens": llm.get("max_tokens", 200),
                "temperature": llm.get("temperature", 0.7),
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "[LLM offline — start llama-server or Ollama first]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ── TTS ───────────────────────────────────────────────────────────────────
def speak(text):
    if not text:
        return
    if text.startswith("["):
        log("ERROR", text)
        return
    # Strip markdown code fences before speaking
    spoken = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip() or text
    spoken = re.sub(r"[*_`#]", "", spoken).strip() or text

    # Prefer Termux TTS on Android
    if ON_TERMUX and shutil.which("termux-tts-speak"):
        subprocess.run(["termux-tts-speak", spoken], check=False, timeout=60)
        return
    if CFG["tts"].get("prefer_termux_tts") and shutil.which("termux-tts-speak"):
        subprocess.run(["termux-tts-speak", spoken], check=False, timeout=60)
        return

    onnx = VOICES_DIR / f"{CFG['tts']['voice']}.onnx"
    if not onnx.exists():
        if shutil.which("termux-tts-speak"):
            subprocess.run(["termux-tts-speak", spoken], check=False)
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    try:
        cmd = (["piper"] if shutil.which("piper") else ["python3", "-m", "piper"])
        cmd += ["--model", str(onnx), "--output_file", wav]
        p = subprocess.run(
            cmd, input=spoken.encode(), capture_output=True, timeout=60
        )
        if p.returncode != 0:
            if shutil.which("termux-tts-speak"):
                subprocess.run(["termux-tts-speak", spoken], check=False)
            return
        for player in (
            ["termux-media-player", "play", wav],
            ["aplay", wav],
            ["paplay", wav],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav],
            ["play", wav],
        ):
            if shutil.which(player[0]):
                subprocess.run(player, check=False)
                return
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


# ── Command handlers ──────────────────────────────────────────────────────
def handle(req):
    global _history
    cmd = req.get("cmd")

    if cmd == "listen":
        with _worker_lock:
            wav = record(req.get("seconds", CFG.get("record_seconds", 5)))
            if not wav:
                return {"ok": False, "error": "record_failed"}
            try:
                text = transcribe(wav)
            finally:
                try:
                    os.unlink(wav)
                except OSError:
                    pass
            if not text:
                return {"ok": True, "event": "transcript", "text": ""}
            log("USER", text)
            return {"ok": True, "event": "transcript", "text": text}

    if cmd == "ask":
        text = (req.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "empty_text"}
        with _worker_lock:
            reply = ask_llm(text, _history)
            _history.append({"role": "user", "content": text})
            _history.append({"role": "assistant", "content": reply})
            _history = _history[-20:]
            log("AI", reply)
            return {"ok": True, "event": "reply", "text": reply}

    if cmd == "say":
        text = req.get("text", "")
        with _worker_lock:
            speak(text)
        return {"ok": True, "event": "spoken"}

    if cmd == "status":
        return {
            "ok": True,
            "event": "status",
            "pid": os.getpid(),
            "uptime": int(time.time() - START_TIME),
            "termux": ON_TERMUX,
        }

    if cmd == "stop":
        _stop.set()
        return {"ok": True, "event": "stopping"}

    return {"ok": False, "error": f"unknown_cmd:{cmd}"}


# ── Server ────────────────────────────────────────────────────────────────
def serve():
    if SOCK.exists():
        try:
            SOCK.unlink()
        except OSError:
            pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK))
    os.chmod(SOCK, 0o600)
    srv.listen(4)
    srv.settimeout(0.5)

    while not _stop.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        with conn:
            f = conn.makefile("rwb")
            try:
                line = f.readline()
                if not line:
                    continue
                req = json.loads(line.decode())
                resp = handle(req)
                f.write((json.dumps(resp) + "\n").encode())
                f.flush()
            except Exception as e:
                try:
                    f.write(
                        (json.dumps({"ok": False, "error": str(e)}) + "\n").encode()
                    )
                    f.flush()
                except OSError:
                    pass
    srv.close()
    try:
        SOCK.unlink()
    except OSError:
        pass


def acquire_wake_lock():
    if ON_TERMUX and shutil.which("termux-wake-lock"):
        subprocess.run(["termux-wake-lock"], check=False)
    if ON_TERMUX and shutil.which("termux-notification"):
        subprocess.run(
            [
                "termux-notification",
                "--id",
                "ai-voice-daemon",
                "--title",
                "AI Voice",
                "--content",
                "Voice daemon running",
                "--ongoing",
            ],
            check=False,
        )


def release_wake_lock():
    if ON_TERMUX and shutil.which("termux-wake-unlock"):
        subprocess.run(["termux-wake-unlock"], check=False)
    if ON_TERMUX and shutil.which("termux-notification-remove"):
        subprocess.run(
            ["termux-notification-remove", "--id", "ai-voice-daemon"], check=False
        )


def main():
    if PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text().strip())
            os.kill(old, 0)
            print(f"already running (pid {old})", file=sys.stderr)
            sys.exit(1)
        except (ValueError, ProcessLookupError, PermissionError):
            try:
                PIDFILE.unlink()
            except OSError:
                pass

    PIDFILE.write_text(str(os.getpid()))
    acquire_wake_lock()

    def on_sig(sig, frame):
        _stop.set()

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    log("DAEMON", f"start pid={os.getpid()} termux={ON_TERMUX}")
    try:
        serve()
    finally:
        release_wake_lock()
        try:
            PIDFILE.unlink()
        except OSError:
            pass
        log("DAEMON", "stop")


if __name__ == "__main__":
    main()
