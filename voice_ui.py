#!/usr/bin/env python3
"""Voice control for a Soundcraft Ui mixer: STT -> LLM (JSON intent) -> dispatch -> TTS."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

import requests

from ui_conn import UiConnection, dispatch

MIXER_IP = os.environ.get("MIXER_IP", "192.168.1.50")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "local")

SYSTEM = (
    "You control a Soundcraft Ui mixer. Respond with ONLY a JSON object "
    "with keys op, target, args. Allowed ops: "
    "mixer.set_fader (target [channel], args {value}), "
    "mixer.set_mute (target [channel], args {value: 0|1}), "
    "mixer.set_pan (target [channel], args {value: -100..100}), "
    "mixer.set_aux (target [channel], args {aux: int, value: 0..1}), "
    "mixer.query (target [], args {}). "
    "No prose. No code fences."
)

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def record(seconds=5):
    fd, path = tempfile.mkstemp(prefix="vu_", suffix=".m4a")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    try:
        subprocess.run(
            ["termux-microphone-record", "-f", path, "-l", str(int(seconds))],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    last = -1
    stable = 0
    deadline = time.time() + seconds + 10
    while time.time() < deadline:
        time.sleep(0.1)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > 1000 and size == last:
            stable += 1
            if stable >= 3:
                return path
        else:
            stable = 0
        last = size
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    return None


def ask_llm(text):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        "max_tokens": 200,
        "temperature": 0.2,
    }
    try:
        r = requests.post(LLM_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "[LLM offline]"
    except Exception as exc:
        return "[LLM error: %s]" % exc


def extract_json(raw):
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except Exception:
        return None


def speak(text):
    clean = FENCE_RE.sub("", text or "").strip()
    if not clean:
        return
    if clean.startswith("[") and clean.endswith("]"):
        print(clean)
        return
    try:
        subprocess.run(["termux-tts-speak", clean], check=False)
    except FileNotFoundError:
        print(clean)


def main():
    conn = UiConnection(MIXER_IP).connect()
    try:
        while True:
            input("Press Enter to speak…")
            wav = record(5)
            if not wav:
                print("[no recording]")
                continue
            text = input("You said: ").strip()  # placeholder STT
            try:
                os.unlink(wav)
            except OSError:
                pass
            if not text:
                continue
            raw = ask_llm(text)
            cmd = extract_json(raw)
            if not cmd:
                print("[no valid command] raw=%r" % raw)
                speak("Sorry, I didn't get a valid command.")
                continue
            result = dispatch(conn, cmd)
            print("cmd=%s result=%s" % (cmd.get("op"), result))
            if result.get("ok"):
                speak("Done: %s" % cmd.get("op", "command"))
            else:
                speak("Failed: %s" % result.get("error", "unknown"))
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
