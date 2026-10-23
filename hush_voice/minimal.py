#!/usr/bin/env python3
"""
Minimal voice loop for Termux — no Whisper/Piper required.
Uses termux-microphone-record + termux-tts-speak + local LLM.
Best for low-RAM devices or when the full stack fails.
"""

import subprocess
import tempfile
import os
import time
import requests

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
# For Ollama change to: "http://127.0.0.1:11434/v1/chat/completions"

def record(seconds=5):
    path = tempfile.mktemp(suffix=".wav")
    print(f"🎤 Recording {seconds}s…")
    subprocess.run(["termux-microphone-record", "-f", path, "-l", str(seconds)], check=False)
    time.sleep(seconds + 0.8)
    subprocess.run(["termux-microphone-record", "-q"], check=False)
    return path if os.path.exists(path) else None

def transcribe_placeholder(path):
    """
    Placeholder — without Whisper we cannot transcribe.
    For a pure offline minimal path you still need a local STT
    (whisper.cpp binary or Vosk). Here we just return a note.
    """
    print("(No local STT binary found — type your text instead)")
    return input("You: ").strip()

def ask(text):
    try:
        r = requests.post(
            LLM_URL,
            json={
                "model": "local",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 150,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error: {e}]"

def speak(text):
    subprocess.run(["termux-tts-speak", text], check=False)

if __name__ == "__main__":
    print("Minimal AI Voice (Termux TTS + local LLM)")
    print("Make sure llama-server or Ollama is running.")
    print("Ctrl+C to quit.\n")
    while True:
        try:
            input("Press Enter to speak (or just type after)…")
            wav = record(5)
            text = transcribe_placeholder(wav)
            if not text:
                continue
            print(f"You: {text}")
            reply = ask(text)
            print(f"AI: {reply}")
            speak(reply)
        except KeyboardInterrupt:
            print("\nBye.")
            break
