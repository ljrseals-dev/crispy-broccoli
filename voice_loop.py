#!/usr/bin/env python3
"""Foreground voice loop backed by voice-daemon."""
import sys
import time
from pathlib import Path

# Ensure we can import voice_client from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_client as vc


def main():
    st = vc.status()
    if not st.get("ok"):
        print(
            "Daemon not running. Start it with:\n  bash ~/ai-voice/ai-voice.sh daemon",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Connected to daemon pid={st.get('pid')} uptime={st.get('uptime')}s")
    print("Press Enter to speak. Ctrl+C to quit.\n")

    while True:
        try:
            input("Press Enter to speak… ")
            tr = vc.listen(5)
            if not tr.get("ok"):
                print("listen error:", tr.get("error"))
                continue
            text = tr.get("text", "")
            if not text:
                print("(nothing heard)")
                continue
            print(f"You: {text}")
            rep = vc.ask(text)
            if not rep.get("ok"):
                print("ask error:", rep.get("error"))
                continue
            print(f"AI:  {rep.get('text', '')}")
            vc.say(rep.get("text", ""))
        except KeyboardInterrupt:
            print("\nBye.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
