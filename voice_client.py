#!/usr/bin/env python3
"""Thin client for voice-daemon. Used by voice-loop.py and hotkeys."""
import json
import socket
import sys
from pathlib import Path

SOCK = Path(__file__).resolve().parent / "ctl.sock"


def call(req, timeout=180):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(SOCK))
        s.sendall((json.dumps(req) + "\n").encode())
        f = s.makefile("rb")
        line = f.readline()
        if not line:
            return {"ok": False, "error": "no_response"}
        return json.loads(line.decode())
    except FileNotFoundError:
        return {"ok": False, "error": "daemon_not_running"}
    except ConnectionRefusedError:
        return {"ok": False, "error": "daemon_not_running"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            s.close()
        except OSError:
            pass


def listen(seconds=5):
    return call({"cmd": "listen", "seconds": seconds})


def ask(text):
    return call({"cmd": "ask", "text": text})


def say(text):
    return call({"cmd": "say", "text": text})


def status():
    return call({"cmd": "status"}, timeout=5)


def stop():
    return call({"cmd": "stop"}, timeout=5)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps(status()))
        sys.exit(0)
    verb = sys.argv[1]
    if verb == "listen":
        sec = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(json.dumps(listen(sec)))
    elif verb == "ask":
        print(json.dumps(ask(" ".join(sys.argv[2:]))))
    elif verb == "say":
        print(json.dumps(say(" ".join(sys.argv[2:]))))
    elif verb == "status":
        print(json.dumps(status()))
    elif verb == "stop":
        print(json.dumps(stop()))
    else:
        print("usage: voice_client.py {listen|ask|say|status|stop} [...]", file=sys.stderr)
        sys.exit(1)
