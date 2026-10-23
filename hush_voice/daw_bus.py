"""Host-agnostic DAW command bus (filesystem based).

Other processes write ops to ~/daw_bus/inbox and read responses from
~/daw_bus/outbox.  The DAW-side executor polls inbox and replies.
"""

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

BUS = Path.home() / "daw_bus"
INBOX = BUS / "inbox"
OUTBOX = BUS / "outbox"
PROCESSING = BUS / "processing"
for _d in (INBOX, OUTBOX, PROCESSING):
    _d.mkdir(parents=True, exist_ok=True)


def _atomic(path, obj):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        os.write(fd, json.dumps(obj).encode("utf-8"))
        os.close(fd)
        os.replace(tmp, path)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def send(op, target=None, args=None, timeout=5.0):
    req_id = str(uuid.uuid4())
    msg = {
        "id": req_id,
        "op": op,
        "target": target,
        "args": args or {},
        "ts": time.time(),
    }
    inbox_path = INBOX / (req_id + ".json")
    _atomic(str(inbox_path), msg)
    out_path = OUTBOX / (req_id + ".json")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if out_path.exists():
            try:
                with open(out_path) as f:
                    resp = json.load(f)
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
                return resp
            except Exception:
                pass
        time.sleep(0.05)
    return {"ok": False, "error": "timeout"}


def alive(max_age=2.0):
    hb = BUS / "heartbeat"
    try:
        return (time.time() - hb.stat().st_mtime) <= max_age
    except FileNotFoundError:
        return False


def cli_main(argv=None):
    """Entry point for aoc-daw console script."""
    import sys, json
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: aoc-daw [--alive|--list|<op> [json-args]]")
        print("  --alive   check executor heartbeat")
        print("  --list    list canonical ops")
        print("  <op>      send op (e.g. session.query)")
        return 0
    if argv[0] == "--alive":
        print("alive" if alive() else "not alive")
        return 0
    if argv[0] == "--list":
        try:
            from hush_voice.ops import OPS
            for k in OPS:
                print(k)
        except Exception:
            print("session.query\ntransport.play\ntransport.stop\ntrack.set_volume\ntrack.mute")
        return 0
    op = argv[0]
    args = {}
    target = None
    if len(argv) > 1:
        try:
            extra = json.loads(argv[1])
            if isinstance(extra, dict):
                args = extra.get("args", extra)
                target = extra.get("target")
        except Exception:
            pass
    result = send(op, target=target, args=args)
    print(json.dumps(result))
    return 0 if result.get("ok") else 1

if __name__ == "__main__":
    import sys
    sys.exit(cli_main())
