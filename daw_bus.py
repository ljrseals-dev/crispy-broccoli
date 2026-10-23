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
