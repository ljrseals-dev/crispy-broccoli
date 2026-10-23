"""Soundcraft Ui mixer connection + op dispatch (websocket, socket.io v1)."""

import json
import threading
import time

import websocket  # websocket-client


class UiConnection:
    def __init__(self, ip, port=80):
        self.ip = ip
        self.port = port
        self.ws = None
        self.state = {}
        self._lock = threading.Lock()
        self._running = False
        self._threads = []

    def connect(self):
        url = "ws://%s:%d/socket.io/1/websocket/" % (self.ip, self.port)
        self.ws = websocket.create_connection(url, timeout=10)
        self._running = True
        t1 = threading.Thread(target=self._keepalive, daemon=True)
        t2 = threading.Thread(target=self._recv_loop, daemon=True)
        self._threads = [t1, t2]
        t1.start()
        t2.start()
        return self

    def _keepalive(self):
        while self._running and self.ws is not None:
            try:
                self.ws.send("3:::ALIVE")
            except Exception:
                break
            time.sleep(2.0)

    def _recv_loop(self):
        while self._running and self.ws is not None:
            try:
                raw = self.ws.recv()
            except Exception:
                break
            if not raw:
                continue
            if raw.startswith("3:::"):
                continue
            if raw.startswith("4:::"):
                payload = raw[4:]
                self._ingest(payload)

    def _ingest(self, payload):
        with self._lock:
            for line in payload.splitlines():
                if not line.startswith("SETD^"):
                    continue
                parts = line.split("^", 2)
                if len(parts) < 3:
                    continue
                self.state[parts[1]] = parts[2]

    def send(self, key, value):
        if self.ws is None:
            return
        try:
            self.ws.send("SETD^%s^%s" % (key, value))
        except Exception:
            pass

    def close(self):
        self._running = False
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None


def _filter_channels(state):
    chans = {}
    for k, v in state.items():
        if k.startswith("i."):
            parts = k.split(".")
            if len(parts) >= 2:
                ch = parts[1]
                chans.setdefault(ch, {})[parts[2] if len(parts) > 2 else "value"] = v
    return chans


def dispatch(conn, cmd):
    op = cmd.get("op")
    target = cmd.get("target") or []
    args = cmd.get("args") or {}

    if op == "mixer.set_fader":
        ch = str(target[0]) if target else str(args.get("channel"))
        conn.send("i.%s.mix" % ch, args.get("value"))
        return {"ok": True}

    if op == "mixer.set_mute":
        ch = str(target[0]) if target else str(args.get("channel"))
        conn.send("i.%s.mute" % ch, args.get("value"))
        return {"ok": True}

    if op == "mixer.set_pan":
        ch = str(target[0]) if target else str(args.get("channel"))
        conn.send("i.%s.pan" % ch, args.get("value"))
        return {"ok": True}

    if op == "mixer.set_aux":
        ch = str(target[0]) if target else str(args.get("channel"))
        n = args.get("aux")
        conn.send("i.%s.aux.%s" % (ch, n), args.get("value"))
        return {"ok": True}

    if op == "mixer.query":
        with conn._lock:
            return {"ok": True, "channels": _filter_channels(conn.state)}

    return {"ok": False, "error": "unknown op"}
