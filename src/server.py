"""
Dashboard + API server with a SQLite database. Runs on the ThinkPad; the device (live.py,
later on the Pi) sends events and heartbeats here and reads its zones + mode from here.
Run from ~/acc-ai-mvp:
    python src/server.py                                   dashboard at http://localhost:8000
    TELEGRAM_TOKEN="<bot token>" python src/server.py      + Telegram alerts (message your bot first)
"""
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, request, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zones import ZONE_TYPES

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS_DIR = os.path.join(PROJECT_DIR, "data", "system")
DB_PATH = os.path.join(SYS_DIR, "events.db")
SNAPSHOT = os.path.join(SYS_DIR, "snapshot.jpg")
DASHBOARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
os.makedirs(SYS_DIR, exist_ok=True)

app = Flask(__name__)
_live = {"state": "OFFLINE", "risk": 0.0, "seen": 0.0, "source": "", "pose": None}
_lock = threading.Lock()


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, severity TEXT,
            zone TEXT, detail TEXT, risk REAL, mode TEXT, feedback TEXT, skeleton TEXT)""")
        con.execute("CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)")


def get_setting(key, default):
    with db() as con:
        row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_setting(key, value):
    with db() as con:
        con.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))


init_db()


# ---------------- Telegram alerts (optional: set TELEGRAM_TOKEN to enable) ----------------
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
_tg_last = {}
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]


def tg_chat_id():
    global TG_CHAT
    if TG_CHAT or not TG_TOKEN:
        return TG_CHAT
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", timeout=5) as r:
            updates = json.loads(r.read()).get("result", [])
        for u in reversed(updates):
            chat = (u.get("message") or u.get("channel_post") or {}).get("chat")
            if chat:
                TG_CHAT = str(chat["id"])
                print(f"Telegram: sending alerts to chat {TG_CHAT}", flush=True)
                break
        if not TG_CHAT:
            print("Telegram: no chat found - send your bot a message first", flush=True)
    except Exception as e:
        print("Telegram setup failed:", e, flush=True)
    return TG_CHAT


def skeleton_png(skel):
    """Draws the skeleton alone on a dark background (no camera image)."""
    import cv2
    import numpy as np
    pts = np.array(skel, dtype=np.float32).reshape(33, 4)
    ok = pts[:, 3] > 0.3
    if ok.sum() < 4:
        return None
    x0, x1 = pts[ok, 0].min(), pts[ok, 0].max()
    y0, y1 = pts[ok, 1].min(), pts[ok, 1].max()
    scale = 300 / max(x1 - x0, y1 - y0, 1e-3)
    img = np.full((360, 360, 3), 30, np.uint8)
    P = lambda i: (int(30 + (pts[i, 0] - x0) * scale), int(30 + (pts[i, 1] - y0) * scale))
    for a, b in BONES:
        if ok[a] and ok[b]:
            cv2.line(img, P(a), P(b), (80, 80, 255), 3)
    for i in range(33):
        if ok[i]:
            cv2.circle(img, P(i), 4, (255, 255, 255), -1)
    good, png = cv2.imencode(".png", img)
    return png.tobytes() if good else None


def tg_notify(d, mode):
    import urllib.parse
    import urllib.request
    import uuid
    chat = tg_chat_id()
    if not chat:
        return
    icon = "\U0001F6A8" if d.get("severity") in ("alarm", "danger") else "\u26A0\uFE0F"
    text = (f"{icon} {str(d.get('severity', '')).upper()}: {d.get('detail') or d.get('kind')}\n"
            f"Mode: {mode} | {datetime.now().strftime('%H:%M:%S')}\n"
            f"Open the dashboard to mark it Real or False alarm.")
    base = f"https://api.telegram.org/bot{TG_TOKEN}"
    try:
        photo = skeleton_png(d["skeleton"]) if d.get("skeleton") else None
        if photo:
            boundary = uuid.uuid4().hex
            body = b""
            for k, v in (("chat_id", chat), ("caption", text)):
                body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            body += (f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="skeleton.png"\r\n'
                     f'Content-Type: image/png\r\n\r\n').encode() + photo + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(base + "/sendPhoto", data=body,
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        else:
            req = urllib.request.Request(base + "/sendMessage",
                                         data=urllib.parse.urlencode({"chat_id": chat, "text": text}).encode())
        urllib.request.urlopen(req, timeout=10).read()
        print(f"Telegram: sent '{d.get('detail') or d.get('kind')}'", flush=True)
    except Exception as e:
        print("Telegram send failed:", e, flush=True)


# ---------------- Routes ----------------
@app.get("/")
def index():
    return send_file(DASHBOARD)


@app.get("/api/zone_types")
def zone_types():
    return jsonify({k: {"label": v["label"], "modes": list(v["modes"])} for k, v in ZONE_TYPES.items()})


@app.get("/api/config")
def config():
    return jsonify(mode=get_setting("mode", "home"), zones=get_setting("zones", []),
                   reset_count=get_setting("reset_count", 0),
                   snapshot_count=get_setting("snapshot_count", 0),
                   view=get_setting("view", "angled"))


@app.post("/api/snapshot_request")
def snapshot_request():
    n = get_setting("snapshot_count", 0) + 1
    set_setting("snapshot_count", n)
    return jsonify(ok=True, requested_at=time.time())


@app.get("/api/snapshot_info")
def snapshot_info():
    return jsonify(mtime=os.path.getmtime(SNAPSHOT) if os.path.exists(SNAPSHOT) else 0)


@app.post("/api/reset")
def reset_alarm():
    n = get_setting("reset_count", 0) + 1
    set_setting("reset_count", n)
    return jsonify(ok=True, reset_count=n)


@app.post("/api/mode")
def set_mode():
    mode = (request.get_json(silent=True) or {}).get("mode")
    if mode not in ("home", "work"):
        return jsonify(error="mode must be home or work"), 400
    set_setting("mode", mode)
    return jsonify(ok=True, mode=mode)


@app.post("/api/view")
def set_view():
    view = (request.get_json(silent=True) or {}).get("view")
    if view not in ("angled", "top"):
        return jsonify(error="view must be angled or top"), 400
    set_setting("view", view)
    return jsonify(ok=True, view=view)


@app.post("/api/zones")
def save_zones():
    clean = []
    for z in (request.get_json(silent=True) or {}).get("zones", []):
        try:
            box = [min(max(float(v), 0.0), 1.0) for v in z["box"]]
        except (KeyError, TypeError, ValueError):
            continue
        if z.get("type") not in ZONE_TYPES or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        clean.append({"id": str(z.get("id", time.time()))[:40],
                      "name": str(z.get("name") or ZONE_TYPES[z["type"]]["label"])[:60],
                      "type": z["type"], "box": box})
    set_setting("zones", clean)
    return jsonify(ok=True, zones=clean)


@app.post("/api/heartbeat")
def heartbeat():
    d = request.get_json(silent=True) or {}
    with _lock:
        _live.update(state=str(d.get("state", "NORMAL")), risk=float(d.get("risk", 0.0)),
                     seen=time.time(), source=str(d.get("source", "")), pose=d.get("pose"))
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    with _lock:
        s = dict(_live)
    online = time.time() - s["seen"] < 5
    return jsonify(state=s["state"] if online else "OFFLINE", risk=s["risk"] if online else 0.0,
                   source=s["source"], online=online, mode=get_setting("mode", "home"),
                   pose=s.get("pose") if online else None)


@app.post("/api/event")
def add_event():
    d = request.get_json(silent=True) or {}
    skeleton = d.get("skeleton")
    mode_now = get_setting("mode", "home")
    with db() as con:
        cur = con.execute(
            "INSERT INTO events(ts, kind, severity, zone, detail, risk, mode, skeleton) VALUES(?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), d.get("kind"), d.get("severity"), d.get("zone"),
             d.get("detail"), d.get("risk"), mode_now,
             json.dumps(skeleton) if skeleton is not None else None))
    if TG_TOKEN and d.get("severity") in ("warning", "alarm", "danger"):
        key, now = (d.get("kind"), d.get("zone")), time.time()
        if now - _tg_last.get(key, 0) >= 5:
            _tg_last[key] = now
            threading.Thread(target=tg_notify, args=(d, mode_now), daemon=True).start()
    return jsonify(id=cur.lastrowid)


@app.get("/api/events")
def list_events():
    limit = min(int(request.args.get("limit", 50)), 500)
    with db() as con:
        rows = con.execute("SELECT id, ts, kind, severity, zone, detail, risk, mode, feedback "
                           "FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/events/<int:eid>/feedback")
def feedback(eid):
    label = (request.get_json(silent=True) or {}).get("label")
    if label not in ("real", "false"):
        return jsonify(error="label must be real or false"), 400
    with db() as con:
        con.execute("UPDATE events SET feedback=? WHERE id=?", (label, eid))
    return jsonify(ok=True)


@app.get("/api/stats")
def stats():
    today = datetime.now().date().isoformat()
    with db() as con:
        q = lambda sql, *a: con.execute(sql, a).fetchone()[0]
        return jsonify(total=q("SELECT COUNT(*) FROM events"),
                       today=q("SELECT COUNT(*) FROM events WHERE ts LIKE ?", today + "%"),
                       real=q("SELECT COUNT(*) FROM events WHERE feedback='real'"),
                       false=q("SELECT COUNT(*) FROM events WHERE feedback='false'"))


@app.get("/api/snapshot")
def get_snapshot():
    if not os.path.exists(SNAPSHOT):
        return jsonify(error="no snapshot yet"), 404
    return send_file(SNAPSHOT, mimetype="image/jpeg", max_age=0)


@app.post("/api/snapshot")
def put_snapshot():
    with open(SNAPSHOT, "wb") as f:
        f.write(request.get_data())
    return jsonify(ok=True)


if __name__ == "__main__":
    print("Dashboard: http://localhost:8000")
    print("Telegram alerts: " + ("ON" if TG_TOKEN else "off (set TELEGRAM_TOKEN to enable)"))
    app.run(host="0.0.0.0", port=8000, threaded=True)
