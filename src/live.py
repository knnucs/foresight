"""
Device program: camera or video -> MediaPipe pose -> fall model + danger zones ->
warnings / alarms (sounds) -> events and live status sent to the server / dashboard.
Run from ~/acc-ai-mvp (start python src/server.py first for the dashboard):
    python src/live.py                          laptop webcam
    python src/live.py --camera 1               another (USB) webcam
    python src/live.py --video "file.mp4"       a recorded clip
    python src/live.py --no-display             no video window (needed on this ThinkPad)
    python src/live.py --save out.avi           also write the annotated video
    python src/live.py --server http://IP:8000  server on another machine (e.g. from the Pi)
Keys (only with a video window): q quit, r reset, + / - warning threshold. Ctrl+C also stops it.
"""
import argparse
import csv
import http.server
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fallnet import FallTCN, fill_missing, window_features
from alert import AlertSystem
from zones import ZoneMonitor, ZONE_TYPES

RESULTS_DIR = "data/oops/results"
LOG_DIR = "data/live_logs"
POSE_MODEL = "pose_landmarker.task"
INCIDENT_SECONDS = 10.0

BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]
COLORS = {"NORMAL": (80, 200, 80), "WARNING": (0, 165, 255), "ALARM": (0, 0, 255)}
STATE_RANK = {"NORMAL": 0, "WARNING": 1, "ALARM": 2}


def make_tone(path, freqs, seconds_each, volume):
    rate = 22050
    parts = []
    for f in freqs:
        t = np.arange(int(rate * seconds_each)) / rate
        env = np.minimum(1.0, np.minimum(t, t[::-1]) * 50)
        parts.append(np.sin(2 * np.pi * f * t) * env * volume)
    data = (np.concatenate(parts) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def play(path):
    if platform.system() == "Darwin" and shutil.which("afplay"):
        cmd = ["afplay", path]
    elif shutil.which("aplay"):
        cmd = ["aplay", "-q", path]
    else:
        print("\a", end="", flush=True)
        return
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ServerLink:
    """Talks to server.py in background threads, so a slow or missing server never blocks the camera loop."""
    def __init__(self, base_url, source):
        self.base = base_url.rstrip("/") if base_url else None
        self.source = source
        self.queue = queue.Queue(maxsize=500)
        self.config = {"mode": "home", "zones": [], "reset_count": 0}
        self.config_version = 0
        self.online = None
        self._last_hb = 0.0
        if not self.base:
            return
        try:
            self.config = self._request("/api/config")
            self.config_version = 1
            self._set_online(True)
        except Exception:
            self._set_online(False)
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._config_loop, daemon=True).start()

    def _request(self, path, payload=None):
        if isinstance(payload, bytes):
            data, ctype = payload, "image/jpeg"
        else:
            data, ctype = (None if payload is None else json.dumps(payload).encode()), "application/json"
        req = urllib.request.Request(self.base + path, data=data, method="GET" if data is None else "POST",
                                     headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read() or b"{}")

    def _set_online(self, ok):
        if ok != self.online:
            print("Server connected." if ok else f"Server not reachable at {self.base} - continuing offline.", flush=True)
            self.online = ok

    def _worker(self):
        while True:
            path, payload = self.queue.get()
            try:
                self._request(path, payload)
                self._set_online(True)
            except Exception:
                self._set_online(False)
            finally:
                self.queue.task_done()

    def _config_loop(self):
        while True:
            time.sleep(3)
            try:
                cfg = self._request("/api/config")
                if cfg != self.config:
                    self.config = cfg
                    self.config_version += 1
                self._set_online(True)
            except Exception:
                self._set_online(False)

    def _put(self, path, payload):
        if not self.base:
            return
        try:
            self.queue.put_nowait((path, payload))
        except queue.Full:
            pass

    def heartbeat(self, state, risk):
        now = time.monotonic()
        if now - self._last_hb >= 0.5:
            self._last_hb = now
            self._put("/api/heartbeat", {"state": state, "risk": round(float(risk), 3), "source": self.source})

    def event(self, payload):
        self._put("/api/event", payload)

    def snapshot(self, jpg_bytes):
        self._put("/api/snapshot", jpg_bytes)

    def flush(self, timeout=3.0):
        end = time.monotonic() + timeout
        while self.base and self.queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.05)


class PreviewServer:
    """Developer preview: the annotated camera view as a live MJPEG stream, reachable from this machine only."""
    def __init__(self, port):
        self.jpg = None
        self.lock = threading.Lock()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with owner.lock:
                            jpg = owner.jpg
                        if jpg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.07)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"Live preview: open http://localhost:{port} in the browser (this machine only)", flush=True)

    def update(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self.jpg = jpg.tobytes()


def draw(frame, pose, state, risk, thr, fps_now, note, zones, mode):
    h, w = frame.shape[:2]
    color = COLORS[state]
    for z in zones:
        if mode not in ZONE_TYPES.get(z["type"], {}).get("modes", {}):
            continue
        x1, y1, x2, y2 = [int(v * s) for v, s in zip(z["box"], (w, h, w, h))]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 230), 2)
        cv2.putText(frame, z["name"], (x1 + 4, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 1)
    if pose is not None:
        lm = pose.reshape(33, 4)
        pts = [(int(x * w), int(y * h)) for x, y in lm[:, :2]]
        for a, b in BONES:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for i in set(j for bone in BONES for j in bone):
            cv2.circle(frame, pts[i], 3, (255, 255, 255), -1)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    bar_w = w - 40
    cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 20), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 40), (20 + int(bar_w * min(max(risk, 0.0), 1.0)), h - 20), color, -1)
    tx = 20 + int(bar_w * thr)
    cv2.line(frame, (tx, h - 46), (tx, h - 14), (255, 255, 255), 2)
    cv2.putText(frame, f"{state}   fall risk {risk:.2f}   threshold {thr:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"{mode.upper()} mode   {fps_now:.0f} fps   {note}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--fast", action="store_true", help="with --video: run as fast as possible")
    ap.add_argument("--save", type=str, default=None, help="write the annotated video to this .avi file")
    ap.add_argument("--server", type=str, default="http://localhost:8000")
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--preview", type=int, nargs="?", const=8001, default=None,
                    help="developer view of the annotated camera at http://localhost:8001")
    args = ap.parse_args()

    source = args.video if args.video else args.camera
    source_name = os.path.basename(args.video) if args.video else f"camera {args.camera}"
    link = ServerLink(None if args.no_server else args.server, source_name)

    with open(os.path.join(RESULTS_DIR, "config.json")) as f:
        cfg = json.load(f)
    fps, window = cfg["fps"], cfg["window"]
    thr = args.threshold if args.threshold is not None else cfg.get("live_threshold", cfg["threshold"])

    model = FallTCN()
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "tcn_oops.pt"), map_location="cpu"))
    model.eval()

    os.makedirs(LOG_DIR, exist_ok=True)
    chime, siren = os.path.join(LOG_DIR, "chime.wav"), os.path.join(LOG_DIR, "alarm.wav")
    make_tone(chime, [880, 1320], 0.15, 0.3)
    make_tone(siren, [1000, 700] * 4, 0.2, 0.8)
    log_path = os.path.join(LOG_DIR, "device_events.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["wall_time", "stream_time_s", "kind", "severity", "detail", "risk", "source"])

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open {source_name}" + ("" if args.video else " - try --camera 1"))
    src_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 0.0
    if args.video and (not src_fps or src_fps <= 1 or src_fps > 240):
        src_fps = 30.0

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL,
                                           delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.6, min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6)
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    alert = AlertSystem(thr)
    zone_mon = ZoneMonitor()
    seen_cfg, seen_reset, seen_snap, snap_pending = -1, None, None, False
    buf = deque(maxlen=window)
    incident = deque()
    step = 1.0 / fps
    next_t, frame_idx = 0.0, -1
    wall0 = time.monotonic()
    last_proc, fps_now = None, 0.0
    zone_flash_state, zone_flash_until = "NORMAL", -1.0
    note = "watching"
    writer = None
    preview = PreviewServer(args.preview) if args.preview else None

    def report(t, kind, severity, detail, risk, pose, zone=None):
        log.writerow([datetime.now().isoformat(timespec="seconds"), round(t, 2), kind, severity,
                      detail, round(risk, 3), source_name])
        log_f.flush()
        link.event({"kind": kind, "severity": severity, "detail": detail, "zone": zone,
                    "risk": round(float(risk), 3),
                    "skeleton": None if pose is None else [round(float(v), 4) for v in pose]})
        print(f"[{t:7.2f}s] {kind:<15} {severity:<8} {detail}", flush=True)

    print(f"Running on {source_name} with fall warning threshold {thr:.2f}. Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t = frame_idx / src_fps if args.video else time.monotonic() - wall0
            if t + 1e-6 < next_t:
                continue
            while next_t <= t + 1e-6:
                next_t += step

            if link.config_version != seen_cfg:
                seen_cfg = link.config_version
                zone_mon.set_zones(link.config.get("zones", []), link.config.get("mode", "home"),
                                  link.config.get("view", "angled"))
                rc = link.config.get("reset_count", 0)
                if seen_reset is None:
                    seen_reset = rc
                elif rc != seen_reset:
                    seen_reset = rc
                    alert.reset()
                    buf.clear()
                    zone_flash_until = -1.0
                    note = "reset from dashboard"
                    print(f"[{t:7.2f}s] reset from dashboard", flush=True)
                sc = link.config.get("snapshot_count", 0)
                if seen_snap is None:
                    seen_snap = sc
                elif sc != seen_snap:
                    seen_snap = sc
                    snap_pending = True
                print(f"Mode: {zone_mon.mode}, {len(zone_mon.zones)} zone(s) loaded", flush=True)

            if snap_pending:
                ok_jpg, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_jpg:
                    link.snapshot(jpg.tobytes())
                    print(f"[{t:7.2f}s] snapshot sent to dashboard", flush=True)
                snap_pending = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            pose = None
            if res.pose_landmarks:
                pose = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]],
                                dtype=np.float32).reshape(-1)
                if pose.reshape(33, 4)[[11, 12, 23, 24], 3].mean() < 0.5:
                    pose = None
            buf.append((pose if pose is not None else np.zeros(132, np.float32), pose is not None))
            incident.append((t, buf[-1][0], pose is not None))
            while incident and t - incident[0][0] > INCIDENT_SECONDS:
                incident.popleft()

            risk_raw = 0.0
            if len(buf) == window and any(d for _, d in buf):
                poses = np.stack([p for p, _ in buf])
                det = np.array([d for _, d in buf])
                x = window_features(fill_missing(poses, det), det)
                with torch.no_grad():
                    risk_raw = float(torch.sigmoid(model(torch.from_numpy(x)[None]))[0])
            h, w = frame.shape[:2]
            risk, events = alert.update(t, risk_raw, pose, aspect=w / h)

            for ev in events:
                if ev == "warning":
                    play(chime)
                    note = f"fall warning at {t:.1f}s"
                    report(t, "fall_warning", "warning", "Fall risk rising", risk, pose)
                elif ev == "prevented":
                    note = f"risk passed at {t:.1f}s"
                    report(t, "fall_prevented", "info", "Risk passed - person back on their feet", risk, pose)
                elif ev == "cleared_unsure":
                    note = f"check on person ({t:.1f}s)"
                    report(t, "fall_check", "warning", "Warning ended but person not clearly upright - check on them", risk, pose)
                elif ev == "recovered":
                    note = f"recovered at {t:.1f}s"
                    report(t, "fall_recovered", "info", "Person stood back up for 10 s - alarm cleared", risk, pose)
                elif ev == "alarm":
                    play(siren)
                    near = zone_mon.zones_near(pose)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    np.savez_compressed(os.path.join(LOG_DIR, f"incident_{stamp}.npz"),
                                        times=np.array([i[0] for i in incident], np.float32),
                                        poses=np.stack([i[1] for i in incident]),
                                        detected=np.array([i[2] for i in incident]))
                    detail = "Fall confirmed" + (" near " + ", ".join(near) if near else "")
                    note = detail
                    report(t, "fall_alarm", "alarm", detail, risk, pose, zone=", ".join(near) or None)

            for zev in zone_mon.update(t, pose):
                danger = zev["severity"] == "danger"
                play(siren if danger else chime)
                new_state = "ALARM" if danger else "WARNING"
                if t >= zone_flash_until or STATE_RANK[new_state] > STATE_RANK[zone_flash_state]:
                    zone_flash_state = new_state
                zone_flash_until = t + 3.0
                note = zev["detail"]
                report(t, "zone", zev["severity"], zev["detail"], risk, pose, zone=zev["zone"])

            shown = alert.state
            if t < zone_flash_until and STATE_RANK[zone_flash_state] > STATE_RANK[shown]:
                shown = zone_flash_state
            link.heartbeat(shown, risk)

            now = time.monotonic()
            if last_proc is not None:
                fps_now = 0.9 * fps_now + 0.1 / max(now - last_proc, 1e-3)
            last_proc = now

            if not args.no_display or args.save or preview:
                draw(frame, pose, shown, risk, alert.threshold, fps_now, note, zone_mon.zones, zone_mon.mode)
            if preview:
                preview.update(frame)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Accident prediction", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    alert.reset()
                    buf.clear()
                    zone_flash_until = -1.0
                    note = "reset"
                if key in (ord("+"), ord("=")):
                    alert.set_threshold(alert.threshold + 0.02)
                if key in (ord("-"), ord("_")):
                    alert.set_threshold(alert.threshold - 0.02)
            if args.video and not args.fast:
                lag = t - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        pass
    finally:
        landmarker.close()
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video to {args.save}")
        log_f.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        link.flush()
    print(f"Stopped. Events logged to {log_path}")


if __name__ == "__main__":
    main()
