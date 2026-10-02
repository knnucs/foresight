"""
Grabs one frame from the webcam (or a video at a chosen second) and uploads it to the
server as the background for drawing zones in the dashboard. Run from ~/acc-ai-mvp:
    python src/snapshot.py                          webcam
    python src/snapshot.py --video file.mp4 --at 5  frame 5 s into a video
"""
import argparse
import sys
import urllib.request

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--video", type=str, default=None)
ap.add_argument("--at", type=float, default=1.0, help="seconds into the video")
ap.add_argument("--server", type=str, default="http://localhost:8000")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video if args.video else args.camera)
if not cap.isOpened():
    sys.exit("Could not open the camera/video" + ("" if args.video else " - try --camera 1"))
if args.video:
    cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000)
else:
    for _ in range(15):
        cap.read()
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit("Could not read a frame")
ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
req = urllib.request.Request(args.server.rstrip("/") + "/api/snapshot", data=jpg.tobytes(),
                             headers={"Content-Type": "image/jpeg"}, method="POST")
urllib.request.urlopen(req, timeout=5).read()
print(f"Snapshot uploaded ({frame.shape[1]}x{frame.shape[0]}). Press 'Refresh snapshot' in the dashboard.")
