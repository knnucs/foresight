"""
Danger zones, approach / reach prediction and motionless-person detection, all from the skeleton.
Shared by live.py (the engine) and server.py (the list of zone types).
Boxes are normalized 0-1 image coordinates [x1, y1, x2, y2], y pointing down.

Depth: one camera cannot see depth, so a box on the image would also "contain" a hand held right
in front of the lens. We use the floor instead. With an angled camera (high in a corner), where a
person's feet touch the ground tells how far away they are; draw each box down to the floor where
the hazard stands. With a top-down camera the image is already a floor map, so the body's center
(hips) is used as the floor position. Falls are detected best with the angled view.

Rule kinds:
  hand       a hand is inside the box AND the person stands at the zone
  body       the person's floor position is inside the zone
  above_top  hips rise above the box's top edge while at the zone   (climbing out of a crib)
  feet_on    feet are up on the box, off the floor                  (climbing furniture / a sill)
Predictions (warn BEFORE the rule fires):
  approach   moving toward the zone, will reach it within APPROACH_S seconds
  reach      standing at a hand-hazard zone, a hand will touch it within REACH_S seconds
Independent of zones: alarm when a person lies down without moving, or stays still in water.
Severity: "warning" = soft chime, "danger" = loud alarm sound.
"""
import math
from collections import deque

ZONE_TYPES = {
    "stove": {"label": "Stove / oven / hot surface", "approach": {"home": "warning"}, "modes": {
        "home": [("hand", 0.3, "danger"), ("body", 1.5, "warning")],
        "work": [("hand", 1.0, "warning")]}},
    "electrical": {"label": "Socket / electrical panel / wiring", "approach": {"home": "warning"}, "modes": {
        "home": [("hand", 0.3, "danger"), ("body", 2.0, "warning")],
        "work": [("hand", 0.5, "danger")]}},
    "water": {"label": "Bathtub / pool / water", "approach": {"home": "warning", "work": "warning"}, "modes": {
        "home": [("body", 0.5, "danger"), ("hand", 1.0, "warning")],
        "work": [("body", 2.0, "warning")]}},
    "restricted": {"label": "Restricted area (other)", "approach": {"home": "warning", "work": "warning"}, "modes": {
        "home": [("body", 1.0, "warning")],
        "work": [("body", 1.0, "warning")]}},
    "bathroom": {"label": "Bathroom / off-limits room", "approach": {"home": "warning"}, "modes": {
        "home": [("body", 1.0, "warning")]}},
    "cabinet": {"label": "Medicine / cleaning cabinet", "approach": {}, "modes": {
        "home": [("hand", 0.5, "danger")]}},
    "window": {"label": "Window / balcony", "approach": {"home": "warning"}, "modes": {
        "home": [("feet_on", 0.3, "danger"), ("body", 1.5, "warning")]}},
    "stairs": {"label": "Stairs", "approach": {"home": "warning"}, "modes": {
        "home": [("body", 0.5, "warning")]}},
    "crib": {"label": "Crib / bed with rails", "approach": {}, "modes": {
        "home": [("above_top", 0.5, "danger")]}},
    "furniture": {"label": "Dresser / shelf / TV stand (tip-over)", "approach": {}, "modes": {
        "home": [("feet_on", 0.3, "danger")]}},
    "machine": {"label": "Machinery / moving parts", "approach": {"work": "warning"}, "modes": {
        "work": [("hand", 0.2, "danger"), ("body", 1.0, "warning")]}},
    "edge": {"label": "Unprotected edge / floor opening", "approach": {"work": "warning"}, "modes": {
        "work": [("body", 0.5, "danger")]}},
    "drop_zone": {"label": "Drop zone (under a load / scaffold)", "approach": {"work": "warning"}, "modes": {
        "work": [("body", 1.0, "danger")]}},
    "vehicle": {"label": "Vehicle / forklift lane", "approach": {"work": "warning"}, "modes": {
        "work": [("body", 1.5, "warning")]}},
}

RULE_TEXT = {"hand": "Hand in", "body": "Entered", "above_top": "Climbing over", "feet_on": "Climbing on"}

HANDS = (15, 16, 19, 20)
WRISTS = (15, 16)
SHOULDERS = (11, 12)
HIPS = (23, 24)
FEET = (27, 28, 31, 32)
TRACK_JOINTS = (0, 11, 12, 15, 16, 23, 24, 27, 28)
MIN_VISIBILITY = 0.5
COOLDOWN_S = 10.0
REACH = 0.6
APPROACH_S = 2.0
MIN_SPEED = 0.25
REACH_S = 0.8
MIN_HAND_SPEED = 0.8
MOTIONLESS_S = {"home": 10.0, "work": 15.0}
WATER_STILL_S = 8.0
STILL_RANGE = 0.15


def _points(lm, idxs):
    return [(float(lm[i][0]), float(lm[i][1])) for i in idxs if lm[i][3] >= MIN_VISIBILITY]


def _mean(pts):
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _inside(p, box, pad=0.0):
    return box[0] - pad <= p[0] <= box[2] + pad and box[1] - pad <= p[1] <= box[3] + pad


class Body:
    """Measurements of one skeleton: torso size, floor position, lying / upside-down."""
    def __init__(self, lm, aspect=1.0, view="angled"):
        self.lm = lm
        sh, hip = _mean(_points(lm, SHOULDERS)), _mean(_points(lm, HIPS))
        self.ok = sh is not None and hip is not None
        if not self.ok:
            return
        self.hip = hip
        self.torso = max(math.hypot(sh[0] - hip[0], sh[1] - hip[1]), 0.02)
        if view == "top":
            self.ground = hip
        else:
            feet = _points(lm, FEET)
            self.ground = _mean(feet) if feet else (hip[0], hip[1] + 1.6 * self.torso)
        dx, dy = (sh[0] - hip[0]) * aspect, sh[1] - hip[1]
        self.down = math.degrees(math.atan2(abs(dx), abs(dy))) >= 60 or dy > 0.3 * math.hypot(dx, dy)


def at_zone_depth(body, box, reach=REACH):
    """True when the person stands at the zone (judged from their floor position)."""
    r = reach * body.torso
    gx, gy = body.ground
    return box[0] - r <= gx <= box[2] + r and box[1] - r * 0.5 <= gy <= box[3] + r


def check_rule(kind, body, box):
    lm = body.lm
    if kind == "hand":
        return at_zone_depth(body, box) and any(_inside(p, box) for p in _points(lm, HANDS))
    if kind == "body":
        gx, gy = body.ground
        return box[0] <= gx <= box[2] and box[1] <= gy <= box[3] + 0.3 * body.torso
    if kind == "above_top":
        return at_zone_depth(body, box) and box[0] <= body.hip[0] <= box[2] and body.hip[1] < box[1]
    if kind == "feet_on":
        lower_limit = box[3] - 0.25 * (box[3] - box[1])
        return any(box[0] <= p[0] <= box[2] and box[1] <= p[1] <= lower_limit for p in _points(lm, FEET))
    return False


def _time_to_enter(p, v, box):
    """Seconds until point p moving at velocity v enters box (None if it never does)."""
    tmin, tmax = 0.0, float("inf")
    for i, (lo, hi) in enumerate(((box[0], box[2]), (box[1], box[3]))):
        if abs(v[i]) < 1e-9:
            if not lo <= p[i] <= hi:
                return None
        else:
            a, b = (lo - p[i]) / v[i], (hi - p[i]) / v[i]
            a, b = min(a, b), max(a, b)
            tmin, tmax = max(tmin, a), min(tmax, b)
            if tmin > tmax:
                return None
    return tmin


class ZoneMonitor:
    """Checks each skeleton against the zones of the current mode; returns events."""
    def __init__(self, zones=None, mode="home", view="angled"):
        self.history = deque()
        self.set_zones(zones or [], mode, view)

    def set_zones(self, zones, mode, view="angled"):
        self.zones, self.mode, self.view = list(zones), mode, view
        self.since, self.last_fired = {}, {}

    def _active(self, z):
        return self.mode in ZONE_TYPES.get(z["type"], {}).get("modes", {})

    def _has_hand_rule(self, z):
        return any(k == "hand" for k, _, _ in ZONE_TYPES.get(z["type"], {}).get("modes", {}).get(self.mode, []))

    def _fire(self, key, t, cooldown=COOLDOWN_S):
        if t - self.last_fired.get(key, -1e9) >= cooldown:
            self.last_fired[key] = t
            return True
        return False

    @staticmethod
    def _event(kind, severity, z, detail):
        return {"kind": kind, "severity": severity, "zone": z["name"] if z else None,
                "zone_type": z["type"] if z else None, "detail": detail}

    def update(self, t, pose, aspect=1.0):
        events, active = [], set()
        body = None
        if pose is not None:
            b = Body(pose.reshape(33, 4), aspect, self.view)
            if b.ok:
                body = b
        if body is not None:
            joints = {i: (float(body.lm[i][0]), float(body.lm[i][1]))
                      for i in TRACK_JOINTS if body.lm[i][3] >= MIN_VISIBILITY}
            self.history.append((t, body.ground, joints, body.torso))
        while self.history and t - self.history[0][0] > max(MOTIONLESS_S.values()) + 1.0:
            self.history.popleft()

        if body is not None:
            for z in self.zones:
                for kind, dwell_s, severity in ZONE_TYPES.get(z["type"], {}).get("modes", {}).get(self.mode, []):
                    key = (z["id"], kind)
                    if not check_rule(kind, body, z["box"]):
                        continue
                    active.add(key)
                    self.since.setdefault(key, t)
                    if t - self.since[key] >= dwell_s and self._fire(key, t):
                        events.append(self._event("zone", severity, z, f"{RULE_TEXT[kind]}: {z['name']}"))
            events += self._approach(t, body)
            events += self._reach(t, body)
            events += self._motionless(t, body)

        for key in list(self.since):
            if key not in active:
                del self.since[key]
        return events

    def _approach(self, t, body):
        out = []
        past = [h for h in self.history if 0.4 <= t - h[0] <= 0.8]
        if not past:
            return out
        t0, g0 = past[0][0], past[0][1]
        gx, gy = body.ground
        v = ((gx - g0[0]) / (t - t0), (gy - g0[1]) / (t - t0))
        if math.hypot(*v) / body.torso < MIN_SPEED:
            return out
        for z in self.zones:
            severity = ZONE_TYPES.get(z["type"], {}).get("approach", {}).get(self.mode)
            if not severity:
                continue
            x1, y1, x2, y2 = z["box"]
            r = REACH * body.torso
            floor_area = (x1 - r, y1 - r * 0.5, x2 + r, y2 + r)
            if _inside((gx, gy), floor_area):
                continue
            eta = _time_to_enter((gx, gy), v, floor_area)
            if eta is not None and eta <= APPROACH_S and self._fire((z["id"], "approach"), t):
                out.append(self._event("zone_approach", severity, z,
                                       f"Heading toward {z['name']} - about {eta:.1f} s away"))
        return out

    def _reach(self, t, body):
        out = []
        past = [h for h in self.history if 0.15 <= t - h[0] <= 0.4]
        if not past:
            return out
        t0, joints0 = past[0][0], past[0][2]
        dt = t - t0
        for z in self.zones:
            if not self._has_hand_rule(z) or not at_zone_depth(body, z["box"], reach=1.2):
                continue
            best, already_in = None, False
            for j in WRISTS:
                if j not in joints0 or body.lm[j][3] < MIN_VISIBILITY:
                    continue
                p = (float(body.lm[j][0]), float(body.lm[j][1]))
                if _inside(p, z["box"]):
                    already_in = True
                    break
                v = ((p[0] - joints0[j][0]) / dt, (p[1] - joints0[j][1]) / dt)
                if math.hypot(*v) / body.torso < MIN_HAND_SPEED:
                    continue
                eta = _time_to_enter(p, v, z["box"])
                if eta is not None and eta <= REACH_S and (best is None or eta < best):
                    best = eta
            if not already_in and best is not None and self._fire((z["id"], "reach"), t, cooldown=5.0):
                out.append(self._event("zone_reach", "warning", z,
                                       f"Reaching toward {z['name']} - contact in about {best:.1f} s"))
        return out

    def _motionless(self, t, body):
        in_water = any(z["type"] == "water" and self._active(z) and check_rule("body", body, z["box"])
                       for z in self.zones)
        need = WATER_STILL_S if in_water else (MOTIONLESS_S.get(self.mode, 10.0) if body.down else None)
        if need is None:
            return []
        window = [h for h in self.history if t - h[0] <= need]
        if not window or t - window[0][0] < need * 0.9:
            return []
        ranges = []
        for j in TRACK_JOINTS:
            pts = [h[2][j] for h in window if j in h[2]]
            if len(pts) >= 3:
                xs, ys = zip(*pts)
                ranges.append(max(max(xs) - min(xs), max(ys) - min(ys)))
        if len(ranges) < 3 or sorted(ranges)[len(ranges) // 2] / body.torso >= STILL_RANGE:
            return []
        if not self._fire(("motionless", in_water), t, cooldown=30.0):
            return []
        where = "in water" if in_water else "lying down"
        return [self._event("motionless", "danger", None,
                            f"Person {where} and not moving for {need:.0f} s - check immediately")]

    def zones_near(self, pose, aspect=1.0):
        """Names of active-mode zones the person is at (used to rate fall severity)."""
        if pose is None:
            return []
        b = Body(pose.reshape(33, 4), aspect, self.view)
        if not b.ok:
            return []
        return [z["name"] for z in self.zones if self._active(z) and at_zone_depth(b, z["box"], reach=1.2)]
