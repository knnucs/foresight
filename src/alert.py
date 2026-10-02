"""
Turns the model's risk score (and the skeleton) into device behavior.
Shared by live.py and calibrate.py, so calibration measures exactly what the device does.

States:
  NORMAL  -> WARNING : smoothed risk stays above the threshold for hold_s     -> soft chime
  WARNING -> NORMAL  : risk stays clearly below for clear_s                    -> logged as "prevented"
  any     -> ALARM   : person lies down (torso near horizontal) for lying_hold_s,
                       right after a warning OR after a fast drop from upright -> loud alarm
  ALARM stays until reset() (a caregiver acknowledges it).
"""
import math
from collections import deque

import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24


class RiskSmoother:
    """Exponential moving average: removes frame-to-frame flicker."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class WarningTrigger:
    """Starts a warning when risk stays >= threshold for hold_s seconds.
    Ends it when risk stays below (threshold - hysteresis) for clear_s seconds."""
    def __init__(self, threshold, hold_s=0.5, clear_s=1.5, hysteresis=0.1):
        self.threshold = threshold
        self.hold_s, self.clear_s, self.hysteresis = hold_s, clear_s, hysteresis
        self.reset()

    def reset(self):
        self.active = False
        self.above_since = None
        self.below_since = None

    def update(self, risk, t):
        if not self.active:
            if risk >= self.threshold:
                if self.above_since is None:
                    self.above_since = t
                if t - self.above_since >= self.hold_s:
                    self.active, self.below_since = True, None
                    return "warn"
            else:
                self.above_since = None
        else:
            if risk < self.threshold - self.hysteresis:
                if self.below_since is None:
                    self.below_since = t
                if t - self.below_since >= self.clear_s:
                    self.active, self.above_since = False, None
                    return "clear"
            else:
                self.below_since = None
        return None


class FallConfirmer:
    """Confirms a fall: torso near horizontal for lying_hold_s, reached quickly from
    upright with the hips dropping, or at any speed right after a warning."""
    def __init__(self, lying_angle=60.0, upright_angle=45.0, lying_hold_s=2.0,
                 max_transition_s=1.2, drop_thresh=0.4, history_s=8.0):
        self.lying_angle, self.upright_angle = lying_angle, upright_angle
        self.lying_hold_s, self.max_transition_s = lying_hold_s, max_transition_s
        self.drop_thresh, self.history_s = drop_thresh, history_s
        self.hist = deque()
        self.lying_since = None
        self.fired = False
        self.last_posture = None

    def reset(self):
        self.hist.clear()
        self.lying_since = None
        self.fired = False
        self.last_posture = None

    def update(self, t, pose, aspect=1.0, warned_recently=False):
        while self.hist and t - self.hist[0][0] > self.history_s:
            self.hist.popleft()
        if pose is None:
            self.last_posture = None
            return False
        lm = np.asarray(pose, dtype=np.float32).reshape(33, 4)
        sh = (lm[L_SH, :2] + lm[R_SH, :2]) / 2
        hip = (lm[L_HIP, :2] + lm[R_HIP, :2]) / 2
        dx, dy = (sh[0] - hip[0]) * aspect, sh[1] - hip[1]
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        torso = math.hypot(dx, dy)
        inverted = dy > 0.3 * torso
        upright = angle < self.upright_angle and not inverted
        down = angle >= self.lying_angle or inverted
        self.last_posture = "down" if down else ("upright" if upright else "between")
        self.hist.append((t, float(hip[1]), torso, angle, upright))

        if not down:
            self.lying_since = None
            if upright:
                self.fired = False
            return False
        if self.lying_since is None:
            self.lying_since = t
        if self.fired or t - self.lying_since < self.lying_hold_s:
            return False

        if warned_recently:
            self.fired = True
            return True
        upright = [h for h in self.hist if h[0] < self.lying_since and h[4]]
        if not upright:
            return False
        last_upright_t = upright[-1][0]
        if self.lying_since - last_upright_t > self.max_transition_s:
            return False
        ref = [h for h in upright if h[0] >= last_upright_t - 1.5]
        ref_hip = float(np.median([h[1] for h in ref]))
        ref_torso = max(float(np.median([h[2] for h in ref])), 1e-3)
        lying_hip = float(np.median([h[1] for h in self.hist if h[0] >= self.lying_since]))
        if (lying_hip - ref_hip) / ref_torso >= self.drop_thresh:
            self.fired = True
            return True
        return False


class AlertSystem:
    def __init__(self, threshold, warned_window_s=6.0):
        self.smoother = RiskSmoother()
        self.trigger = WarningTrigger(threshold)
        self.confirmer = FallConfirmer()
        self.warned_window_s = warned_window_s
        self.state = "NORMAL"
        self.last_warning_t = -1e9
        self.recover_s = 10.0
        self.upright_since = None

    @property
    def threshold(self):
        return self.trigger.threshold

    def set_threshold(self, thr):
        self.trigger.threshold = float(min(max(thr, 0.05), 0.99))

    def update(self, t, risk_raw, pose, aspect=1.0):
        """Returns (smoothed risk, list of events: 'warning', 'prevented', 'alarm')."""
        events = []
        risk = self.smoother.update(risk_raw)
        if self.state == "ALARM":
            self.confirmer.update(t, pose, aspect)
            if self.confirmer.last_posture == "upright":
                if self.upright_since is None:
                    self.upright_since = t
                if t - self.upright_since >= self.recover_s:
                    self.reset()
                    events.append("recovered")
            else:
                self.upright_since = None
            return risk, events
        ev = self.trigger.update(risk, t)
        if ev == "warn":
            self.state, self.last_warning_t = "WARNING", t
            events.append("warning")
        elif ev == "clear" and self.state == "WARNING":
            posture = self.confirmer.last_posture
            if posture == "down":
                self.state = "ALARM"
                events.append("alarm")
                return risk, events
            self.state = "NORMAL"
            events.append("prevented" if posture == "upright" else "cleared_unsure")
        warned = self.state == "WARNING" or t - self.last_warning_t <= self.warned_window_s
        if self.confirmer.update(t, pose, aspect, warned_recently=warned):
            self.state = "ALARM"
            events.append("alarm")
        return risk, events

    def reset(self):
        self.smoother.reset()
        self.trigger.reset()
        self.confirmer.reset()
        self.upright_since = None
        self.state = "NORMAL"
        self.last_warning_t = -1e9
