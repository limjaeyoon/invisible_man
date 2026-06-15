"""Pinch detection — ported from the AIRMPC press algorithm.

Robustness comes from three ideas (not a fixed distance threshold):
  * normalize thumb-index gap by hand size  -> distance/zoom invariant
  * track an adaptive OPEN-hand baseline (EMA) -> adapts to your hand
  * hysteresis: press below baseline*(1-PRESS_FRAC), release above
    baseline*(1-RELEASE_FRAC), with a fast-close shortcut.

Uses MediaPipe Tasks HandLandmarker (works on builds without mp.solutions).
"""
import math
import os
import threading
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")   # quiet MediaPipe INFO/WARNING spam

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "assets" / "models" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

# --- AIRMPC tuning ---
PINCH_SMOOTH_ALPHA = 0.78
BASELINE_ALPHA = 0.08
BASELINE_BAND_FRAC = 0.10
PRESS_FRAC = 0.80          # press when gap < 20% of open baseline (firm pinch)
RELEASE_FRAC = 0.55
PINCH_CLOSING_DELTA_TH = 0.03
PINCH_STABLE_FRAMES = 1
FAST_PRESS_STABLE_FRAMES = 2
COOLDOWN_SEC = 0.5

# Hand detection is run on a downscaled frame: the pinch metric is a *ratio*
# (thumb-index gap / hand size), so it's resolution-invariant — shrinking the
# input only makes MediaPipe faster, it doesn't change the measurement.
DETECT_W = 480


def ema(prev, new, alpha):
    return float(new) if prev is None else float(alpha * new + (1.0 - alpha) * prev)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PinchPress:
    """Single-target pinch-as-press state machine (OPEN <-> PINCHED)."""
    def __init__(self):
        self.baseline = None
        self.smooth = None
        self.prev_raw = None
        self.state = "OPEN"
        self._press = 0
        self._last_trigger = -1e9

    def _update_baseline(self, ratio):
        if self.baseline is None:
            self.baseline = ratio
            return
        floor = self.baseline * (1.0 - BASELINE_BAND_FRAC)
        if ratio >= floor:                      # don't let a pinch drag it down
            self.baseline = ema(self.baseline, ratio, BASELINE_ALPHA)

    def update(self, ratio_raw, now):
        """Feed the current (raw) pinch ratio. Returns True on a press event."""
        if ratio_raw is None:
            self.state = "OPEN"
            self._press = 0
            return False

        delta = 0.0
        if self.prev_raw is not None:
            delta = float(np.clip(self.prev_raw - ratio_raw, -0.25, 0.25))
        self.prev_raw = ratio_raw

        self.smooth = ema(self.smooth, ratio_raw, PINCH_SMOOTH_ALPHA)
        self._update_baseline(self.smooth)
        if self.baseline is None:
            return False

        press_th = self.baseline * (1.0 - PRESS_FRAC)
        release_th = self.baseline * (1.0 - RELEASE_FRAC)
        closing_fast = delta >= PINCH_CLOSING_DELTA_TH
        required = FAST_PRESS_STABLE_FRAMES if closing_fast else PINCH_STABLE_FRAMES

        trigger = False
        if self.state == "OPEN":
            press_like = (self.smooth < press_th) or (closing_fast and self.smooth < release_th)
            self._press = self._press + 1 if press_like else 0
            if self._press >= required:
                if now - self._last_trigger >= COOLDOWN_SEC:
                    trigger = True
                    self._last_trigger = now
                self.state = "PINCHED"
                self._press = 0
        else:  # PINCHED
            if self.smooth > release_th:
                self.state = "OPEN"
        return trigger


def _finger_up(lm, mcp, pip, tip):
    """A finger is extended-and-pointing-up when each joint sits above the last
    (image y grows downward)."""
    return lm[tip].y < lm[pip].y < lm[mcp].y


def hand_presented(lm):
    """True only for a deliberately presented hand: open and raised. We require
    at least two of middle/ring/pinky extended upward — a pinch gesture keeps
    those fingers up while thumb+index close, whereas a hand resting in view has
    them curled or pointing down. This gates out accidental pinches.
    """
    fingers = ((9, 10, 12), (13, 14, 16), (17, 18, 20))   # middle, ring, pinky
    return sum(_finger_up(lm, *f) for f in fingers) >= 2


class PinchDetector:
    def __init__(self):
        if not MODEL.exists():
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading hand landmarker model ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL)
            print("  saved ->", MODEL)
        # VIDEO mode tracks landmarks between frames instead of re-running full
        # palm detection every call -> much lower latency and smoother tracking.
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hl = vision.HandLandmarker.create_from_options(opts)
        self.press = PinchPress()
        self.hands = []                       # latest landmarks (for drawing)
        self._ts = 0                          # monotonic ms timestamp for VIDEO mode

    def update(self, frame_bgr, now):
        """Detect hands; return True on a confirmed pinch-press this frame.

        Only a *presented* hand (open & raised, see hand_presented) is eligible
        to pinch, so a relaxed hand resting in view can't trigger by accident.
        """
        h, w = frame_bgr.shape[:2]
        if w > DETECT_W:                              # cheaper detection, same ratio
            s = DETECT_W / float(w)
            frame_bgr = cv2.resize(frame_bgr, (DETECT_W, max(1, int(round(h * s)))))
            h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = int(now * 1000)
        if ts <= self._ts:                           # must strictly increase
            ts = self._ts + 1
        self._ts = ts
        res = self.hl.detect_for_video(img, ts)

        self.hands = res.hand_landmarks or []
        best_ratio = None
        for lm in self.hands:
            if not hand_presented(lm):               # ignore down/relaxed hands
                continue
            wrist = (lm[0].x * w, lm[0].y * h)
            thumb = (lm[4].x * w, lm[4].y * h)
            imcp = (lm[5].x * w, lm[5].y * h)
            itip = (lm[8].x * w, lm[8].y * h)
            hand_scale = dist(wrist, imcp)
            if hand_scale <= 1.0:
                continue
            ratio = dist(thumb, itip) / hand_scale   # the most-pinched hand wins
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio

        return self.press.update(best_ratio, now)

    def close(self):
        try:
            self.hl.close()
        except Exception:
            pass


class ThreadedPinch:
    """Run pinch detection on a worker thread so MediaPipe never stalls the
    render loop.

    The main loop `submit()`s the latest frame and `poll()`s for a press event;
    detection happens in the background (MediaPipe releases the GIL during
    inference, so it truly overlaps the matte/render work). Press events are
    latched, so a pinch is never missed even if the main loop polls late.
    """
    def __init__(self, detector=None):
        self.det = detector or PinchDetector()
        self._lock = threading.Lock()
        self._frame = None
        self._now = 0.0
        self._fired = False
        self._hands = []
        self.count = 0              # frames processed (debug)
        self.ms = 0.0              # EMA detection time, ms (debug)
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def submit(self, frame_bgr, now):
        with self._lock:
            self._frame = frame_bgr
            self._now = now

    def poll(self):
        """Return True once per detected pinch-press, then clear the latch."""
        with self._lock:
            fired, self._fired = self._fired, False
        return fired

    def get_hands(self):
        """Latest hand landmarks (list per hand of 21 normalized points)."""
        with self._lock:
            return self._hands

    def _loop(self):
        while self._run:
            with self._lock:
                f, now = self._frame, self._now
                self._frame = None
            if f is None:
                time.sleep(0.003)
                continue
            try:
                t0 = time.time()
                fired = self.det.update(f, now)
                dt = (time.time() - t0) * 1000.0
                with self._lock:
                    if fired:
                        self._fired = True
                    self._hands = self.det.hands
                    self.count += 1
                    self.ms = dt if self.count == 1 else 0.9 * self.ms + 0.1 * dt
            except Exception:
                time.sleep(0.02)

    def close(self):
        self._run = False
        self.det.close()


# MediaPipe hand skeleton (21 landmarks): bones to draw between points.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
)


def render_hands(h, w, hands):
    """Render a minimal monochrome hand skeleton on a transparent layer.

    Returns (overlay_bgr, alpha) so the caller can composite it BELOW the chrome
    (alpha gated by chrome coverage). Style: thin white bones over a faint dark
    underlay for legibility, small white node dots ringed in black. Landmarks are
    normalized, so they map onto any display size.
    """
    overlay = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    color = (0, 0, 0)                            # single pure color (black)
    for lm in hands:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:            # one thin, sharp line
            cv2.line(overlay, pts[a], pts[b], color, 1, cv2.LINE_AA)
            cv2.line(alpha, pts[a], pts[b], 255, 1, cv2.LINE_AA)
        for x, y in pts:                         # small dot
            cv2.circle(overlay, (x, y), 2, color, -1, cv2.LINE_AA)
            cv2.circle(alpha, (x, y), 2, 255, -1, cv2.LINE_AA)
    return overlay, alpha
