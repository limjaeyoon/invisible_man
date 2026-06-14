"""Person segmentation -> a soft mask of where you are in the frame.

Works across mediapipe versions: tries the legacy `solutions` API first, then
falls back to the current `tasks` ImageSegmenter (auto-downloads the model).
"""
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "assets" / "models"
MODEL_PATH = MODEL_DIR / "selfie_segmenter.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter/float16/latest/selfie_segmenter.tflite")


def _legacy():
    """Return a callable(frame_bgr)->float mask using mp.solutions, or None."""
    try:
        ss = mp.solutions.selfie_segmentation
    except AttributeError:
        try:
            from mediapipe.python.solutions import selfie_segmentation as ss
        except Exception:
            return None, None
    seg = ss.SelfieSegmentation(model_selection=1)

    def run(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        m = seg.process(rgb).segmentation_mask
        return m

    return run, seg


def _tasks():
    """Return a callable using the modern Tasks ImageSegmenter."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MODEL_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading selfie segmenter model (~250 KB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  saved ->", MODEL_PATH)

    opts = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
    )
    segmenter = vision.ImageSegmenter.create_from_options(opts)

    def run(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(rgb))
        res = segmenter.segment(mp_img)
        return res.confidence_masks[0].numpy_view()

    return run, segmenter


class Segmenter:
    def __init__(self, model_selection=1):
        run, handle = _legacy()
        if run is None:
            print("mediapipe.solutions unavailable; using Tasks API.")
            run, handle = _tasks()
        self._run = run
        self._handle = handle
        # matte: pixels closer to the plate than this are always background
        # (this is what keeps finger gaps out, at any coverage level).
        self.matte_floor = 16.0
        self.matte_band = 20.0

    def _seg(self, frame_bgr):
        m = self._run(frame_bgr)
        if m is None:
            h, w = frame_bgr.shape[:2]
            return np.zeros((h, w), np.float32)
        m = np.clip(np.asarray(m, np.float32), 0.0, 1.0)
        if m.shape[:2] != frame_bgr.shape[:2]:
            m = cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return m

    def mask(self, frame_bgr, plate_bgr=None, coverage=1.0):
        """Return float32 alpha [0,1].

        `coverage` (0..1) is how much of your body is currently selected: it
        keeps the top `coverage`-fraction of pixels by contrast-vs-plate, so the
        invisible region spreads evenly across you as it rises. Finger gaps
        (pixels matching the plate) are excluded at any coverage via the floor.
        """
        seg = self._seg(frame_bgr)
        if plate_bgr is None:
            seg = cv2.dilate(seg, np.ones((5, 5), np.uint8), 1)
            return cv2.GaussianBlur(seg, (0, 0), 1.5)
        if coverage <= 0.001:
            return np.zeros(frame_bgr.shape[:2], np.float32)

        # per-pixel difference between live frame and the empty room
        a = frame_bgr.astype(np.float32)
        b = plate_bgr.astype(np.float32)
        d = np.sqrt(np.sum((a - b) ** 2, axis=2))

        # region gate around the person blob (ignore background clutter/shadows)
        gate = (seg > 0.25).astype(np.uint8)
        gate = cv2.dilate(gate, np.ones((7, 7), np.uint8), 2)
        gate_f = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), 3.0)

        # threshold = the contrast value above which the top `coverage`-fraction
        # of body pixels sits. Kept >= floor so background gaps never pass.
        vals = d[gate > 0]
        if vals.size == 0:
            return np.zeros_like(d)
        thr = float(np.quantile(vals, 1.0 - float(np.clip(coverage, 0, 1))))
        lo = max(thr, self.matte_floor)
        hi = lo + self.matte_band

        diff = np.clip((d - lo) / (hi - lo), 0, 1)
        alpha = diff * gate_f
        au = (alpha * 255).astype(np.uint8)
        au = cv2.morphologyEx(au, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        au = cv2.medianBlur(au, 3)
        return cv2.GaussianBlur(au.astype(np.float32) / 255.0, (0, 0), 1.0)

    @staticmethod
    def height_from_mask(mask, blur=21):
        """Blur the silhouette into a rounded height field (edge=low, core=high)."""
        binm = (mask > 0.5).astype(np.uint8)
        if binm.sum() == 0:
            return np.zeros_like(binm, np.uint8)
        dist = cv2.distanceTransform(binm, cv2.DIST_L2, 5)
        if dist.max() > 0:
            dist = dist / dist.max()
        dist = cv2.GaussianBlur(dist, (0, 0), blur / 3.0)
        return (np.clip(dist, 0, 1) * 255).astype(np.uint8)

    def close(self):
        try:
            self._handle.close()
        except Exception:
            pass
