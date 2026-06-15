"""Accurate body matting via RobustVideoMatting (RVM) ONNX.

RVM segments the person by learned structure, not color, so a dark shirt on a
dark background is still covered, and fine finger/hair edges are preserved.
It's recurrent: the r1..r4 states carry temporal memory between frames.
"""
import threading
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "assets" / "models" / "rvm_mobilenetv3_fp32.onnx"
MODEL_URL = ("https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
             "v1.0.0/rvm_mobilenetv3_fp32.onnx")

BGM_MODEL = ROOT / "assets" / "models" / "bgmv2_mobilenetv2_hd.onnx"
BGM_URL = ("https://downloads.sourceforge.net/project/backgroundmattingv2.mirror/"
           "v1.0.0/onnx_mobilenetv2_hd.onnx")


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


class RVMatte:
    def __init__(self, scale=0.5, thr=0.55):
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit(
                "RobustVideoMatting needs onnxruntime.\n"
                "    pip install onnxruntime\n"
                "then run again.")
        if not MODEL.exists():
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading RobustVideoMatting model (~15 MB) ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL)
            print("  saved ->", MODEL)

        avail = ort.get_available_providers()
        use = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
               if p in avail] or ["CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(str(MODEL), providers=use)
        except Exception as e:                       # CoreML EP can be flaky
            print("provider", use, "failed (", e, "); using CPU.")
            self.sess = ort.InferenceSession(str(MODEL),
                                             providers=["CPUExecutionProvider"])
        print("RVM running on:", self.sess.get_providers()[0])

        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        # recurrent state inputs (r1i..r4i) start as zeros
        self.rec = {n: np.zeros((1, 1, 1, 1), np.float32)
                    for n in self.in_names if n.startswith("r") and n.endswith("i")}
        self.scale = float(scale)
        self.thr = float(thr)          # alpha threshold (edge tightness)

    def alpha(self, frame_bgr):
        """Return float32 alpha matte [0,1] at the frame's full resolution."""
        H, W = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (0, 0), fx=self.scale, fy=self.scale)
        h, w = small.shape[:2]
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = np.transpose(rgb, (2, 0, 1))[None]                 # 1,3,h,w
        # RVM needs its semantic branch downsampled (~400px on the long side),
        # else it loses global context and fires on bright background objects.
        ratio = float(np.clip(400.0 / max(h, w), 0.2, 1.0))

        feeds = {"src": src,
                 "downsample_ratio": np.array([ratio], np.float32)}
        feeds.update(self.rec)
        outs = self.sess.run(self.out_names, feeds)
        od = dict(zip(self.out_names, outs))
        for k in self.rec:                                       # r1i <- r1o
            self.rec[k] = od[k[:-1] + "o"]

        pha = od["pha"][0, 0]                                    # h,w in [0,1]
        if self.thr > 0:                                        # tighten edges
            pha = np.clip((pha - self.thr) / (1.0 - self.thr), 0, 1)
        return cv2.resize(pha, (W, H))

    def reset(self):
        for k in self.rec:
            self.rec[k] = np.zeros((1, 1, 1, 1), np.float32)

    def set_background(self, plate_bgr):
        """No-op: RVM segments the person directly and needs no plate. The
        captured plate is still used elsewhere (refraction background), so the
        app calls this uniformly across matte backends."""
        return None


SELFIE_MODEL = ROOT / "assets" / "models" / "selfie_segmenter.tflite"
SELFIE_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
              "selfie_segmenter/float16/latest/selfie_segmenter.tflite")


class SelfieMatte:
    """Human-only foreground segmentation via MediaPipe Selfie Segmentation.

    Purpose-built to answer "is this pixel part of the person?", so furniture and
    other objects are never selected — fixing matte spill onto non-human areas.
    Fast (256px internally) and low-latency; soft edges are fine because the
    chrome cover + captured-room base hide them.
    """
    def __init__(self, thr=0.5):
        import mediapipe as mp
        self.mp = mp
        self._handle = None
        self._run = self._build()
        self.thr = float(thr)

    def _build(self):
        mp = self.mp
        # Tasks API first (known to behave on a worker thread here), then legacy.
        try:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            if not SELFIE_MODEL.exists():
                SELFIE_MODEL.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading selfie segmenter model (~250 KB) ...")
                urllib.request.urlretrieve(SELFIE_URL, SELFIE_MODEL)
                print("  saved ->", SELFIE_MODEL)
            opts = vision.ImageSegmenterOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(SELFIE_MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True)
            seg = vision.ImageSegmenter.create_from_options(opts)
            self._handle = seg
            print("SelfieMatte: mediapipe Tasks ImageSegmenter")

            def run(frame_bgr):
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
                return seg.segment(img).confidence_masks[0].numpy_view()
            return run
        except Exception as e:
            print("Tasks selfie segmenter unavailable (", e, "); using solutions.")

        ss = mp.solutions.selfie_segmentation
        seg = ss.SelfieSegmentation(model_selection=1)
        self._handle = seg
        print("SelfieMatte: mediapipe.solutions SelfieSegmentation")

        def run(frame_bgr):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return seg.process(rgb).segmentation_mask
        return run

    def alpha(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        m = self._run(frame_bgr)
        if m is None:
            return np.zeros((H, W), np.float32)
        m = np.asarray(m, np.float32)
        if m.ndim > 2:                                  # (H,W,1) -> (H,W)
            m = m[..., 0]
        m = np.clip(m, 0.0, 1.0)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H))
        if self.thr > 0:                                # tighten the edge
            m = np.clip((m - self.thr) / (1.0 - self.thr), 0.0, 1.0)
        return m

    def set_background(self, plate_bgr):
        return None

    def reset(self):
        return None


def keep_significant(pha, min_frac=0.0006):
    """Keep the person *and* any sizeable disconnected parts, dropping only
    small speckle noise.

    The old version kept just the single largest blob, so a hand raised into a
    head-and-shoulders shot — which reads as a separate island from the torso —
    vanished. Here we keep every component whose area is at least `min_frac` of
    the frame (a hand easily clears that), which preserves limbs that aren't
    connected in-frame while still discarding tiny matte flicker.
    """
    h, w = pha.shape[:2]
    b = (pha > 0.4).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
    if n <= 1:
        return pha
    areas = stats[1:, cv2.CC_STAT_AREA]           # skip label 0 (background)
    floor = max(64.0, min_frac * h * w)
    keep_ids = np.nonzero(areas >= floor)[0] + 1
    if keep_ids.size == 0:                         # everything is tiny: keep top
        keep_ids = np.array([1 + int(np.argmax(areas))])
    keep = np.isin(lab, keep_ids).astype(np.float32)   # no dilation: keep edges tight
    return pha * keep


class BGMatte:
    """BackgroundMattingV2: matte = f(current frame, captured empty-room plate).

    Because the model is TOLD the background (our plate), static furniture is
    definitively background even when your hand is on it, and a dark shirt over
    a dark wall is still segmented correctly. Needs set_background() first.
    """
    def __init__(self, scale=0.6):
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit("BackgroundMattingV2 needs onnxruntime.\n"
                             "    pip install onnxruntime")
        if not BGM_MODEL.exists():
            BGM_MODEL.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading BackgroundMattingV2 model (~20 MB) ...")
            urllib.request.urlretrieve(BGM_URL, BGM_MODEL)
            print("  saved ->", BGM_MODEL)

        # CoreML can't build an execution plan for this model on Apple Silicon
        # (fails every frame), so run on CPU. Threading keeps the video smooth.
        self.sess = ort.InferenceSession(str(BGM_MODEL),
                                         providers=["CPUExecutionProvider"])
        print("BGMv2 running on:", self.sess.get_providers()[0])
        print("  inputs:", [(i.name, i.shape) for i in self.sess.get_inputs()])
        print("  outputs:", [o.name for o in self.sess.get_outputs()])

        # model input size: honor static dims, else use a /4-aligned scaled size
        shp = self.sess.get_inputs()[0].shape           # [1,3,H,W]
        self._fixed = (isinstance(shp[2], int) and isinstance(shp[3], int))
        self._fixed_hw = (int(shp[2]), int(shp[3])) if self._fixed else None
        self.scale = float(scale)
        self.thr = 0.0
        self._bgr = None                                # bgr tensor (1,3,h,w)
        self._hw = None                                 # (h,w) used for inference

    def _target_hw(self, H, W):
        if self._fixed:
            return self._fixed_hw
        h = (int(H * self.scale) // 4) * 4
        w = (int(W * self.scale) // 4) * 4
        return max(4, h), max(4, w)

    def _to_tensor(self, bgr_img, hw):
        img = cv2.resize(bgr_img, (hw[1], hw[0]))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])

    def set_background(self, plate_bgr):
        H, W = plate_bgr.shape[:2]
        self._hw = self._target_hw(H, W)
        self._bgr = self._to_tensor(plate_bgr, self._hw)

    def alpha(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        if self._bgr is None:                           # need a plate first
            return np.zeros((H, W), np.float32)
        src = self._to_tensor(frame_bgr, self._hw)
        pha, = self.sess.run(["pha"], {"src": src, "bgr": self._bgr})
        pha = pha[0, 0]
        if self.thr > 0:
            pha = np.clip((pha - self.thr) / (1.0 - self.thr), 0, 1)
        return cv2.resize(pha, (W, H))


class ThreadedMatte:
    """Run RVM on a worker thread so the video keeps rendering at full fps.

    The render loop submits the latest frame and reads the most recent alpha
    (slightly stale, but the displayed video stays smooth and low-latency).
    """
    def __init__(self, matter):
        self.matter = matter
        self.lock = threading.Lock()
        self._frame = None
        self._alpha = None
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def set_background(self, plate_bgr):
        self.matter.set_background(plate_bgr)

    def submit(self, frame_bgr):
        with self.lock:
            self._frame = frame_bgr

    def get(self):
        with self.lock:
            return self._alpha

    def _loop(self):
        printed = False
        while self._run:
            with self.lock:
                f = self._frame
                self._frame = None
            if f is None:
                time.sleep(0.003)
                continue
            try:
                a = self.matter.alpha(f)
                if not printed:
                    print("matte OK: shape", a.shape, "max", round(float(a.max()), 3))
                    printed = True
                with self.lock:
                    self._alpha = a
            except Exception as e:
                if not printed:
                    import traceback
                    traceback.print_exc()
                    printed = True
                time.sleep(0.05)

    @property
    def thr(self):
        return self.matter.thr

    @thr.setter
    def thr(self, v):
        self.matter.thr = v

    def close(self):
        self._run = False
