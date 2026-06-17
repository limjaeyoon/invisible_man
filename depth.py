"""Monocular depth via Depth Anything V2 (small) ONNX -> a real 3D surface.

Gives a true per-pixel depth map of you, so the liquid metal can use real
surface normals, thickness, and reflections instead of a faked silhouette dome.

Notes for Apple Silicon: the stock `onnxruntime` wheel is usually CPU-only, so
this may run on CPU. Lower DEPTH_SIZE for more speed (coarser depth). If your
onnxruntime has CoreML, it'll be used automatically.
"""
import threading
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
# fp32 model.onnx is the most reliable; for more speed on CPU you can switch
# MODEL_FILE to "model_quantized.onnx" or "model_fp16.onnx" (same repo).
MODEL_FILE = "model.onnx"
MODEL = ROOT / "assets" / "models" / "depth_anything_v2_small.onnx"
MODEL_URL = ("https://huggingface.co/onnx-community/depth-anything-v2-small/"
             "resolve/main/onnx/" + MODEL_FILE)

DEPTH_SIZE = 392                      # input side (multiple of 14); lower = faster
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthEstimator:
    def __init__(self, size=DEPTH_SIZE):
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit("Depth needs onnxruntime.\n    pip install onnxruntime")
        if not MODEL.exists():
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading Depth Anything V2 small (~100 MB) ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL)
            print("  saved ->", MODEL)

        avail = ort.get_available_providers()
        use = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
               if p in avail] or ["CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(str(MODEL), providers=use)
        except Exception as e:
            print("depth provider", use, "failed (", e, "); using CPU.")
            self.sess = ort.InferenceSession(str(MODEL),
                                             providers=["CPUExecutionProvider"])
        print("Depth running on:", self.sess.get_providers()[0])

        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.out_name = self.sess.get_outputs()[0].name
        print("  depth in:", inp.name, inp.shape, " out:",
              self.sess.get_outputs()[0].name, self.sess.get_outputs()[0].shape)

        # honor a static input size if the model has one, else use our size
        h, w = inp.shape[2], inp.shape[3]
        if isinstance(h, int) and isinstance(w, int):
            self.size = (h, w)
        else:
            s = (int(size) // 14) * 14
            self.size = (s, s)
        self._printed = False

    def depth(self, frame_bgr):
        """Return depth in [0,1] at the frame's resolution (1 = nearest)."""
        H, W = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.size[1], self.size[0]))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], np.float32)
        d = self.sess.run([self.out_name], {self.in_name: x})[0]
        d = np.squeeze(d).astype(np.float32)         # Depth Anything: inverse depth
        lo, hi = float(d.min()), float(d.max())
        if not self._printed:
            print("depth OK: shape", d.shape, "range [%.2f, %.2f]" % (lo, hi))
            self._printed = True
        d = (d - lo) / (hi - lo + 1e-6)              # near = bright
        return cv2.resize(d, (W, H))


class ThreadedDepth:
    """Run depth on a worker thread; the render loop reads the latest map."""
    def __init__(self, estimator):
        self.est = estimator
        self.lock = threading.Lock()
        self._frame = None
        self._depth = None
        self.count = 0
        self.ms = 0.0
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def submit(self, frame_bgr):
        with self.lock:
            self._frame = frame_bgr

    def get(self):
        with self.lock:
            return self._depth

    def _loop(self):
        while self._run:
            with self.lock:
                f = self._frame
                self._frame = None
            if f is None:
                time.sleep(0.003)
                continue
            try:
                t0 = time.time()
                d = self.est.depth(f)
                dt = (time.time() - t0) * 1000.0
                with self.lock:
                    self._depth = d
                    self.count += 1
                    self.ms = dt if self.count == 1 else 0.9 * self.ms + 0.1 * dt
            except Exception:
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def close(self):
        self._run = False
