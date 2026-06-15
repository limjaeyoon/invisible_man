"""Threaded webcam capture — always hand back the freshest frame.

Decoding frames inside the render loop adds latency, and most webcams keep an
internal buffer that piles up several frames behind real time. A background
grabber fixes both: it continuously drains the camera and keeps only the most
recent frame, so main.py never blocks on I/O and never shows a stale image.

Two extra wins applied here:
  * MJPG pixel format — most USB webcams only reach 30 fps at HD in MJPG; the
    default (YUYV) often caps at 5-10 fps for the same resolution.
  * BUFFERSIZE = 1 — ask the driver not to queue frames behind us.
"""
import threading
import cv2


class Camera:
    def __init__(self, index=0, width=1280, height=720, fps=30, mjpg=True):
        self.cap = cv2.VideoCapture(index)
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._lock = threading.Lock()
        self._frame = None
        self._ok = self.cap.isOpened()
        if self._ok:                       # prime with the first frame
            ok, f = self.cap.read()
            if ok:
                self._frame = f
            else:
                self._ok = False

        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def isOpened(self):
        return self._ok

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                self._ok = False
                break
            with self._lock:
                self._frame = f

    def read(self):
        """Return (ok, latest_frame). The frame may repeat if the render loop
        outruns the camera; callers copy before mutating, so that's harmless."""
        if not self._ok:                   # camera failed/closed -> signal stop
            return False, None
        with self._lock:
            f = self._frame
        return True, f

    def release(self):
        self._run = False
        try:
            self.t.join(timeout=0.5)
        except Exception:
            pass
        self.cap.release()
