# silverchrome

Real-time liquid-chrome (Silver Surfer) effect. Webcam in → you become flowing reflective metal → composited over your real room.

## Architecture (simple version)

Four stages, each its own file. Frame flows left to right:

```
  Webcam            Segmentation         Chrome shader          Composite
 (OpenCV)   ──▶   (MediaPipe mask)  ──▶   (moderngl/GLSL)  ──▶   + show/record
  capture.py        segment.py            chrome.py             main.py
```

1. **capture** — grab webcam frames.
2. **segment** — MediaPipe finds the person, outputs a black/white mask (where you are).
3. **chrome** — the GPU magic. Blur the mask into a rounded height field → fake surface normals → animate noise to make it flow → look up a chrome "matcap" image by normal direction → that's the liquid metal. Add a bright fresnel rim + bloom.
4. **main** — glues it together: chrome person pasted over the real background, live preview, keyboard controls, optional recording.

**The material** lives entirely in `assets/matcaps/` — a chrome sphere PNG. Swap the file → chrome becomes gold or oil-slick, no code change.

## Performance & tracking

Built for low latency and accurate rotoscoping:
- **Threaded capture** (`capture.py`) — a background grabber always serves the
  freshest frame; MJPG + a 1-frame driver buffer avoid stale, queued images.
- **Threaded matte & hand detection** — segmentation and pinch detection each
  run on their own worker thread, so neither stalls rendering. Hand detection
  runs on a downscaled frame (the pinch metric is resolution-invariant).
- **Idle skip** — matte inference is skipped while the effect is fully off.
- **Disconnected parts kept** — the matte keeps every sizeable region, not just
  the largest blob, so a hand raised into a head-and-shoulders shot (a separate
  island from the torso) still gets the effect.
- **Averaged background plate** — pressing `c` denoises by averaging a short
  burst, sharpening the live-vs-plate contrast the matte depends on.
- **Temporal smoothing** (`a`/`s`) — optional EMA steadies edges and stops
  near-threshold blobs from flickering; set to 1.0 for zero added lag.

## Status
Working. Real-time webcam effect with the pipeline above.
