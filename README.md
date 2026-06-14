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

## Knobs (planned hotkeys)
- flow speed = how liquidy
- displacement strength = how much it ripples
- material = cycle matcap images
- background = real / black / custom

## Status
Scaffolding. Build in progress.
