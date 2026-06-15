# INVISIBLE MAN

> They never see you go. One moment you're standing in the room; the next, a
> sheet of living mercury pours down over your skin, swallows your outline, and
> when it slides away there's nothing left but the empty room behind you. No
> trapdoors, no green screen — just light bending around a person who decided to
> stop being seen. **silverchrome** turns your webcam into that trick: pinch your
> fingers and liquid chrome floods over you, then the world quietly closes over
> the space where you used to be. You are the invisible man, and the only tell is
> a faint cold constellation tracing your hand.

Real-time on a plain webcam. No green screen, no special hardware — just a moment
where you step out of frame once so it can learn the empty room.

---

## What it does

Pinch (or press space) and a flowing liquid-chrome layer grows over your body.
Once it has you fully covered, the **entire background dissolves from your live
camera into a still capture of the empty room** — so you settle as a chrome
figure standing in a room that no longer contains you. Pinch again and it
reverses: the room fades back to live, the chrome lifts, and you reappear.

The clever part: because the layer underneath the chrome becomes the *empty-room
capture*, the body-tracking mask never has to be pixel-perfect. If it lags while
you move, the gap shows empty room instead of your real body — so you genuinely
vanish.

---

## Requirements

- **Python 3.9–3.12** (avoid 3.13 — MediaPipe wheels lag behind)
- A **webcam** and a **desktop OpenGL** context (a normal macOS/Windows/Linux
  session; pure headless SSH won't work — the chrome shader needs real OpenGL)
- macOS, Windows, or Linux

Python packages (`requirements.txt`): `opencv-python`, `mediapipe`, `moderngl`,
`numpy`, `pillow`, `onnxruntime`.

ML models (hand landmarker + selfie segmenter, ~a few MB total) **download
automatically on first run** into `assets/models/`.

---

## Setup

```bash
git clone <your-repo-url> silverchrome
cd silverchrome

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick start

```bash
python main.py
```

1. **Step out of frame** and press **`c`** — captures the empty room (the
   background you'll dissolve into). The terminal prints
   `background plate captured (averaged).`
2. **Step back in.** You look normal.
3. **Pinch** thumb + index on either hand — *with your hand open and raised* —
   or press **`SPACE`**. Liquid chrome sweeps over you and the room closes in;
   you vanish.
4. **Pinch / `SPACE` again** to come back.
5. **`q`** or **`ESC`** to quit.

> **The pinch only registers when your hand is presented** — open and raised,
> with at least two of middle/ring/pinky extended upward. A relaxed hand resting
> in view (fingers naturally together) won't trigger it by accident.

> **A plate is required.** Without pressing `c` first there's no empty room to
> dissolve into, so a pinch just prints a reminder.

---

## Controls

| Key | Action |
|----|--------|
| `c` | Capture background plate (do once, out of frame; averaged) |
| `SPACE` / pinch | Toggle invisibility |
| `q` / `ESC` | Quit |
| `r` | Start / stop recording to `output/` |
| `m` | Mirror the camera |
| `j` | Toggle the hand-skeleton overlay |
| `p` | Toggle the debug perf log (render / matte / hand fps + state) |
| `v` | Matte debug view (see exactly what's tracked) |
| **Tracking** | |
| `t` / `g` | Matte edge tighter / fuller |
| `a` / `s` | Matte temporal smoothing (steadier / more responsive) |
| `o` / `i` | Chrome peak cover width during the morph (wider / narrower) |
| **Chrome look** | |
| `9` / `0` | Chrome amount (clear glass ↔ mirror) |
| `,` / `.` | Refraction strength |
| `-` / `=` | Ripple depth |
| `;` / `'` | Ripple size |
| `[` / `]` | Flow speed |
| `k` / `l` | Chromatic aberration |
| `f` / `d` | Edge rim brightness |

---

## How it works

### The two-layer illusion

```
   live you            chrome curtain              empty-room capture
  (webcam)      +     (liquid metal, GPU)    over     (still plate)
```

A transition runs in two phases (`tp` 0 → 1):

1. **Cover** — chrome grows over the *live* you.
2. **Dissolve** — once covered, the whole-frame base fades from live camera to
   the captured empty room (`u_base_plate`). The chrome stays as the figure.

Reverse runs it backwards: the base returns to live (you reappear *under* the
chrome), then the chrome lifts. The cover width even **swells across the morph
and thins once settled**, so the chrome fully hides your real edge during the
hand-off but hugs your form at rest.

### The pipeline (one file per stage)

```
  Webcam            Body matte           Chrome shader          Composite
 (threaded)   ──▶  (human-only)    ──▶   (moderngl/GLSL)  ──▶   + hands + show/record
  capture.py        matte.py             chrome.py              main.py
                    gesture.py (pinch + hand skeleton) ─────────┘
```

| File | Role |
|------|------|
| `capture.py` | Threaded webcam grabber — always serves the freshest frame (MJPG, 1-frame buffer) to cut latency. |
| `matte.py` | Body segmentation. **`SelfieMatte`** (MediaPipe Selfie Segmentation, human-only) is the default; `RVMatte` / `BGMatte` remain available. `ThreadedMatte` runs it off the render thread; `keep_significant` keeps the person + disconnected limbs while dropping small non-human blobs. |
| `gesture.py` | `PinchDetector` (MediaPipe HandLandmarker, VIDEO tracking mode) with the presented-hand gate; `ThreadedPinch` runs it on a worker thread; `render_hands` draws the minimal skeleton overlay. |
| `chrome.py` | `ChromeRenderer` — offscreen moderngl context; uploads textures, runs the shader, reads the result back for OpenCV. Continuously exposure/white-balance matches the plate to the live feed so the dissolve is seamless. |
| `shaders/chrome.frag` | The GPU look: animated liquid surface → normals → refraction of the room + chrome matcap reflection + fresnel rim, composited over the live↔capture base. |
| `main.py` | Orchestrator — capture loop, transition state machine, matte/pinch threads, hand overlay, controls, recording. |
| `tools/make_matcap.py` | Generates the chrome matcap PNG. |

### Performance notes

- **Everything heavy is threaded** — capture, matte, and hand tracking each have
  their own worker, so none stalls rendering.
- **Hand tracking** uses MediaPipe VIDEO (tracking) mode on a downscaled frame,
  detected every frame, for low-latency, smooth landmarks.
- **Idle skip** — matte inference only runs while a transition is in flight or
  you're invisible.
- Press **`p`** for a live readout of render / matte / hand fps and timings.

---

## The material

The chrome look lives entirely in `assets/matcaps/chrome.png` — a "matcap"
sphere the shader samples by surface normal. **Swap that PNG** and the metal
becomes gold, copper, or oil-slick with no code change (`tools/make_matcap.py`
can generate variants).

---

## Troubleshooting

- **Black window / no camera (macOS):** grant camera permission to your terminal
  (System Settings → Privacy & Security → Camera), then rerun.
- **Wrong camera:** edit `CAM_INDEX` near the top of `main.py` (try `1`, `2`).
- **`moderngl` context error:** you need a real desktop OpenGL session; this
  won't run over plain headless SSH.
- **MediaPipe install fails:** confirm Python is 3.9–3.12.
- **Harmless log spam:** `clearcut … FAILED_PRECONDITION` lines are MediaPipe's
  offline analytics failing — ignore them.
- **The dissolve shows a lighting jump:** re-capture the plate with `c`; the
  plate is exposure-matched to live continuously, but a fresh capture helps.
- **Chrome doesn't fully cover your edge during the morph:** nudge `o` (wider).
  Too puffy when settled? Nudge `i`.

---

## Status

Working real-time invisibility effect with the two-layer dissolve, human-only
matting, gesture control, and the hand-skeleton overlay.
