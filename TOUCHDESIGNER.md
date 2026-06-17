# Liquid Metal in TouchDesigner + POPX — Build Guide

Goal: a **T‑1000 liquid-metal morph** driven by your webcam — pinch and real,
reflective liquid metal erupts from your hand, flows over and conforms to your
form, then morphs back. The leap over the Python version is **real reflections**
(POPX path tracer) and a **real deforming surface** (POPX soft-body), instead of
a 2D shader painted on the video.

> Honesty note: this guide is an architecture/plan. TouchDesigner is a node app,
> so it gets built on your machine, not from a code repo. The TouchDesigner
> operator design below is solid; the **POPX-specific node names/params are from
> research** of Mini UV's docs/tutorials and may differ slightly — map each step
> to the actual POPX nodes using his Soft Body tutorial. Where I'm inferring, it
> says so.

---

## 0. What to install / get (shopping list)

| Thing | What / where | Cost |
|---|---|---|
| **TouchDesigner** | [derivative.ca](https://derivative.ca/) — Non-Commercial license | Free (≤1280×1280 output, fine here) |
| **MediaPipe plugin** | [github.com/torinmb/mediapipe-touchdesigner](https://github.com/torinmb/mediapipe-touchdesigner) — download `release.zip`, no install; hands/pose/segmentation, GPU, Mac | Free |
| **POPX** | Mini UV [Patreon](https://www.patreon.com/cw/_mini_uv) — get the tier that includes the POPX package + **Soft Body** tutorial + **Path Tracer** | Paid (Patreon) |
| HDRI (optional) | A chrome-friendly studio/room HDRI (Poly Haven) for reflections | Free |

**First, watch:** Mini UV's [POPX Soft Body – Part 1](https://www.youtube.com/watch?v=U31CnnmhC_M) and his Path Tracer posts. Build *his* cloth example once before adapting it — you'll learn the POPX constraint/solver/material/path-tracer nodes you'll reuse here.

---

## The pipeline at a glance

```
Webcam ─┬─> MediaPipe: hands (CHOP) ─────────> pinch trigger + seed point
        ├─> MediaPipe: segmentation (TOP) ───> body mask
        │
        └─> body mask ─> SDF/height (GLSL TOP) ─> displaced relief mesh (POP/SOP)
                                                      │
                                       POPX Soft Body solver (conform + flow)
                                                      │
                              POPX chrome material + Path Tracer (real reflections)
                                                      │  (room/HDRI as environment)
                                          render (RGBA) ─> Over the live webcam
```

The pinch drives a **coverage field** that grows the metal from your hand across
the soft-body — that's the morph.

---

## Milestone 0 — Setup & signals

1. New project. **Video Device In TOP** → pick your webcam.
2. Drop in the **MediaPipe** component (from the plugin `.toe`/`.tox`). Point it at the same camera. Enable **Hands** and **Image Segmentation**.
3. Confirm you get:
   - Hand landmarks as a **CHOP** (per-landmark x/y/z, normalized).
   - A **segmentation/foreground TOP** (you = white).
4. Sanity-check both update live.

---

## Milestone 1 — Build a body-shaped surface (no depth model needed yet)

We give the path tracer real geometry shaped like you, from the silhouette.

1. **Mask → height/SDF.** Take the segmentation TOP → **GLSL TOP** that computes a smooth interior field (a cheap jump-flood/at-distance approximation, or just a heavy **Blur TOP** of the mask as a v1). Result: bright in your core, soft at the edges = a relief height map.
2. **Displace a grid.** A high-res **Grid SOP** (or POPX grid) → displace its points along Z by the height map (GLSL/POP displace). You now have a 2.5D "relief of you" facing camera. Keep it dense enough to deform smoothly (e.g., 256×256).
3. Position/scale it to overlay your camera silhouette (orthographic-ish camera so it lines up with the video).

> Upgrade later (Milestone 6): replace the silhouette height with a real **depth
> map** for fuller 3D. See depth note at the end.

---

## Milestone 2 — POPX Soft Body: make it liquid

Convert the relief into a flowing soft-body skin (this is exactly the cloth
tutorial, repurposed):

1. Feed the relief mesh into the **POPX Soft Body** setup.
2. **Constraints (POPX Constraint stack):**
   - **Pin-to-target** → target = the live relief mesh, so the skin *conforms to
     your current shape every frame* (this is what makes it "you").
   - **Distance-along-edge** → resist stretching (keeps it cohesive like metal).
   - **Bend-across-triangles** → folds/wobble (the liquid jiggle).
3. **Forces:** add gentle **turbulence/noise force** + a small **gravity** so the
   surface flows and sags like viscous metal rather than sitting rigid.
4. Tune stiffness/damping so it reads as **heavy liquid metal** (higher damping,
   moderate stiffness) — not bouncy cloth.

Result: a reflective-ready surface that hugs your form and undulates.

---

## Milestone 3 — Chrome material + Path Tracer (the realism)

This is the payoff POPX buys us.

1. Apply a **POPX chrome/metal material**: metallic = 1, **roughness very low**
   (mirror), base color near-white/neutral steel.
2. Set up the **POPX Path Tracer** with lights:
   - An **Area light** or two for the bright moving streaks.
   - **Environment / HDRI** for the reflected world — this is what sells mercury.
3. **Reflect your real room:** feed your **captured room** (a clean webcam frame,
   or an HDRI of your space) as the **environment** so the chrome mirrors *your*
   surroundings. (Equivalent of the plate reflection in the Python version, but
   now real.) Start with an HDRI to get it looking right, then swap to the
   room feed.
4. Raise path-tracer samples until reflections are clean (balance vs. FPS).

You should now have a **mirror-metal version of your body** that reflects the
environment and deforms — already far past the screen-space look.

---

## Milestone 4 — The morph (pinch → liquid grows from your hand)

Drive coverage from the gesture.

1. **Pinch detection (CHOPs):** from the hands CHOP, take **thumb tip (4)** and
   **index tip (8)**; **distance** between them, normalized by hand size
   (wrist→index-mcp) so it's zoom-invariant. **Logic/Trigger CHOP** fires when the
   ratio drops below a threshold; debounce with a **Filter/Lag** + cooldown.
2. **Coverage value:** the trigger toggles a target 0↔1; smooth it with a
   **Filter/Speed CHOP** over ~1.5 s → `coverage`.
3. **Seed point:** the pinch hand's screen position (thumb–index midpoint) →
   a point in the mesh's UV/space.
4. **Growth field:** compute per-point **distance from the seed** (POPX falloff /
   a GLSL field over the mesh), normalized 0→1; the metal exists where
   `distance < coverage`. Wiggle the threshold with noise for tendrils.
5. Hook `coverage`/growth to: the **soft-body activation/displacement amount**
   and/or the **material's metal mask** — so the liquid literally spreads from
   your hand across the body. Reverse on the next pinch (coverage → 0).
6. **Rolling lip:** boost displacement + a brighter spec where the growth field is
   mid-transition (the advancing front) → the chrome wave that leads the morph.

---

## Milestone 5 — Composite & polish

1. Render the path-traced metal with **alpha** → **Over TOP** onto the live
   **Video Device In** (so the metal you sits in your real room).
2. Align the render camera to the webcam framing (orthographic, matched scale).
3. Polish: turbulence amount, droplet/bead noise on the surface, edge tendrils,
   light streak speed, reflection HDRI choice.

---

## Milestone 6 (optional, big 3D upgrade) — real depth

The silhouette relief is 2.5D. For true depth-correct form:

- Get a **depth map** into TD. The MediaPipe plugin has no depth, so options are:
  - run a depth model (Depth Anything V2) in a **separate process** and pipe the
    depth image into TD via **Spout/Syphon** (Syphon on Mac) or **NDI**;
  - or a **Script TOP** running ONNX (slower).
- Displace the grid by depth (instead of the silhouette height) → fuller 3D body
  the soft-body wraps. Everything downstream (soft-body, material, path tracer)
  stays the same.

(We already have `depth.py` here producing a normalized depth map — it can be the
external process that Syphons depth into TD.)

---

## Build order (do these in sequence)

1. M0 setup → see hands + segmentation live.
2. Mini UV's **cloth tutorial** end-to-end (learn POPX).
3. M1 relief mesh → M2 soft-body on it.
4. M3 chrome + path tracer + HDRI (get the metal looking real *static* first).
5. M4 pinch-driven growth (the morph).
6. M5 composite over webcam.
7. M6 depth upgrade if you want fuller 3D.

## Where I can help from here

- Design/refine the **CHOP network** for pinch + seed + coverage (I can write the
  exact operator chain and any **Python/GLSL TOP** snippets).
- Write **GLSL TOP** code for the mask→height/SDF, the growth field, and the
  advancing-lip displacement.
- Stand up the **depth → Syphon** bridge process (reusing `depth.py`).
- Translate any of our Python logic (pinch algorithm, seed-front, exposure match)
  into TD equivalents.

Tell me which milestone you're on and what POPX nodes you actually see, and I'll
get specific.

## Sources
- [TouchDesigner (Derivative)](https://derivative.ca/) · [MediaPipe TD plugin](https://github.com/torinmb/mediapipe-touchdesigner) · [plugin tutorial](https://derivative.ca/community-post/tutorial/face-hand-pose-tracking-more-touchdesigner-mediapipe-gpu-plugin/68278)
- [Mini UV (POPX) Patreon](https://www.patreon.com/cw/_mini_uv) · [POPX about](https://popx.tools/about) · [Soft Body tutorial](https://www.patreon.com/posts/popx-soft-body-1-144985141) · [video](https://www.youtube.com/watch?v=U31CnnmhC_M) · [POPX 1.2 (path tracer lights)](https://www.patreon.com/posts/popx-version-1-2-151783781)
