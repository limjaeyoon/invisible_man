# TouchDesigner build — two routes

You can't hand-author a `.toe` (binary, undocumented). Both routes below build a
**real** network with the supported Python API. Use them together: generate a
starting network with the script, then switch to the live-MCP route on your
laptop to refine it interactively and add POPX.

---

## Route A — Generator script (works from anywhere, incl. this cloud session)

`td_build.py` constructs the network when run inside TouchDesigner.

1. Open TouchDesigner. Drop in the **MediaPipe** plugin component
   ([torinmb/mediapipe-touchdesigner](https://github.com/torinmb/mediapipe-touchdesigner)),
   rename it `mediapipe`, point it at your webcam, enable **Hands** + **Image Segmentation**.
2. Open the Textport (`Alt+T`) and run:
   ```python
   exec(open('/full/path/to/touchdesigner/td_build.py').read())
   ```
   (or paste the file into a Text DAT and right-click → *Run Script*).
3. It builds everything under `/liquid_metal`. Then **File → Save As** → `liquid_metal.toe`.

**Things to verify after it runs** (flagged in the script comments):
- `webcam` device.
- The MediaPipe **paths/channel names** in the CONFIG block — segmentation TOP
  path and hand-landmark channel names differ by plugin version. Open the
  plugin's CHOP/TOP, read the real names, update CONFIG, re-run.
- A couple of operator **type tokens** vary by TD build (e.g. `geoCOMP`); the
  script names the exact node if one fails so you fix just that one.

What you get: webcam → silhouette → blurred mask → height (GLSL) → displaced
256² relief mesh → chrome-ish GLSL material → render → composited over the live
feed, plus the full **pinch→coverage** CHOP brain (`/liquid_metal/coverage`,
0..1). This is the scaffold; POPX replaces the render/material stage.

---

## Route B — Live MCP build (best; needs Claude Code on your laptop)

This lets Claude build/wire/tune nodes **directly in your running TD**.

1. Install the plugin: [satoruhiga/claude-touchdesigner](https://github.com/satoruhiga/claude-touchdesigner).
   Load `TouchDesignerAPI.tox` in your project — its MCP server auto-starts
   (default port **44444**, override with `TDAPI_PORT`).
2. Run **Claude Code locally on your laptop** (not the cloud session — the cloud
   container can't reach `localhost:44444`). Add the MCP server per that repo's
   README so tools `td_execute` / `td_pane` / `td_operators` / `td_selection`
   appear.
3. Ask me (in the local session) to build it. I'll create operators, set params,
   wire them, and read back the network state to self-correct — including the
   POPX soft-body + path-tracer stages I can't verify blind from a script.

---

## POPX upgrade (applies to both routes)

The script's **Render TOP + GLSL chrome material** is the placeholder. Replace with:
- **POPX Soft Body** solver on the relief mesh (pin-to-target = the relief, +
  distance/bend constraints + turbulence) for the flowing liquid surface.
- **POPX chrome material** (metallic 1, roughness ~0) + **POPX Path Tracer** with
  an HDRI / your room as environment → real reflections.
- Drive the morph: feed `/liquid_metal/coverage` into a POPX growth/falloff field
  seeded at the pinch hand position. See `../TOUCHDESIGNER.md` milestones 2–4.
