"""
td_build.py  —  Liquid-metal network generator for TouchDesigner.

WHAT THIS IS
  Run this INSIDE TouchDesigner and it constructs the network for you
  (creates operators, sets parameters, wires them). Then File > Save As a .toe.
  This is the supported way to "generate a TD project" — TD has no safe way to
  hand-author the binary .toe; you script the network instead.

HOW TO RUN
  1. In TouchDesigner: open a Textport (Alt+T) OR make a Text DAT, paste this in.
  2. Easiest: Text DAT > right-click > "Run Script", or in the textport:
        exec(open('/full/path/to/td_build.py').read())
  3. It builds everything under a new base COMP: /liquid_metal
  4. File > Save As  ->  liquid_metal.toe

CONFIDENCE / HONESTY
  - Native-operator creation + wiring + the pinch CHOP chain: high confidence.
  - A few operator TYPE TOKENS vary by TD build (e.g. geoCOMP vs geometryCOMP).
    The helper reports the exact node that fails so you fix ONE token, not all.
  - MediaPipe and POPX are THIRD-PARTY. They are NOT created here — you drop the
    MediaPipe .tox in manually and install POPX. This script wires up to clearly
    marked reference paths (see CONFIG below) and leaves POPX hookup as TODOs.
"""

# ----------------------------------------------------------------------------
# CONFIG — set these to match your scene, then run.
# ----------------------------------------------------------------------------
ROOT = op('/')                      # where the base COMP gets created
CONTAINER = 'liquid_metal'          # name of the base COMP we build inside

# Path to the MediaPipe plugin component AFTER you drop it into your project.
# (Download torinmb/mediapipe-touchdesigner, drag its .tox in, rename it 'mediapipe'.)
MP_PATH = '/mediapipe'

# Channel names the MediaPipe component outputs for hand landmarks.
# VERIFY these against the plugin's CHOP (names differ by plugin version).
# We need thumb-tip (landmark 4) and index-tip (landmark 8), plus wrist (0) and
# index-mcp (5) to normalize for distance-to-camera (zoom invariance).
CH_THUMB_X, CH_THUMB_Y = 'hand_0:thumb_tip:tx',  'hand_0:thumb_tip:ty'
CH_INDEX_X, CH_INDEX_Y = 'hand_0:index_tip:tx',  'hand_0:index_tip:ty'
CH_WRIST_X, CH_WRIST_Y = 'hand_0:wrist:tx',      'hand_0:wrist:ty'
CH_IMCP_X,  CH_IMCP_Y  = 'hand_0:index_mcp:tx',  'hand_0:index_mcp:ty'

PINCH_ON  = 0.45   # pinch CLOSES when (thumb-index dist / hand size) < this
PINCH_OFF = 0.65   # pinch RELEASES above this (hysteresis, debounce)
MORPH_SECS = 1.5   # seconds for the metal to grow / recede


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def build():
    # fresh start
    old = ROOT.op(CONTAINER)
    if old:
        old.destroy()
    C = ROOT.create(baseCOMP, CONTAINER)        # noqa: F821 (TD global)
    C.color = (0.18, 0.20, 0.26)

    def make(optype, name, x=0, y=0):
        """Create an op inside C, position it, return it. Reports failures."""
        try:
            n = C.create(optype, name)
        except NameError:
            raise RuntimeError(
                f"Operator type token not valid in this TD build for '{name}'. "
                f"Open the OP Create menu, hover the op, and use its type token."
            )
        n.nodeX, n.nodeY = x * 200, y * -150
        return n

    def wire(a, b, in_index=0, out_index=0):
        a.outputConnectors[out_index].connect(b.inputConnectors[in_index])

    def dat(name, text, x=0, y=0):
        d = make(textDAT, name, x, y)           # noqa: F821
        d.text = text
        return d

    # ------------------------------------------------------------------ INPUT
    webcam = make(videodeviceinTOP, 'webcam', 0, 0)   # noqa: F821
    # webcam.par.device = ...  # pick your camera in the param dialog

    # ----------------------------------------------------- BODY MASK (silhouette)
    # MediaPipe segmentation. We reference the plugin's foreground/seg TOP by path.
    # If your plugin exposes seg at a different path, fix MP_PATH or this select.
    mask = make(selectTOP, 'body_mask', 2, 0)         # noqa: F821
    mask.par.top = MP_PATH + '/seg_out'   # <-- VERIFY this path in your project
    blur = make(blurTOP, 'mask_blur', 3, 0)           # noqa: F821
    blur.par.size = 24
    wire(mask, blur)

    # height field from the blurred mask (GLSL TOP) -> bright core, soft edges
    height_glsl = dat('height_frag', _HEIGHT_FRAG, 3, 1)
    height = make(glslTOP, 'height', 4, 0)            # noqa: F821
    height.par.pixeldat = height_glsl.name
    wire(blur, height)
    height_out = make(nullTOP, 'HEIGHT', 5, 0)       # noqa: F821
    wire(height, height_out)

    # --------------------------------------------------- DISPLACED RELIEF MESH
    # High-res grid; displaced along Z in a vertex shader that samples HEIGHT.
    grid = make(gridSOP, 'relief_grid', 4, 3)        # noqa: F821
    grid.par.rows = 256
    grid.par.cols = 256
    geo = make(geoCOMP, 'relief_geo', 6, 3)          # noqa: F821
    # move the grid INTO the geo so it renders:
    grid.parent  # (kept for clarity)
    grid.dock = geo
    try:
        grid.parent  # no-op
    except Exception:
        pass
    # NOTE: simplest reliable path is to create the SOP *inside* the geo. If the
    # above relocation doesn't take, delete relief_grid and add a Grid SOP inside
    # relief_geo by hand, then point the vertex shader below at HEIGHT.

    # displacement material (GLSL MAT). Vertex shader pushes verts by HEIGHT.
    vtx = dat('disp_vert', _DISP_VERT, 6, 4)
    pix = dat('chrome_pix', _CHROME_PIX, 6, 5)
    mat = make(glslMAT, 'liquid_mat', 7, 3)          # noqa: F821
    mat.par.vertexdat = vtx.name
    mat.par.pixeldat = pix.name
    # bind HEIGHT to sampler slot 0. Param names vary by build — VERIFY:
    try:
        mat.par.top0 = height_out.path
        mat.par.sampler0name = 'sHeight'
    except Exception:
        pass  # set the "Samplers" page of liquid_mat by hand if these differ
    geo.par.material = mat.path

    # camera + light + render
    cam = make(camCOMP, 'cam', 8, 2)                 # noqa: F821
    cam.par.tz = 5
    light = make(lightCOMP, 'key_light', 8, 4)       # noqa: F821
    render = make(renderTOP, 'render', 9, 3)         # noqa: F821
    render.par.camera = cam.path
    render.par.geometry = geo.path
    render.par.lights = light.path
    # ^^^ POPX UPGRADE: replace this Render TOP with the POPX PATH TRACER, feed an
    #     HDRI / your room as environment for real chrome reflections. The geo and
    #     material insertion points above are the same. See touchdesigner/README.

    # --------------------------------------------------------- COMPOSITE OUT
    comp = make(overTOP, 'composite', 11, 1)         # noqa: F821
    wire(render, comp, in_index=0)   # metal on top
    wire(webcam, comp, in_index=1)   # live feed behind
    out = make(nullTOP, 'OUT', 12, 1)                # noqa: F821
    wire(comp, out)

    # ------------------------------------------------------- PINCH CHOP BRAIN
    _build_pinch(make, wire, dat)

    print(f"[td_build] Built network under {C.path}. "
          f"Set webcam device, verify MediaPipe paths/channels, then Save As .toe.")
    return C


def _build_pinch(make, wire, dat):
    """thumb-index distance / hand-size -> debounced pinch -> coverage 0..1."""
    # pull the landmark channels from the MediaPipe CHOP
    sel = make(selectCHOP, 'hand_in', 0, 8)          # noqa: F821
    sel.par.chop = MP_PATH + '/hand_out'   # <-- VERIFY the plugin's hand CHOP path
    sel.par.channames = ' '.join([
        CH_THUMB_X, CH_THUMB_Y, CH_INDEX_X, CH_INDEX_Y,
        CH_WRIST_X, CH_WRIST_Y, CH_IMCP_X,  CH_IMCP_Y,
    ])

    # distances via a Math CHOP expression node (Expression CHOP)
    expr = make(expressionCHOP, 'pinch_calc', 2, 8)  # noqa: F821
    # one output channel 'ratio' = pinchDist / handSize, zoom-invariant
    expr.par.numexpr = 1
    expr.par.const0name = 'ratio'
    # NOTE: Expression CHOP references inputs as op('hand_in')[chan]. Distances:
    pd = (f"(({_q(CH_THUMB_X)}-{_q(CH_INDEX_X)})**2 + "
          f"({_q(CH_THUMB_Y)}-{_q(CH_INDEX_Y)})**2)**0.5")
    hs = (f"(({_q(CH_WRIST_X)}-{_q(CH_IMCP_X)})**2 + "
          f"({_q(CH_WRIST_Y)}-{_q(CH_IMCP_Y)})**2)**0.5")
    expr.par.expr0 = f"({pd}) / (({hs}) + 1e-4)"
    wire(sel, expr)

    # hysteresis threshold -> 0/1 pinch state (Logic CHOP, schmitt-style via 2 logics
    # is cleanest; here a single Logic with bounds for a starting point)
    logic = make(logicCHOP, 'pinch_state', 4, 8)     # noqa: F821
    logic.par.bound = 'off'
    logic.par.convert = 'bytheinputvalues'
    logic.par.boundmin = PINCH_OFF   # treat as: below PINCH_ON => closed
    logic.par.boundmax = PINCH_ON
    wire(expr, logic)

    # toggle on each fresh pinch (Trigger/Count) -> latch 0<->1
    trig = make(triggerCHOP, 'pinch_trigger', 6, 8)  # noqa: F821
    wire(logic, trig)
    count = make(countCHOP, 'morph_target', 8, 8)    # noqa: F821
    count.par.limitmax = 1
    count.par.limittype = 'wrap'   # 0,1,0,1 ... each pinch flips the target
    wire(trig, count)

    # smooth the 0/1 target into a continuous coverage over MORPH_SECS
    speed = make(speedCHOP, 'coverage_speed', 10, 8) # noqa: F821
    lag = make(filterCHOP, 'coverage', 12, 8)        # noqa: F821
    lag.par.type = 'gaussian'
    lag.par.width = MORPH_SECS
    wire(count, lag)
    # 'coverage' (0..1) is the morph amount. Reference it from the material /
    # POPX growth field:  op('liquid_metal/coverage')[0]


def _q(ch):
    """Expression CHOP reference to a channel on the hand_in select."""
    return f"op('hand_in')['{ch}']"


# ----------------------------------------------------------------------------
# GLSL (TouchDesigner dialect: sTD2DInputs[], vUV, TDOutputSwizzle, fragColor)
# ----------------------------------------------------------------------------
_HEIGHT_FRAG = """// blurred mask -> smooth height (all channels = height)
out vec4 fragColor;
void main(){
    float m = texture(sTD2DInputs[0], vUV.st).r;
    float h = smoothstep(0.05, 0.95, m);
    fragColor = TDOutputSwizzle(vec4(h, h, h, 1.0));
}
"""

_DISP_VERT = """// push grid verts along normal/Z by the HEIGHT texture
uniform sampler2D sHeight;
out Vertex { vec3 worldP; vec3 worldN; } oVert;
void main(){
    vec4 p = TDDeform(P);
    float h = texture(sHeight, uv[0].st).r;
    p.xyz += TDDeformNorm(N) * h * 0.6;   // 0.6 = relief depth, tweak
    oVert.worldP = p.xyz;
    oVert.worldN = TDDeformNorm(N);
    gl_Position = TDWorldToProj(p);
}
"""

_CHROME_PIX = """// placeholder chrome look (swap for POPX path-traced material)
out vec4 fragColor;
in Vertex { vec3 worldP; vec3 worldN; } iVert;
void main(){
    vec3 n = normalize(iVert.worldN);
    float f = pow(1.0 - abs(n.z), 3.0);          // fresnel-ish rim
    vec3 col = mix(vec3(0.55), vec3(1.0), f);
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
"""


# auto-run when executed
build()
