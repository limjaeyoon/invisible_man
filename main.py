"""
silverchrome — pinch to turn invisible. Live on your webcam.

    python main.py

How it works (liquid-metal morph):
    1. (Optional) Press  c  out of frame to capture the room -> the chrome then
       mirrors a clean empty room instead of reflecting yourself.
    2. Pinch (thumb + index on either hand, presented) -> liquid metal erupts
       from your pinch and floods across your body as a connected front with a
       rolling chrome lip, until you become a mirror T-1000 figure in the room.
    3. Pinch again to morph back. (SPACE instead = it wells up all over.)

Controls:
    c         capture room for a clean mirror reflection (optional, out of frame)
    SPACE     morph manually (wells up all over instead of from a hand)
    t / g     matte edge tightness (tighter / fuller)
    a / s     matte temporal smoothing (steadier / more responsive)
    o / i     chrome peak cover width during the morph (wider / narrower)
    j         toggle hand skeleton overlay (drawn under the chrome)
    x         view the 3D depth map (Depth Anything) — verify it loads/looks right
    p         toggle debug perf log (render/matte/hand/depth fps + state)
    q / ESC   quit

Pinch only registers when the hand is presented (open & raised), so a
relaxed hand resting in view won't trigger it.
    r         record on / off  (output/)
    90        chrome amount        ,. refraction
    -=        ripple depth         ;' ripple size
    []        flow speed           kl chromatic aberration
    fd        edge rim             m mirror
    12 IOR    34 absorption        56 specular   78 flow swirl
    bn droplets   yu room-reflection   90 mirror amount
"""
import time
from pathlib import Path
import cv2
import numpy as np

from capture import Camera
from matte import SelfieMatte, RVMatte, BGMatte, ThreadedMatte, keep_significant, height_from_mask
from chrome import ChromeRenderer
from gesture import ThreadedPinch, render_hands
from depth import ThreadedDepth, DepthEstimator


def make_noise(h, w, seed=7):
    """Static organic field in [0,1] used to dissolve the coverage in/out."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 0.5, 0.0
    for cells in (3, 6, 12, 24):
        small = rng.random((cells, cells)).astype(np.float32)
        acc += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    acc /= tot
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)

CAM_INDEX = 0
WIDTH, HEIGHT = 1280, 720
OUT_DIR = Path(__file__).resolve().parent / "output"
TRANSITION_SECONDS = 1.6     # total sweep time
COVER_FRAC = 0.7             # share spent growing/lifting the chrome (the visible
                            # part); the rest is the near-invisible base dissolve
SETTLE_GROW = 0.0           # chrome width once settled (0 = hugs body; ~1 = margin)
WIDTH_LEAD = 0.15           # start swelling this much (in tp) before the morph
WIDTH_HOLD_END = 0.9        # hold peak width until here, then thin to SETTLE_GROW
FRONT_RIDGE = 0.7           # height of the rolling liquid lip at the advancing front


def reveal_field(cov, noise):
    """Spatial chrome curtain for a global coverage `cov` in [0,1].

    Dissolves in/out organically along the static noise field, but normalized
    so cov=0 -> nothing and cov=1 -> fully covered everywhere. Full coverage at
    the peak is what hides the live<->plate swap underneath.
    """
    return np.clip((1.4 * cov - noise) / 0.4, 0.0, 1.0)


def grow_mask(m, px, feather=2.0):
    """Grow the body mask outward by `px` pixels (fractional, feathered) so the
    chrome fully covers the real body edge — otherwise a rim of you sits outside
    the chrome and visibly dissolves when the base morphs to the capture. Uses a
    distance transform so `px` can be tuned in fine (sub-pixel) steps. Keep it
    modest or the figure puffs up (Michelin man)."""
    if px <= 0.05:
        return m
    binm = (m > 0.5).astype(np.uint8)
    if not binm.any():
        return m
    dist_out = cv2.distanceTransform(1 - binm, cv2.DIST_L2, 3)   # 0 inside, grows out
    wide = np.clip(1.0 - (dist_out - px) / max(0.5, feather), 0.0, 1.0)
    return np.maximum(m, wide).astype(np.float32)


def width_profile(tp, peak, settle=SETTLE_GROW):
    """Chrome cover-width over the transition: thin -> swell across the base
    morph -> thin to `settle`. A trapezoid (with a lead-in before the morph and
    a thin-out as it settles), eased on the ramps. Because it's a function of
    `tp`, reverse mirrors it: the figure swells to cover the returning rim, then
    the chrome lifts at the settled width."""
    up0, up1 = COVER_FRAC - WIDTH_LEAD, COVER_FRAC
    if tp <= up0:
        b = 0.0
    elif tp < up1:
        b = (tp - up0) / (up1 - up0)
    elif tp <= WIDTH_HOLD_END:
        b = 1.0
    elif tp < 1.0:
        b = 1.0 - (tp - WIDTH_HOLD_END) / (1.0 - WIDTH_HOLD_END)
    else:
        b = 0.0
    b = b * b * (3.0 - 2.0 * b)                      # ease the ramps
    return settle + (peak - settle) * b


def pinch_seed_points(hands, w, h):
    """Screen-space seed(s) at each hand's thumb-index pinch point."""
    pts = []
    for lm in hands:
        pts.append(((lm[4].x + lm[8].x) * 0.5 * w,
                    (lm[4].y + lm[8].y) * 0.5 * h))
    return pts


def body_seed_points(body, n=8):
    """Random points inside the body -> many fronts that well up and merge
    (the SPACE / no-hand fallback)."""
    ys, xs = np.where(body > 0.5)
    if xs.size == 0:
        return []
    idx = np.random.choice(xs.size, size=min(n, xs.size), replace=False)
    return list(zip(xs[idx].tolist(), ys[idx].tolist()))


def seed_arrival(body, seeds, noise, tendril=0.18):
    """Normalized distance from the morph seed(s), 0 at the seed -> 1 far away.

    This drives a *connected advancing front* (the liquid crawls outward from
    where it erupted) instead of random islands, so the coverage reads as a
    morph rather than a dissolve. A little noise wiggles the front into tendrils.
    """
    h, w = body.shape[:2]
    m = np.ones((h, w), np.uint8)
    placed = False
    for (x, y) in seeds:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            m[yi, xi] = 0
            placed = True
    if not placed:
        m[h // 2, w // 2] = 0
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    inside = d[body > 0.4]
    dmax = float(inside.max()) if inside.size else float(d.max())
    a = d / (dmax + 1e-6) + (noise - 0.5) * tendril
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def plate_gain_match(plate_rgb, live_rgb, body, prev):
    """Per-channel gain that makes the captured plate match the live feed's
    current exposure / white balance.

    The webcam re-meters when you step in/out of frame, so the plate was shot at
    a different brightness/tint than the live image — making the live->capture
    dissolve visible. We compare the two over their SHARED background (pixels the
    matte says aren't you) and scale the plate to match, so the swap is seamless.
    """
    s = (160, 90)
    p = cv2.resize(plate_rgb, s).astype(np.float32)
    l = cv2.resize(live_rgb, s).astype(np.float32)
    b = cv2.resize(body, s)
    bg = b < 0.15                                   # shared, person-free pixels
    if bg.sum() < bg.size * 0.2:                    # too little background to trust
        return prev
    g = np.ones(3, np.float32)
    for c in range(3):
        pm = float(np.median(p[..., c][bg]))
        lm = float(np.median(l[..., c][bg]))
        g[c] = lm / (pm + 1.0)
    return np.clip(g, 0.6, 1.7)


def capture_plate(cap, mirror, n=7):
    """Average a short burst into a clean background plate.

    A single frame carries sensor noise, which weakens the live-vs-plate
    contrast the matte relies on. Averaging a handful of frames denoises the
    plate, so the person separates more cleanly and accurately.
    """
    acc, got = None, 0
    deadline = time.time() + 1.0
    while got < n and time.time() < deadline:
        ok, f = cap.read()
        if not ok or f is None:
            time.sleep(0.01)
            continue
        if mirror:
            f = cv2.flip(f, 1)
        acc = f.astype(np.float32) if acc is None else acc + f
        got += 1
        time.sleep(0.03)                 # space the reads out for distinct frames
    if acc is None:
        return None
    return (acc / got).astype(np.uint8)


def main():
    cap = Camera(CAM_INDEX, WIDTH, HEIGHT)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index %d)." % CAM_INDEX)

    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit("Webcam opened but returned no frame.")
    h, w = frame.shape[:2]

    # MediaPipe Selfie Segmentation: human-only, so it ignores furniture and
    # other objects (RVM was spilling onto them). Fast/low-latency; soft edges
    # are fine since the chrome + captured-room base hide them. RVMatte/BGMatte
    # stay importable as alternatives.
    matter = ThreadedMatte(SelfieMatte())
    pinch = ThreadedPinch()
    ren = ChromeRenderer(w, h, matcap="chrome")
    noise = make_noise(h, w)
    show_matte = False          # 'v' debug: view the raw RVM matte

    # Path A: real 3D depth of you (Depth Anything V2). Optional — if it can't
    # load (no model / slow onnxruntime), we run without it.
    depther = None
    try:
        depther = ThreadedDepth(DepthEstimator())
    except Exception as e:
        print("depth unavailable (", e, "); running without 3D depth.")
    show_depth = False          # 'x' debug: view the depth map

    mirror = True
    show_hands = True           # sci-fi hand skeleton overlay
    debug = False               # 'p': periodic perf/state log
    recording = False
    writer = None

    # transition state. tp in [0,1]: 0 = plain live you, 1 = settled chrome figure
    # over the captured empty room. First half (0..0.5) grows the chrome over the
    # live you; second half (0.5..1) dissolves the whole base live->capture.
    chrome_on = False           # toggle target
    tp = 0.0                    # animated progress toward chrome_on (front sweep)
    pending_toggle = False      # a pinch/SPACE waiting to be applied
    morph_active = False        # has the current morph's seed been chosen?
    seeds = []                  # screen-space origin point(s) of the liquid

    plate_rgb0 = None           # the captured plate (rgb), before exposure matching
    plate_gain = np.ones(3, np.float32)   # smoothed live<->plate exposure/WB match
    mask_grow = 4.0             # PEAK chrome width during the morph (o/i, 0.1 steps);
                               # thins to SETTLE_GROW once settled
    frame_i = 0

    matte_smooth = 1.0          # temporal EMA on the matte (1.0 = off, no lag/trails)
    body_prev = None

    fps_t, fps_n, fps = time.time(), 0, 0.0
    dbg_t, dbg_mc, dbg_pc, dbg_dc = time.time(), 0, 0, 0
    last = time.time()
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:                # no fresh frame yet; stay responsive
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.time()
        dt = now - last
        last = now

        # one copy shared by both worker threads (they only read it)
        frame_i += 1
        shared = frame.copy()

        # --- pinch / SPACE -> request a transition (latched, consumed here) ---
        # Feed the hand tracker every frame (its own thread) for lowest latency.
        pinch.submit(shared, now)
        if pinch.poll():
            pending_toggle = True

        if pending_toggle:
            pending_toggle = False
            chrome_on = not chrome_on            # reversible mid-morph

        # the front sweeps 0..1 across the body; you stay a chrome figure in the
        # live room (no background dissolve) and pinch again to morph back.
        step = dt / TRANSITION_SECONDS
        tp += float(np.clip((1.0 if chrome_on else 0.0) - tp, -step, step))
        tp = float(np.clip(tp, 0.0, 1.0))
        front = float(tp * tp * (3.0 - 2.0 * tp))    # smoothstep front level
        base_plate = 0.0                             # T-1000: mirror chrome over live room
        ren.params["u_base_plate"] = base_plate

        # Run the matte whenever any chrome is on screen so the figure tracks you.
        active = tp > 0.001
        if active:
            matter.submit(shared)
        if depther is not None and (active or show_depth):
            depther.submit(shared)
        body = matter.get()
        if body is None:
            body = np.zeros((h, w), np.float32)
        else:
            body = keep_significant(body)
            if matte_smooth < 0.999 and body_prev is not None \
                    and body_prev.shape == body.shape:
                body = matte_smooth * body + (1.0 - matte_smooth) * body_prev
            body_prev = body

        # On the rising edge of a morph, pick where the liquid erupts: the pinch
        # hand(s), or (no hand / SPACE) many points that well up across the body.
        if chrome_on and not morph_active:
            morph_active = True
            seeds = pinch_seed_points(pinch.get_hands(), w, h) or body_seed_points(body, 8)
        elif not chrome_on and tp <= 0.001:
            morph_active = False

        # keep the room reflection exposure/WB-matched to the live feed
        if active and plate_rgb0 is not None:
            g = plate_gain_match(plate_rgb0, frame_rgb, body, plate_gain)
            plate_gain = 0.85 * plate_gain + 0.15 * g
            matched = np.clip(plate_rgb0.astype(np.float32) * plate_gain,
                              0, 255).astype(np.uint8)
            ren.write_plate(matched)

        if front <= 0.001:                              # no chrome
            cover = np.zeros((h, w), np.float32)
            height = np.zeros((h, w), np.uint8)
        elif front >= 0.999:                            # fully chrome (settled)
            cover = grow_mask(body, width_profile(tp, mask_grow))
            height = height_from_mask(cover)
        else:                                           # mid-morph: advancing front
            arrival = seed_arrival(body, seeds, noise)  # connected front from the seed
            cover = grow_mask(body, width_profile(tp, mask_grow)) * reveal_field(front, arrival)
            height = height_from_mask(cover)
            # rolling liquid lip: raise a ridge where the front is mid-coverage
            ridge = np.exp(-((cover - 0.5) ** 2) / 0.045) * (FRONT_RIDGE * 255.0)
            height = np.clip(height.astype(np.float32) + ridge, 0, 255).astype(np.uint8)
        out_rgb = ren.render(frame_rgb, cover, height)
        out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        if show_matte:                     # debug: see exactly what the matte selects
            dbg = cv2.cvtColor((body * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(out, 0.4, dbg, 0.6, 0)

        if show_depth and depther is not None:   # debug: view the 3D depth map
            dm = depther.get()
            if dm is not None:
                out = cv2.applyColorMap((np.clip(dm, 0, 1) * 255).astype(np.uint8),
                                        cv2.COLORMAP_INFERNO)

        if show_hands:                     # hand skeleton, part of the live layer
            hands = pinch.get_hands()
            if hands:
                overlay, alpha = render_hands(h, w, hands)
                # under the chrome (1-cover) AND gone once invisible (1-base_plate):
                # it belongs to the live-you layer, so it dissolves with it.
                a = (alpha.astype(np.float32) / 255.0) \
                    * (1.0 - np.clip(cover, 0, 1)) * (1.0 - base_plate)
                a = a[..., None]
                out = (out.astype(np.float32) * (1 - a)
                       + overlay.astype(np.float32) * a).astype(np.uint8)

        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t, fps_n = now, 0

        if debug and now - dbg_t >= 1.0:
            iv = now - dbg_t
            mfps = (matter.count - dbg_mc) / iv
            pfps = (pinch.count - dbg_pc) / iv
            state = "live" if tp <= 0.001 else ("chrome" if tp >= 0.999 else "morph")
            dms = depther.ms if depther is not None else 0.0
            dfps = (depther.count - dbg_dc) / iv if depther is not None else 0.0
            print("[dbg] render %2.0ffps | matte %3.0fms %2.0ffps | hand %3.0fms %2.0ffps"
                  " | depth %4.0fms %2.0ffps | %-6s tp=%.2f hands=%d"
                  % (fps, matter.ms, mfps, pinch.ms, pfps, dms, dfps, state, tp,
                     len(pinch.get_hands())))
            dbg_t, dbg_mc, dbg_pc = now, matter.count, pinch.count
            dbg_dc = depther.count if depther is not None else 0

        if recording and writer is not None:
            writer.write(out)

        cv2.imshow("silverchrome", out)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("c"):
            plate = capture_plate(cap, mirror)    # averaged, denoised plate
            if plate is not None:
                matter.set_background(plate)      # BGMv2 background
                plate_rgb0 = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
                plate_gain = np.ones(3, np.float32)
                ren.capture_plate(plate_rgb0)
                body_prev = None                  # drop smoothing history
                print("background plate captured (averaged).")
            else:
                print("plate capture failed (no frames).")
        elif k == ord("g"):                 # looser matte edge (fuller)
            matter.thr = max(0.0, matter.thr - 0.05)
        elif k == ord("t"):                 # tighter matte edge
            matter.thr = min(0.9, matter.thr + 0.05)
        elif k == ord("a"):                 # more temporal smoothing (steadier)
            matte_smooth = max(0.2, matte_smooth - 0.05)
        elif k == ord("s"):                 # less smoothing (more responsive)
            matte_smooth = min(1.0, matte_smooth + 0.05)
        elif k == ord("o"):                 # chrome wider (covers body edge)
            mask_grow = min(40.0, mask_grow + 0.1)
            print("cover width: %.1f" % mask_grow)
        elif k == ord("i"):                 # chrome narrower (less puffy)
            mask_grow = max(0.0, mask_grow - 0.1)
            print("cover width: %.1f" % mask_grow)
        elif k == ord(" "):
            pending_toggle = True
        elif k == ord("m"):
            mirror = not mirror
        elif k == ord("v"):
            show_matte = not show_matte
        elif k == ord("j"):
            show_hands = not show_hands
        elif k == ord("x"):
            show_depth = not show_depth
        elif k == ord("p"):
            debug = not debug
            print("debug logging:", "on" if debug else "off")
        elif k == ord("["):
            ren.params["u_flow_speed"] = max(0.0, ren.params["u_flow_speed"] - 0.05)
        elif k == ord("]"):
            ren.params["u_flow_speed"] += 0.05
        elif k == ord("-"):
            ren.params["u_liquid_amp"] = max(0.0, ren.params["u_liquid_amp"] - 0.05)
        elif k == ord("="):
            ren.params["u_liquid_amp"] += 0.05
        elif k == ord(";"):
            ren.params["u_liquid_scale"] = max(1.0, ren.params["u_liquid_scale"] - 1.0)
        elif k == ord("'"):
            ren.params["u_liquid_scale"] += 1.0
        elif k == ord(","):
            ren.params["u_refract"] = max(0.0, ren.params["u_refract"] - 6.0)
        elif k == ord("."):
            ren.params["u_refract"] += 6.0
        elif k == ord("k"):
            ren.params["u_chroma"] = max(0.0, ren.params["u_chroma"] - 0.02)
        elif k == ord("l"):
            ren.params["u_chroma"] += 0.02
        elif k == ord("9"):
            ren.params["u_reflect"] = max(0.0, ren.params["u_reflect"] - 0.05)
        elif k == ord("0"):
            ren.params["u_reflect"] = min(1.0, ren.params["u_reflect"] + 0.05)
        elif k == ord("d"):
            ren.params["u_rim"] = max(0.0, ren.params["u_rim"] - 0.1)
        elif k == ord("f"):
            ren.params["u_rim"] += 0.1
        elif k == ord("1"):                 # IOR (glass density) down/up
            ren.params["u_ior"] = max(1.0, ren.params["u_ior"] - 0.02)
        elif k == ord("2"):
            ren.params["u_ior"] = min(2.2, ren.params["u_ior"] + 0.02)
        elif k == ord("3"):                 # Beer-Lambert absorption (liquid tint)
            ren.params["u_absorb"] = max(0.0, ren.params["u_absorb"] - 0.1)
        elif k == ord("4"):
            ren.params["u_absorb"] += 0.1
        elif k == ord("5"):                 # specular sheen (wet highlight)
            ren.params["u_spec"] = max(0.0, ren.params["u_spec"] - 0.05)
        elif k == ord("6"):
            ren.params["u_spec"] += 0.05
        elif k == ord("7"):                 # flow swirl (domain warp)
            ren.params["u_warp"] = max(0.0, ren.params["u_warp"] - 0.1)
        elif k == ord("8"):
            ren.params["u_warp"] += 0.1
        elif k == ord("b"):                 # fewer metal droplets
            ren.params["u_bead"] = max(0.0, ren.params["u_bead"] - 0.05)
        elif k == ord("n"):                 # more metal droplets
            ren.params["u_bead"] += 0.05
        elif k == ord("y"):                 # less room reflection (more matcap)
            ren.params["u_env"] = max(0.0, ren.params["u_env"] - 0.05)
        elif k == ord("u"):                 # more room reflection (mirror)
            ren.params["u_env"] = min(1.0, ren.params["u_env"] + 0.05)
        elif k == ord("r"):
            recording = not recording
            if recording:
                OUT_DIR.mkdir(exist_ok=True)
                fn = OUT_DIR / time.strftime("chrome_%Y%m%d_%H%M%S.mp4")
                rec_fps = float(round(fps)) if fps > 1.0 else 30.0
                writer = cv2.VideoWriter(
                    str(fn), cv2.VideoWriter_fourcc(*"mp4v"), rec_fps, (w, h))
                print("recording ->", fn, "@ %.0f fps" % rec_fps)
            elif writer is not None:
                writer.release()
                writer = None
                print("saved.")

    if writer is not None:
        writer.release()
    matter.close()
    pinch.close()
    if depther is not None:
        depther.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
