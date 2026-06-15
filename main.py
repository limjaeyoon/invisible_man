"""
silverchrome — pinch to turn invisible. Live on your webcam.

    python main.py

How it works (two layers):
    1. Press  c  while you are OUT of frame to capture the empty room (required).
    2. Step back in. You look normal.
    3. Pinch (thumb + index on EITHER hand) -> liquid chrome grows over the real
       you; once it covers you, the WHOLE background dissolves from live camera
       into the empty-room capture. You stay as a chrome figure standing in the
       captured room, so a lagging mask never exposes your real body. Pinch again
       to dissolve the room back to live and lift the chrome.

Controls:
    c         capture background plate (do this once, out of frame; averaged)
    SPACE     toggle coverage manually (same as a pinch)
    t / g     matte edge tightness (tighter / fuller)
    a / s     matte temporal smoothing (steadier / more responsive)
    q / ESC   quit
    r         record on / off  (output/)
    90        chrome amount        ,. refraction
    -=        ripple depth         ;' ripple size
    []        flow speed           kl chromatic aberration
    fd        edge rim             m mirror
"""
import time
from pathlib import Path
import cv2
import numpy as np

from capture import Camera
from matte import SelfieMatte, RVMatte, BGMatte, ThreadedMatte, keep_significant, height_from_mask
from chrome import ChromeRenderer
from gesture import ThreadedPinch


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
TRANSITION_SECONDS = 2.0     # chrome grows (first half) then base dissolves (second half)


def reveal_field(cov, noise):
    """Spatial chrome curtain for a global coverage `cov` in [0,1].

    Dissolves in/out organically along the static noise field, but normalized
    so cov=0 -> nothing and cov=1 -> fully covered everywhere. Full coverage at
    the peak is what hides the live<->plate swap underneath.
    """
    return np.clip((1.4 * cov - noise) / 0.4, 0.0, 1.0)


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

    mirror = True
    recording = False
    writer = None

    # transition state. tp in [0,1]: 0 = plain live you, 1 = settled chrome figure
    # over the captured empty room. First half (0..0.5) grows the chrome over the
    # live you; second half (0.5..1) dissolves the whole base live->capture.
    chrome_on = False           # toggle target
    tp = 0.0                    # animated progress toward chrome_on
    pending_toggle = False      # a pinch/SPACE waiting to be applied

    plate_rgb0 = None           # the captured plate (rgb), before exposure matching
    plate_gain = np.ones(3, np.float32)   # smoothed live<->plate exposure/WB match
    frame_i = 0

    matte_smooth = 1.0          # temporal EMA on the matte (1.0 = off, no lag/trails)
    body_prev = None

    fps_t, fps_n, fps = time.time(), 0, 0.0
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
        # Detection runs on a background thread (see ThreadedPinch).
        if frame_i % 2 == 0:                 # feed hands every other frame
            pinch.submit(shared, now)
        if pinch.poll():
            pending_toggle = True

        if pending_toggle:
            pending_toggle = False
            if not chrome_on and not ren.has_plate:
                print("capture the empty room first: step out and press c.")
            else:
                chrome_on = not chrome_on        # reversible mid-transition

        # animate tp toward the target. Two phases, each ~half the time:
        #   tp 0.0..0.5  chrome grows over the LIVE you   (base still live)
        #   tp 0.5..1.0  base dissolves live -> capture   (chrome stays full)
        step = dt / TRANSITION_SECONDS
        tp += float(np.clip((1.0 if chrome_on else 0.0) - tp, -step, step))
        tp = float(np.clip(tp, 0.0, 1.0))

        cov_t = np.clip(tp / 0.5, 0.0, 1.0)              # chrome coverage phase
        base_t = np.clip((tp - 0.5) / 0.5, 0.0, 1.0)     # base dissolve phase
        cov = float(cov_t * cov_t * (3.0 - 2.0 * cov_t))         # smoothstep
        base_plate = float(base_t * base_t * (3.0 - 2.0 * base_t))
        ren.params["u_base_plate"] = base_plate

        # Run the matte whenever any chrome is on screen so the figure tracks you.
        active = tp > 0.001
        if active:
            matter.submit(shared)
        body = matter.get()
        if body is None:
            body = np.zeros((h, w), np.float32)
        else:
            body = keep_significant(body)
            # temporal smoothing: steadies edges and stops near-threshold
            # disconnected blobs from popping in/out. alpha=1.0 disables it.
            if matte_smooth < 0.999 and body_prev is not None \
                    and body_prev.shape == body.shape:
                body = matte_smooth * body + (1.0 - matte_smooth) * body_prev
            body_prev = body

        # keep the plate exposure/WB-matched to the live feed so the dissolve and
        # the settled background read identically (no visible lighting jump).
        if active and plate_rgb0 is not None:
            g = plate_gain_match(plate_rgb0, frame_rgb, body, plate_gain)
            plate_gain = 0.85 * plate_gain + 0.15 * g
            matched = np.clip(plate_rgb0.astype(np.float32) * plate_gain,
                              0, 255).astype(np.uint8)
            ren.write_plate(matched)

        if cov <= 0.001:
            cover = np.zeros((h, w), np.float32)
        else:
            cover = body * reveal_field(cov, noise)
        height = height_from_mask(cover)
        out_rgb = ren.render(frame_rgb, cover, height)
        out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        if show_matte:                     # debug: see exactly what RVM selects
            dbg = cv2.cvtColor((body * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(out, 0.4, dbg, 0.6, 0)

        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t, fps_n = now, 0

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
        elif k == ord(" "):
            pending_toggle = True
        elif k == ord("m"):
            mirror = not mirror
        elif k == ord("v"):
            show_matte = not show_matte
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
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
