"""
silverchrome — pinch to turn invisible. Live on your webcam.

    python main.py

How it works:
    1. Press  c  while you are OUT of frame to capture the empty room.
    2. Step back in. You look normal.
    3. Pinch (touch thumb + index on EITHER hand) -> the invisible liquid-glass
       coverage slowly spreads across your whole body. Pinch again to reverse.

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
from matte import RVMatte, BGMatte, ThreadedMatte, keep_significant, height_from_mask
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
LEVEL_SECONDS = 1.4          # how long the coverage takes to spread in/out


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

    # RobustVideoMatting segments the person directly from each frame — no
    # background plate needed, robust at angles, and it keeps disconnected
    # limbs (a hand entering from the side). BGMatte stays available but is
    # fragile (needs a perfect plate) and was returning empty mattes.
    matter = ThreadedMatte(RVMatte())
    pinch = ThreadedPinch()
    ren = ChromeRenderer(w, h, matcap="chrome")
    noise = make_noise(h, w)
    show_matte = False          # 'v' debug: view the raw RVM matte

    mirror = True
    recording = False
    writer = None

    level = 0.0                 # current matte coverage 0..1
    target = 0.0                # where we're heading
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

        # --- pinch press -> toggle matte-coverage target ---
        # Detection runs on a background thread (see ThreadedPinch); we just
        # hand it the latest frame and consume any press it has latched.
        if frame_i % 2 == 0:                 # feed hands every other frame
            pinch.submit(shared, now)
        if pinch.poll():
            target = 0.0 if target > 0.5 else 1.0

        # animate the coverage LEVEL toward target (slow spread in/out)
        step = dt / LEVEL_SECONDS
        level += np.clip(target - level, -step, step)
        level = float(np.clip(level, 0.0, 1.0))

        # effect applies wherever the matte selects. A captured plate (c) makes
        # the refraction truly see-through; without one it bends the live frame
        # (wavy glass), so you still see the effect work immediately.
        ren.params["u_morph"] = 1.0

        # BGMv2 matte on a worker thread. Only pay for inference while the effect
        # is active (or ramping) — when fully off, leave the CPU to nothing so
        # the active path runs cooler and fresher.
        active = target > 0.5 or level > 0.001
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

        if level <= 0.001:
            mask = np.zeros((h, w), np.float32)
        else:
            reveal = np.clip((level - noise) / 0.18 + 0.5, 0.0, 1.0)
            mask = body * reveal
        height = height_from_mask(mask)
        out_rgb = ren.render(frame_rgb, mask, height)
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
                ren.capture_plate(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
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
            target = 0.0 if target > 0.5 else 1.0
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
