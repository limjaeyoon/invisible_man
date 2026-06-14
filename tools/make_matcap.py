"""
Generate a chrome matcap (material-capture sphere image) procedurally.

A matcap encodes how a material looks from every surface-normal direction,
packed into a sphere. The chrome shader samples it by normal.xy. Swap this
image to change the metal entirely (gold, oil-slick, etc).

Usage:
    python tools/make_matcap.py            # -> assets/matcaps/chrome.png
    python tools/make_matcap.py gold       # tint variants
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

S = 512  # texture size


def smooth_band(x, center, width):
    """Smooth bright band centered at `center`."""
    return np.exp(-((x - center) ** 2) / (2 * width ** 2))


def build(tint=(1.0, 1.0, 1.0), name="chrome"):
    # pixel grid -> sphere normals
    ys, xs = np.mgrid[0:S, 0:S]
    nx = (xs / (S - 1)) * 2 - 1
    ny = -((ys / (S - 1)) * 2 - 1)  # +y up
    r2 = nx * nx + ny * ny
    inside = r2 <= 1.0
    nz = np.sqrt(np.clip(1.0 - r2, 0, 1))

    elev = ny  # -1 bottom .. +1 top, the main reflection axis

    # --- chrome environment as a function of elevation (vertical gradient) ---
    # high contrast is what reads as METAL: dark sky, blazing horizon, dark
    # band, bright floor. Low contrast looks like gray clay.
    v = np.zeros_like(elev)
    v += 0.40 * np.clip(elev, 0, 1) ** 1.4        # sky: dim, darkens toward top
    v += 1.30 * smooth_band(elev, 0.06, 0.045)     # blazing thin horizon
    v += 0.65 * np.clip(-elev - 0.35, 0, 1) ** 1.1 # bright floor toward bottom
    v += 0.06                                       # base fill
    v = np.clip(v, 0, 1.4)

    # deep dark band just below horizon -> the hard "metal" snap
    v *= 1.0 - 0.70 * smooth_band(elev, -0.16, 0.10)
    # subtle second reflection line lower down
    v += 0.25 * smooth_band(elev, -0.55, 0.05)

    # --- specular highlights (sharp light sources) ---
    def spec(lx, ly, power, amp):
        lz = np.sqrt(max(0.0, 1 - lx * lx - ly * ly))
        d = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
        return amp * d ** power
    v += spec(-0.45, 0.55, 200, 1.2)   # main key light, upper-left
    v += spec(0.5, 0.2, 600, 0.8)      # tight glint, right
    v = np.clip(v, 0, 1.4)

    # slight cool tint in sky, neutral elsewhere -> believable chrome
    rgb = np.stack([v, v, v], axis=-1).astype(np.float32)
    cool = np.clip(elev, 0, 1)[..., None] * np.array([0.0, 0.02, 0.06])
    rgb = np.clip(rgb - cool, 0, 1.4)

    rgb = rgb * np.array(tint, dtype=np.float32)
    rgb = np.clip(rgb, 0, 1)

    # gentle filmic-ish rolloff so highlights don't clip ugly
    rgb = rgb / (rgb + 0.25) * 1.25
    rgb = np.clip(rgb, 0, 1)

    img = (rgb * 255).astype(np.uint8)
    # outside the disc: replicate edge so any stray sample stays metallic
    img[~inside] = img[inside].mean(axis=0).astype(np.uint8)

    out = Path(__file__).resolve().parents[1] / "assets" / "matcaps" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    print("wrote", out)


TINTS = {
    "chrome": (1.0, 1.0, 1.0),
    "gold": (1.15, 0.92, 0.55),
    "mercury": (0.85, 0.88, 0.95),
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "chrome"
    build(TINTS.get(which, (1, 1, 1)), which)
