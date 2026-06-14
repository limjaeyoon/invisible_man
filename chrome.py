"""GPU chrome renderer: webcam frame + mask -> liquid-metal person.

Renders offscreen with moderngl and reads the result back so the rest of the
app can stay in plain OpenCV land (display, recording, etc).
"""
from pathlib import Path
import time
import numpy as np
import cv2
import moderngl

ROOT = Path(__file__).resolve().parent

# fullscreen quad: pos(x,y) + uv(u,v)
QUAD = np.array([
    -1, -1, 0, 0,
     1, -1, 1, 0,
    -1,  1, 0, 1,
     1,  1, 1, 1,
], dtype="f4")


class ChromeRenderer:
    def __init__(self, width, height, matcap="chrome"):
        self.w, self.h = width, height
        self.ctx = moderngl.create_standalone_context(require=330)

        vert = (ROOT / "shaders" / "quad.vert").read_text()
        frag = (ROOT / "shaders" / "chrome.frag").read_text()
        self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

        vbo = self.ctx.buffer(QUAD.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(vbo, "2f 2f", "in_pos", "in_uv")]
        )

        # textures (filled per frame)
        self.tex_frame = self.ctx.texture((width, height), 3)
        self.tex_mask = self.ctx.texture((width, height), 1, dtype="f4")    # chrome curtain
        self.tex_region = self.ctx.texture((width, height), 1, dtype="f4")  # silhouette
        self.tex_height = self.ctx.texture((width, height), 1, dtype="f1")
        for t in (self.tex_frame, self.tex_mask, self.tex_region, self.tex_height):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False   # clamp so refraction doesn't wrap the edges
            t.repeat_y = False

        # clean background plate (room without you); empty until captured
        self.tex_plate = self.ctx.texture((width, height), 3)
        self.tex_plate.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_plate.repeat_x = False
        self.tex_plate.repeat_y = False
        self.has_plate = 0

        mc = cv2.imread(str(ROOT / "assets" / "matcaps" / f"{matcap}.png"))
        mc = cv2.cvtColor(mc, cv2.COLOR_BGR2RGB)
        self.tex_matcap = self.ctx.texture((mc.shape[1], mc.shape[0]), 3, mc.tobytes())
        self.tex_matcap.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_matcap.build_mipmaps()

        # bind texture units
        self.prog["u_frame"] = 0
        self.prog["u_mask"] = 1
        self.prog["u_height"] = 2
        self.prog["u_matcap"] = 3
        self.prog["u_plate"] = 4
        self.prog["u_region"] = 5
        self.prog["u_has_plate"] = 0
        self.prog["u_texel"] = (1.0 / width, 1.0 / height)

        # offscreen target
        self.color = self.ctx.texture((width, height), 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color])

        self.t0 = time.time()
        # default look — invisible glass man (values from your screenshot)
        self.params = dict(
            u_flow_speed=0.45, u_liquid_scale=9.0, u_liquid_amp=1.40,
            u_normal=0.18, u_refract=252.0, u_chroma=0.06,
            u_fresnel=2.0, u_reflect=0.0, u_rim=2.5,
            u_base_plate=0.0,
        )

    def set(self, **kw):
        self.params.update(kw)

    def capture_plate(self, frame_rgb):
        """Store the current frame as the clean background plate."""
        self.tex_plate.write(np.ascontiguousarray(frame_rgb, np.uint8).tobytes())
        self.has_plate = 1
        self.prog["u_has_plate"] = 1

    def load_matcap(self, name):
        path = ROOT / "assets" / "matcaps" / f"{name}.png"
        if not path.exists():
            return False
        mc = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        self.tex_matcap.release()
        self.tex_matcap = self.ctx.texture((mc.shape[1], mc.shape[0]), 3, mc.tobytes())
        self.tex_matcap.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_matcap.build_mipmaps()
        return True

    def render(self, frame_rgb, region_f, cover_f, height_u8):
        self.tex_frame.write(np.ascontiguousarray(frame_rgb, np.uint8).tobytes())
        self.tex_mask.write(np.ascontiguousarray(cover_f, np.float32).tobytes())
        self.tex_region.write(np.ascontiguousarray(region_f, np.float32).tobytes())
        self.tex_height.write(np.ascontiguousarray(height_u8, np.uint8).tobytes())

        self.tex_frame.use(0)
        self.tex_mask.use(1)
        self.tex_height.use(2)
        self.tex_matcap.use(3)
        self.tex_plate.use(4)
        self.tex_region.use(5)

        for k, v in self.params.items():
            self.prog[k] = v
        self.prog["u_time"] = time.time() - self.t0

        self.fbo.use()
        self.ctx.clear(0, 0, 0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        data = self.fbo.read(components=3, dtype="f1")
        out = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3)
        # Textures are uploaded row0-first and the quad maps uv.y=0 -> bottom of
        # the framebuffer, so the readback already comes out upright. No flip.
        return out
