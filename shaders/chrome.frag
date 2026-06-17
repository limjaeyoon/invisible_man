#version 330

in vec2 v_uv;
out vec4 frag;

uniform sampler2D u_frame;    // webcam (rgb) — the live "real you" layer
uniform sampler2D u_plate;    // empty-room capture (rgb) — the layer we dissolve TO
uniform sampler2D u_mask;     // chrome coverage (r) — where/how much chrome shows
uniform sampler2D u_height;   // silhouette dome (r) -> overall body curvature
uniform sampler2D u_matcap;   // chrome sphere -> the REFLECTION (environment)

uniform vec2  u_texel;
uniform float u_time;
uniform float u_base_plate;   // full-frame base dissolve: 0 = live camera, 1 = capture
uniform int   u_has_plate;
uniform float u_flow_speed;   // how fast the liquid flows
uniform float u_liquid_scale; // ripple size (bigger = finer ripples)
uniform float u_liquid_amp;   // ripple depth -> drives reflection/refraction
uniform float u_normal;       // overall surface strength
uniform float u_refract;      // background bend amount (px)
uniform float u_chroma;       // chromatic dispersion
uniform float u_fresnel;      // edge falloff
uniform float u_reflect;      // base chrome amount (0 glass .. 1 mirror)
uniform float u_rim;          // bright bevel edge
uniform float u_ior;          // index of refraction (water 1.33 .. glass 1.5)
uniform float u_absorb;       // Beer-Lambert absorption (liquid tint/depth)
uniform float u_spec;         // specular sheen (wet highlight)
uniform float u_warp;         // flow domain-warp (swirl)
uniform float u_bead;         // metaball droplet amount (liquid-metal beads)

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p){
    vec2 i = floor(p), f = fract(p);
    float a = hash(i), b = hash(i + vec2(1,0));
    float c = hash(i + vec2(0,1)), d = hash(i + vec2(1,1));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm(vec2 p){
    // few octaves -> smooth, flowing surface (more octaves = crinkly foil)
    float v = 0.0, a = 0.6;
    for(int i=0;i<3;i++){ v += a*noise(p); p = p*1.9 + 1.7; a *= 0.5; }
    return v;
}
// metaball droplets: nearest animated cell point -> rounded merging blobs
vec2 hash2(vec2 p){
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}
float worley(vec2 p, float t){
    vec2 ip = floor(p), fp = fract(p);
    float d = 1.0;
    for(int j=-1;j<=1;j++)
    for(int i=-1;i<=1;i++){
        vec2 g = vec2(i, j);
        vec2 o = 0.5 + 0.5 * sin(t + 6.2831 * hash2(ip + g));   // points drift
        vec2 r = g + o - fp;
        d = min(d, dot(r, r));
    }
    return sqrt(d);
}
float Hd(vec2 uv){ return texture(u_height, clamp(uv,0.0,1.0)).r; }
vec3 frame(vec2 uv){ return texture(u_frame, clamp(uv,0.0,1.0)).rgb; }
// background source for refraction: the clean plate if captured, else live frame
vec3 plate(vec2 uv){
    uv = clamp(uv, 0.0, 1.0);
    return mix(texture(u_frame, uv).rgb, texture(u_plate, uv).rgb, float(u_has_plate));
}

// liquid surface = body dome + animated flowing ripples
float surf(vec2 p, vec2 flow){
    // domain-warped, downward-biased flow -> swirling liquid that runs down
    vec2 w = vec2(noise(p * u_liquid_scale * 0.5 + flow),
                  noise(p * u_liquid_scale * 0.5 + flow + 5.2)) - 0.5;
    vec2 q = flow + w * u_warp;
    float liq = fbm(p * u_liquid_scale + q);
    // one gentle second layer for organic, non-repetitive motion
    liq += 0.35 * fbm(p * u_liquid_scale * 1.7 - q * 1.3);
    float h = Hd(p) * 1.2 + liq * u_liquid_amp;
    // raised, merging metaball droplets -> glossy liquid-metal beading
    if (u_bead > 0.001){
        float d = worley(p * u_liquid_scale * 0.9 + flow * 0.4, u_time * u_flow_speed * 0.6);
        h += u_bead * smoothstep(0.55, 0.0, d);
    }
    return h;
}

void main(){
    vec2 uv = v_uv;
    vec3 live = frame(uv);                 // the live "real you" layer
    // BASE = whole-frame layer that dissolves live camera -> empty-room capture.
    // Once it's the capture, a lagging chrome mask exposes empty room, not you.
    vec3 base = mix(live, texture(u_plate, clamp(uv, 0.0, 1.0)).rgb, u_base_plate);

    float cover = clamp(texture(u_mask, uv).r, 0.0, 1.0);   // chrome over the body
    if (cover < 0.004) {                   // no chrome here -> just the base layer
        frag = vec4(base, 1.0);
        return;
    }

    vec2 flow = vec2(0.15, -1.0) * (u_time * u_flow_speed);

    // surface normal from the rippling liquid.
    // central-difference SLOPE = delta / distance (the /2d is essential, else
    // the normals stay flat and nothing reflects).
    float dx = 1.5 * u_texel.x;
    float dy = 1.5 * u_texel.y;
    float sx = (surf(uv + vec2(dx,0.0), flow) - surf(uv - vec2(dx,0.0), flow)) / (2.0*dx);
    float sy = (surf(uv + vec2(0.0,dy), flow) - surf(uv - vec2(0.0,dy), flow)) / (2.0*dy);
    vec3 n = normalize(vec3(-sx * u_normal, -sy * u_normal, 1.0));

    float thick = clamp(Hd(uv), 0.0, 1.0);          // dome height as a thickness proxy
    vec3 V = vec3(0.0, 0.0, 1.0);                    // view dir (toward camera)
    vec3 Ir = vec3(0.0, 0.0, -1.0);                 // incident ray into the surface

    // REFRACTION with a real index of refraction + per-channel dispersion (a
    // prism): the ray bends more where the liquid is thicker, splitting colour.
    float disp = u_chroma * 0.06;
    vec2 oR = refract(Ir, n, 1.0 / max(1.0, u_ior - disp)).xy;
    vec2 oG = refract(Ir, n, 1.0 / u_ior).xy;
    vec2 oB = refract(Ir, n, 1.0 / (u_ior + disp)).xy;
    float k = u_refract * (0.4 + thick);
    vec3 refr = vec3(
        plate(uv + oR * k * u_texel).r,
        plate(uv + oG * k * u_texel).g,
        plate(uv + oB * k * u_texel).b);

    // BEER-LAMBERT absorption: thicker liquid dims and cools the room behind it,
    // so the body reads as a translucent volume rather than a flat cutout.
    vec3 ext = u_absorb * vec3(1.0, 0.6, 0.35);     // absorb red most -> cool tint
    refr *= exp(-ext * thick * 2.5);

    // REFLECTION (matcap environment), blended by Schlick fresnel from the IOR
    vec2 muv = vec2(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5);
    vec3 refl = texture(u_matcap, clamp(muv, 0.0, 1.0)).rgb;
    float F0 = pow((u_ior - 1.0) / (u_ior + 1.0), 2.0);
    float F = F0 + (1.0 - F0) * pow(1.0 - clamp(n.z, 0.0, 1.0), u_fresnel);
    float mix_amt = clamp(u_reflect + (1.0 - u_reflect) * F, 0.0, 1.0);
    vec3 glass = mix(refr, refl, mix_amt);

    // SPECULAR sheen (wet highlight) from a virtual key light
    vec3 H = normalize(normalize(vec3(0.5, 0.8, 0.9)) + V);
    glass += pow(max(dot(n, H), 0.0), 80.0) * u_spec * vec3(1.0);

    // bright meniscus rim at grazing angles (surface tension edge)
    glass += smoothstep(0.6, 1.0, F) * u_rim * vec3(0.95, 0.98, 1.0);

    // liquid figure composited over the base layer (live early, capture once set)
    frag = vec4(mix(base, glass, cover), 1.0);
}
