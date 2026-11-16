"""Chroma carries no data; colorize() guarantees luma is untouched."""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, minimum_filter, uniform_filter
from .codec import CANVAS, INNER, ORIGIN, SIDE, T, BMOD, NT
from .features import tile_frame, tonal_centroid

_K = np.array([[0.0, 1.402], [-0.344136, -0.714136], [1.772, 0.0]])   # (Cb, Cr) -> dR, dG, dB
_W = np.array([0.299, 0.587, 0.114])
_CANDS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)

def luma_exact_round(rgb, Y):
    lo = np.floor(rgb)
    best = besterr = None
    for c in _CANDS:
        cand = np.clip(lo + c, 0, 255)
        err = np.abs(cand @ _W - Y) + 1e-3*((cand - rgb)**2).sum(-1)
        if best is None:
            best, besterr = cand, err
        else:
            m = err < besterr
            best[m] = cand[m]; besterr[m] = err[m]
    return best.astype(np.uint8)

def colorize(Y, cb, cr, head=16.0, nb=6, exact=True):
    """head: RGB headroom kept free of clipping. nb: neighbourhood (px) over which chroma
    must be safe, because JPEG averages chroma over 2x2 px and interpolates on decode."""
    d = np.stack([_K[i, 0]*cb + _K[i, 1]*cr for i in range(3)], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(d > 1e-9, np.maximum(0, 255.0 - head - Y)[..., None]/d, np.inf)
        dn = np.where(d < -1e-9, np.maximum(0, Y - head)[..., None]/(-d), np.inf)
    s = np.clip(np.minimum(up, dn).min(-1), 0, 1)
    if nb > 1:
        s = np.minimum(s, uniform_filter(minimum_filter(s, size=nb), size=nb))
    rgb = Y[..., None] + s[..., None]*d
    return luma_exact_round(rgb, Y) if exact else np.clip(np.round(rgb), 0, 255).astype(np.uint8)

def colorize_naive(Y, cb, cr):
    """The old, broken method (kept only for the regression test)."""
    d = np.stack([_K[i, 0]*cb + _K[i, 1]*cr for i in range(3)], -1)
    return np.clip(Y[..., None] + d, 0, 255).astype(np.uint8)

def hue_cbcr(h, r):
    """h in turns (0 red, 1/6 yellow, 1/3 green, 1/2 cyan, 2/3 blue, 5/6 magenta)."""
    a = np.deg2rad(108.7) + 2*np.pi*np.asarray(h)
    return r*np.cos(a), r*np.sin(a)

def tiles_to_canvas(v, cells, sigma=6.0, fill=None):
    g = np.empty(NT); g[cells] = v
    px = np.kron(g.reshape(SIDE, SIDE), np.ones((T*BMOD, T*BMOD)))
    px = gaussian_filter(px, sigma) if sigma else px
    out = np.full((CANVAS, CANVAS), np.mean(v) if fill is None else fill)
    out[ORIGIN:ORIGIN+INNER, ORIGIN:ORIGIN+INNER] = px
    return gaussian_filter(out, 3) if sigma else out

def lerp_hue(a, b, t):
    d = ((b - a + 0.5) % 1.0) - 0.5
    return a + d*t

def oklch_cbcr(L, C, h_turn):
    """Chroma offsets (Cb, Cr) of the OKLCH colour (L, C, hue); hue in turns, 0 = red.
    OKLCH is perceptually even, so equal C looks about equally vivid at every hue."""
    h = 2*np.pi*(np.asarray(h_turn) + 29/360)
    a, b = C*np.cos(h), C*np.sin(h)
    l = (L + 0.3963377774*a + 0.2158037573*b)**3
    m = (L - 0.1055613458*a - 0.0638541728*b)**3
    s = (L - 0.0894841775*a - 1.2914855480*b)**3
    lin = [4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
           -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
           -0.0041960863*l - 0.7034186147*m + 1.7076147010*s]
    g = [255*np.sign(v)*np.where(np.abs(v) <= 0.0031308, 12.92*np.abs(v),
                                 1.055*np.abs(v)**(1/2.4) - 0.055) for v in lin]
    return -0.168736*g[0] - 0.331264*g[1] + 0.5*g[2], 0.5*g[0] - 0.418688*g[1] - 0.081312*g[2]

def best_luma(cb, cr, head=16.0):
    """Luma at which the chroma direction (cb, cr) has the most room inside the RGB cube
    (yellow: high, blue and red: low)."""
    d = np.array([_K[i, 0]*cb + _K[i, 1]*cr for i in range(3)])/np.hypot(cb, cr)
    ys = np.arange(40, 216)
    room = [min(((255 - head - y)/v if v > 0 else (y - head)/-v) for v in d if abs(v) > 1e-9) for y in ys]
    return int(ys[int(np.argmax(room))])


# ---------------------------------------------------------------- palette and the Droplets look
def palette(ft):
    rel = (ft["tonic"] + (3 if ft["mode"] == "minor" else 0)) % 12
    h0 = ((7*rel) % 12)/12 + 0.04
    sgn = 1 if ft["mode"] == "major" else -1
    spread = 0.10 + 0.12*np.clip((ft["brightness"] - 400)/2600, 0, 1)
    sat = float(np.clip(0.45 + 0.55*ft["tonality"], 0.45, 1.0))
    return dict(h0=h0 % 1, h1=(h0 + sgn*spread) % 1, sgn=sgn, sat=sat, rel=rel)


def per_tile(ft, pal, j=None):
    """j: STFT frame of every tile (default: tile t starts near sample 63*t)."""
    j = tile_frame(np.arange(NT), ft["hop"]) if j is None else j
    j = np.clip(j, 0, ft["S"].shape[1] - 1)
    L = 20*np.log10(ft["rms"] + 1e-9); Ln = np.clip((L - (L.max() - 42))/42, 0, 1)
    pc = ft["chroma"].argmax(0)
    d = (7*(pc - pal["rel"])) % 12; dsig = ((d + 6) % 12) - 6
    return dict(loud=Ln[j], dist=dsig[j], onset=ft["flux"][j])


def look_droplets(Y, ft, pal, cells, tmap, span=0.45, vivid=1.0, loud_shift=0.06, blur=5.0, tsig=10, j=None):
    """Colour that follows the time path continuously, so it has no 16 px steps.
    tmap       tile-time of every canvas pixel (from a2i.shapes)
    hue        key hue + span * harmony (tonal centroid on the circle of fifths) + a small
               loudness shift. span 0.45 keeps related colours around the key hue.
    vividness  how clearly tonal the moment is, times loudness. Perceptual (OKLCH) wheel."""
    j = tile_frame(np.arange(NT), ft["hop"]) if j is None else j
    j = np.clip(j, 0, ft["S"].shape[1] - 1)
    t = per_tile(ft, pal, j)
    turn, mag = tonal_centroid(ft, pal["rel"], tsig)
    loud = gaussian_filter1d(t["loud"], 6, mode="nearest")
    hue = pal["h0"] + span*turn[j] + pal["sgn"]*loud_shift*(loud - 0.5)
    v = vivid*(0.45 + 0.55*np.clip(mag[j]/0.64, 0, 1))*(0.4 + 0.6*loud)
    cb_t, cr_t = oklch_cbcr(0.62, 0.03 + 0.17*v, hue)
    return (gaussian_filter(np.interp(tmap, np.arange(NT), cb_t), blur),
            gaussian_filter(np.interp(tmap, np.arange(NT), cr_t), blur))

def brightness_for(pal):
    """Encoder policy (not a decoder constant): the median tile is drawn at the luma where the
    key hue has the most room, limited to 128..165. When it is raised, loud moments are drawn
    dark (negative A_ENV) so that they move towards mid grey instead of clipping."""
    centre = float(np.clip(best_luma(*oklch_cbcr(0.62, 0.15, pal["h0"])), 128, 165))
    return centre, (-1.0 if centre > 128 else 1.0)
