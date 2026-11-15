"""Chroma carries no data; colorize() guarantees luma is untouched."""
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, uniform_filter
from .codec import CANVAS, INNER, ORIGIN, SIDE, T, BMOD, NT

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
