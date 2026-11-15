import io
import numpy as np
from PIL import Image
from scipy.ndimage import shift as ndshift

def jpeg(rgb, q, sub=2):
    im = Image.fromarray(np.clip(np.asarray(rgb), 0, 255).astype(np.uint8), "RGB")
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, subsampling=sub); b.seek(0)
    return np.asarray(Image.open(b).convert("RGB")).astype(float)

def transmit(rgb, q=None, screenshot=False, reg=0.0, scale=1179):
    """q=None -> lossless PNG. Returns luma (float) as the decoder would see it.
    NOTE: pass already-rounded uint8 images (np.round(Y) for grey)."""
    y = np.asarray(rgb, float)
    if y.ndim == 2:
        y = np.repeat(y[:, :, None], 3, 2)
    if q is not None:
        y = jpeg(y, q)
    if screenshot:
        n = y.shape[0]
        im = Image.fromarray(np.clip(y, 0, 255).astype(np.uint8), "RGB")
        im = im.resize((scale, scale), Image.LANCZOS).resize((n, n), Image.LANCZOS)
        y = jpeg(np.asarray(im).astype(float), q or 90)
    L = y[..., 0]*0.299 + y[..., 1]*0.587 + y[..., 2]*0.114
    return ndshift(L, (reg, reg), order=3, mode="nearest") if reg else L

CONDITIONS = [
    ("lossless PNG",            None, False, 0.0),
    ("JPEG q85 file",           85,   False, 0.0),
    ("screenshot, 0.15 px reg", 75,   True,  0.15),
    ("screenshot, 0.35 px reg", 75,   True,  0.35),
    ("q60 screenshot, 0.5 px",  60,   True,  0.50),
]

def snr(a, b):
    n = min(len(a), len(b)); return 10*np.log10(np.var(a[:n]) / np.var(a[:n] - b[:n]))

def segsnr(a, b, w=1024):
    n = min(len(a), len(b))//w*w; A = a[:n].reshape(-1, w); B = b[:n].reshape(-1, w)
    p = (A**2).mean(1); e = ((A-B)**2).mean(1) + 1e-12; m = p > p.max()*1e-4
    return float(np.clip(10*np.log10(p[m]/e[m]), -10, 60).mean())
