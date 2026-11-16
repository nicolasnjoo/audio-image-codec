"""Self-contained image, end to end: encode, turn the image into a simulated phone screenshot
(scaled, at a fractional position, surrounded by an interface), decode from that picture alone.

usage:  python scripts/container_test.py [NAME=file.wav ...] [--at 60]
        (without files: the three synthetic pieces, upsampled to 32 kHz)

Reports the audio SNR (best-fit level, see container.snr_gain), the alignment error against
the true placement, and what a perfect alignment would have given.
"""
import argparse, io, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))    # also works under python -I
import numpy as np
from PIL import Image
from _common import OUT, save_json
from a2i import container as ct
from a2i.codec import CANVAS
from a2i.testsignals import PIECES

PAD = 16                                                     # background drawn around the image before scaling

def screenshot(rgb, W, H, scale, tx, ty, bg=0, how="lanczos", seed=0):
    """A W x H picture with the image drawn at `scale`, its top-left corner at (tx, ty) (may be
    fractional), on background `bg`, with a status bar, a toolbar and some high-contrast clutter."""
    rng = np.random.default_rng(seed)
    shot = np.full((H, W, 3), bg, float)
    shot[:90] = 245 if bg < 128 else 20                      # status bar
    shot[H - 180:] = 230 if bg < 128 else 35                 # toolbar
    for _ in range(40):                                      # icons / text-like blocks
        y, x = rng.integers(0, H - 40), rng.integers(0, W - 60)
        shot[y:y + rng.integers(6, 40), x:x + rng.integers(6, 60)] = rng.choice([0, 255])
    src = np.full((CANVAS + 2*PAD, CANVAS + 2*PAD, 3), bg, float)
    src[PAD:-PAD, PAD:-PAD] = rgb
    n = int(np.ceil(CANVAS*scale)) + 2
    x0, y0 = int(np.floor(tx)) - 1, int(np.floor(ty)) - 1    # integer origin of the patch
    if how == "lanczos":
        Wx = ct._lanczos(n, CANVAS + 2*PAD, 1/scale, (x0 - tx)/scale + PAD)
        Wy = ct._lanczos(n, CANVAS + 2*PAD, 1/scale, (y0 - ty)/scale + PAD)
        patch = np.stack([Wy @ src[..., c] @ Wx.T for c in range(3)], -1)
    else:                                                    # bilinear, as many viewers do
        def lin(n_out, off):
            u = (np.arange(n_out) + 0.5 + off)/scale + PAD - 0.5
            i = np.clip(np.floor(u).astype(int), 0, CANVAS + 2*PAD - 2); f = np.clip(u - i, 0, 1)
            M = np.zeros((n_out, CANVAS + 2*PAD)); M[np.arange(n_out), i] = 1 - f; M[np.arange(n_out), i + 1] = f
            return M
        Wx, Wy = lin(n, x0 - tx), lin(n, y0 - ty)
        patch = np.stack([Wy @ src[..., c] @ Wx.T for c in range(3)], -1)
    ys, xs = slice(max(0, y0), min(H, y0 + n)), slice(max(0, x0), min(W, x0 + n))
    shot[ys, xs] = patch[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    return np.clip(np.round(shot), 0, 255).astype(np.uint8)

def as_jpeg(a, q):
    b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=2); b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"))

CASES = [  # name, W, H, scale, tx, ty, background, viewer interpolation, file
    ("the image file itself (PNG)",              1080, 1080, 1.0,        0.0,    0.0,    0, "lanczos", None),
    ("the image file re-saved as JPEG q85",      1080, 1080, 1.0,        0.0,    0.0,    0, "lanczos", 85),
    ("phone 1080 wide, screenshot PNG",          1080, 2400, 1.0,        0.0,  660.0,    0, "lanczos", None),
    ("phone 1179 wide, screenshot PNG",          1179, 2556, 1179/1080,  0.0,  688.37,   0, "lanczos", None),
    ("phone 1179 wide, bilinear viewer, PNG",    1179, 2556, 1179/1080,  0.0,  688.37, 255, "bilinear", None),
    ("phone 1179 wide, screenshot sent as JPEG q75", 1179, 2556, 1179/1080, 0.0, 688.37, 0, "lanczos", 75),
    ("phone 1440 wide, screenshot PNG",          1440, 3120, 1440/1080,  0.0,  840.5,  255, "lanczos", None),
    ("image with margins, 95 % size, PNG",       1179, 2556, 0.95,      76.3,  700.7,   30, "lanczos", None),
    ("desktop window, 80 % size, PNG",           1920, 1080, 0.80,     528.4,  108.2,  255, "lanczos", None),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="NAME=path.wav")
    ap.add_argument("--at", default="60")
    args = ap.parse_args()
    clips = {}
    if args.files:
        from real_music import load
        for spec in args.files:
            name, path = spec.split("=", 1)
            fs, a = load(path)
            for t0 in (float(t) for t in args.at.split(",")):
                clips[f"{name}@{t0:g}"] = (a[int(t0*fs):int((t0 + 8.3)*fs)], fs)
    else:
        clips = {f"piece {pk}": (fn(), 16000) for pk, fn in PIECES.items()}
    enc = {cid: ct.encode_image(x, fs) for cid, (x, fs) in clips.items()}
    print(f"{len(enc)} clips. SNR in dB with best-fit level; alignment error in canonical pixels (RMS over the image).")
    print(f"{'case':<46} {'SNR':>6} {'min':>6}   {'perfect align':>13}   {'align err px':>12} {'max':>6}   {'scale err %':>11}  found")
    res = {}
    for name, W, H, scale, tx, ty, bg, how, q in CASES:
        snrs, ideal, errs, serr, found = [], [], [], [], 0
        for k, (cid, (rgb, info)) in enumerate(enc.items()):
            shot = rgb if (W, H) == (CANVAS, CANVAS) else screenshot(rgb, W, H, scale, tx, ty, bg, how, seed=k)
            shot = as_jpeg(shot, q) if q else shot
            y, h = ct.decode_image(shot)
            L = ct.luma(shot)
            y0, _ = ct.decode_canonical(ct._lanczos(CANVAS, H, scale, ty) @ L @ ct._lanczos(CANVAS, W, scale, tx).T)
            ideal.append(ct.snr_gain(info["x"], y0) if y0 is not None else np.nan)
            if y is None:
                continue
            found += 1; snrs.append(ct.snr_gain(info["x"], y))
            u = np.linspace(0, CANVAS, 9)
            ex = ((h["ax"] - scale)*u + (h["bx"] - tx))/scale; ey = ((h["ay"] - scale)*u + (h["by"] - ty))/scale
            errs.append(np.sqrt(np.mean(ex**2 + ey**2))); serr.append(100*max(abs(h["ax"]/scale - 1), abs(h["ay"]/scale - 1)))
        res[name] = dict(snr=snrs, perfect=ideal, align_px=errs, scale_err_pct=serr, found=found)
        if snrs:
            print(f"{name:<46} {np.mean(snrs):6.2f} {np.min(snrs):6.2f}   {np.nanmean(ideal):13.2f}   {np.mean(errs):12.3f} {np.max(errs):6.3f}"
                  f"   {np.max(serr):11.4f}  {found}/{len(enc)}")
        else:
            print(f"{name:<46} image not found in any clip (perfect alignment would give {np.nanmean(ideal):.2f} dB)")
    save_json("container_test.json", res)
