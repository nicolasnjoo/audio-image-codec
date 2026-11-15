"""
a2i codec: one function per pipeline stage.
Encoder  prepare -> analyze -> whiten -> pack -> normalize -> modulate -> rasterize
         (colorize lives in looks.py, frame is not built yet)
Decoder  luma -> demodulate -> denormalize -> unpack -> unwhiten -> synthesize
         (register is not built yet: input is assumed aligned to the canvas)
Side information (gain table, reference level, layout, amplitudes) is returned by the
encoder and must be passed to the decoder until `frame` puts it in a header.
"""
from math import gcd
import numpy as np
from scipy.fft import dctn, idctn
from scipy.signal import resample_poly
from .mdct import mdct, imdct

FS, M, F = 16000, 64, 4032
NS       = F * M                    # 258,048 samples = 16.128 s
CANVAS, INNER, BMOD = 1080, 1024, 2
G        = INNER // BMOD            # 512 modules per side
T        = 8                        # modules per tile side
SIDE     = G // T                   # 64 tiles per side
NT, AC   = SIDE * SIDE, T * T - 1   # 4096 tiles, 63 analog slots each
DR       = 9.0                      # envelope range, log2 units (~54 dB)
NB, NBLK = 16, 32                   # whitening grid: bands x time blocks
A_ENV, A_RES = 30.0, 105.0          # function defaults; RECOMMENDED SETTING IS 60 / 75
ORIGIN   = (CANVAS - INNER) // 2

def _zigzag(n=T):
    return sorted(((i, j) for i in range(n) for j in range(n)),
                  key=lambda p: (p[0]+p[1], p[1] if (p[0]+p[1]) % 2 == 0 else -p[1]))
ZZ = _zigzag()
_ZR = np.array([r for r, _ in ZZ[1:]]); _ZC = np.array([c for _, c in ZZ[1:]])

def layout(name="raster"):
    """cells[t] = row*SIDE + col for tile t (a permutation of 0..NT-1)."""
    if name == "raster":
        return np.arange(NT)
    if name == "serpentine":
        g = np.arange(NT).reshape(SIDE, SIDE); g[1::2] = g[1::2, ::-1]
        return g.reshape(-1)
    if name == "hilbert":
        cells = np.empty(NT, int)
        for d in range(NT):
            x = y = 0; t = d; s = 1
            while s < SIDE:
                rx = 1 & (t // 2); ry = 1 & (t ^ rx)
                if ry == 0:
                    if rx == 1: x, y = s-1-x, s-1-y
                    x, y = y, x
                x += s*rx; y += s*ry; t //= 4; s *= 2
            cells[d] = y*SIDE + x
        return cells
    if name in ("spiral_in", "spiral_out"):
        r, c = SIDE//2 - 1, SIDE//2 - 1
        seen, order = set(), []
        step, dirs, k = 1, [(0, 1), (1, 0), (0, -1), (-1, 0)], 0
        order.append((r, c)); seen.add((r, c))
        while len(order) < NT:
            for _ in range(2):
                dr, dc = dirs[k % 4]
                for _ in range(step):
                    r, c = r+dr, c+dc
                    if 0 <= r < SIDE and 0 <= c < SIDE and (r, c) not in seen:
                        seen.add((r, c)); order.append((r, c))
                k += 1
            step += 1
        cells = np.array([a*SIDE + b for a, b in order])
        return cells[::-1] if name == "spiral_in" else cells
    raise ValueError(name)

def prepare(x, fs_in=FS):
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if fs_in != FS:
        g = gcd(int(fs_in), FS); x = resample_poly(x, FS//g, int(fs_in)//g)
    x = np.pad(x[:NS], (0, max(0, NS - len(x))))
    pk = np.abs(x).max()
    return x / (pk * 1.02) if pk > 0 else x

def analyze(x):
    return mdct(x, M)[:F]

def _expand(gains):
    return np.repeat(np.repeat(gains, F//NBLK, 0), M//NB, 1)

def whiten(X):
    gains = np.sqrt((X.reshape(NBLK, F//NBLK, NB, M//NB)**2).mean(axis=(1, 3))) + 1e-12
    return X / _expand(gains), gains

def pack(W):
    return W.reshape(-1)[:NT*AC].reshape(NT, AC)

def normalize(C):
    lp = np.log2(np.abs(C).max(1) + 1e-12)
    ref = lp.max()
    lp = np.clip(lp, ref - DR, ref)
    env = 2*(lp - (ref - DR))/DR - 1
    R = np.clip(C / (2.0**lp)[:, None], -1, 1)
    return env, R, float(ref)

def _grain(env, alpha):
    return 1.0 if alpha == 0 else (0.25 + 0.75*((env+1)/2))**alpha

def modulate(env, R, cells=None, a_env=A_ENV, a_res=A_RES, grain=0.0):
    cells = layout() if cells is None else cells
    g = _grain(env, grain)
    V = np.zeros((NT, T, T))
    V[:, 0, 0] = a_env * env * T                      # ortho DCT: DC = mean * T
    V[:, _ZR, _ZC] = a_res * (R * (g[:, None] if np.ndim(g) else g))
    mods = idctn(V, axes=(1, 2), norm="ortho")
    placed = np.empty_like(mods); placed[cells] = mods
    return placed.reshape(SIDE, SIDE, T, T).transpose(0, 2, 1, 3).reshape(G, G)

def rasterize(P):
    Y = np.full((CANVAS, CANVAS), 128.0)
    Y[ORIGIN:ORIGIN+INNER, ORIGIN:ORIGIN+INNER] = 128 + np.kron(P, np.ones((BMOD, BMOD)))
    clip = float(((Y < 0) | (Y > 255)).mean() * (CANVAS/INNER)**2)
    return np.clip(Y, 0, 255), clip

def encode_luma(x, fs_in=FS, cells=None, a_env=A_ENV, a_res=A_RES, grain=0.0):
    x = prepare(x, fs_in)
    W, gains = whiten(analyze(x))
    env, R, ref = normalize(pack(W))
    Y, clip = rasterize(modulate(env, R, cells, a_env, a_res, grain))
    side = dict(gains=gains, ref=ref, cells=cells, a_env=a_env, a_res=a_res, grain=grain)
    return Y, side, dict(x=x, env=env, clip=clip)

def luma(rgb):
    a = np.asarray(rgb, float)
    return a if a.ndim == 2 else a[..., 0]*0.299 + a[..., 1]*0.587 + a[..., 2]*0.114

def demodulate(Y, cells=None, a_env=A_ENV, a_res=A_RES, grain=0.0):
    cells = layout() if cells is None else cells
    inner = Y[ORIGIN:ORIGIN+INNER, ORIGIN:ORIGIN+INNER] - 128.0
    P = inner.reshape(G, BMOD, G, BMOD).mean(axis=(1, 3))
    grid = P.reshape(SIDE, T, SIDE, T).transpose(0, 2, 1, 3).reshape(NT, T, T)
    V = dctn(grid[cells], axes=(1, 2), norm="ortho")
    env = np.clip(V[:, 0, 0] / (a_env*T), -1, 1)
    g = _grain(env, grain)
    R = np.clip(V[:, _ZR, _ZC] / a_res / (g[:, None] if np.ndim(g) else g), -1, 1)
    return env, R

def denormalize(env, R, ref):
    lp = (env + 1)/2*DR + (ref - DR)
    return R * (2.0**lp)[:, None]

def unpack(C):
    W = np.zeros(F*M); W[:NT*AC] = C.reshape(-1)
    return W.reshape(F, M)

def unwhiten(W, gains):
    return W * _expand(gains)

def synthesize(X):
    return imdct(X, M, NS)

def decode(img, side):
    env, R = demodulate(luma(img), side["cells"], side["a_env"], side["a_res"], side["grain"])
    return synthesize(unwhiten(unpack(denormalize(env, R, side["ref"])), side["gains"]))
