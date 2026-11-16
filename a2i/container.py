"""Self-contained image (PROVISIONAL, version 1): everything the decoder needs is in the picture.

  frame      four QR-style finder squares in the corners of the 28 px border, and a ring of
             4 x 4 px black/white cells (4032 of them) around the data
  header     version, sample rate, polarity, the 12 note weights of the layout, CRC-8:
             64 bits, 6 copies spread around the ring
  gains      the 512 whitening gains, quantised to 3 dB steps (5 bits each) relative to the
             largest, plus CRC-16: 2576 bits in the remaining cells (1.4 copies). The encoder
             whitens with the same quantised gains, so the decoder's gains are exact
  register   finds the four finders in any upright picture of the image (a screenshot, a
             re-saved file), fits offset and scale per axis, resamples to 1080 x 1080

The absolute level is not carried: decode_image returns audio normalised to its peak, and
quality is therefore measured with snr_gain (best-fit level), not channel.snr.

This is a first version built so that screenshots can be tested end to end. It has not been
agreed as the format; the earlier codec functions and their out-of-band side dict still work.
"""
import numpy as np
from scipy.fft import dctn
from . import codec, shapes
from .codec import (CANVAS, INNER, ORIGIN, BMOD, G, T, SIDE, NT, _ZR, _ZC, prepare, analyze, whiten,
                    pack, unpack, normalize, modulate, rasterize, luma, synthesize, envelope_field, _expand)
from .features import features
from .looks import colorize, palette, look_droplets, brightness_for

VERSION = 1
RATES = (16000, 24000, 25600, 32000)        # header field: 2 bits
RATE = 32000                                # adopted: 8.064 s per image
A_ENV, A_RES, DR, SMOOTH = 60.0, 75.0, 6.0, 1.0
MOD, CELL = 4, 4                            # finder module and ring cell, px
HBITS, HCOPIES = 64, 6                      # header bits and copies
GBITS, GSTEP = 5, 0.5                       # gain: 5 bits, steps of 0.5 log2 units (3.01 dB)
FINDER = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1],
                   [1, 0, 1, 1, 1, 0, 1], [1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1]])     # 1 = black
FC = (ORIGIN/2, CANVAS - ORIGIN/2)          # finder centres: 14 and 1066 (pixel-edge coordinates)
THRESH = 128.0


# ---------------------------------------------------------------- bits
def _crc(bits, width, poly):
    c, top, mask = 0, 1 << (width - 1), (1 << width) - 1
    for b in bits:
        c ^= int(b) << (width - 1)
        c = ((c << 1) ^ poly) & mask if c & top else (c << 1) & mask
    return [(c >> k) & 1 for k in range(width - 1, -1, -1)]

def crc8(bits):
    return _crc(bits, 8, 0x07)

def crc16(bits):
    return _crc(bits, 16, 0x1021)

def _field(v, n):
    return [(int(v) >> k) & 1 for k in range(n - 1, -1, -1)]

def _value(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v

def header_bits(rate, sign, w):
    body = _field(VERSION, 4) + _field(RATES.index(int(rate)), 2) + [1 if sign < 0 else 0, 0]
    for v in w:
        body += _field(v, 4)
    return body + crc8(body)

def parse_header(bits):
    """Returns dict(rate, sign, w) or None when the CRC or the version does not match."""
    if crc8(bits[:56]) != [int(b) for b in bits[56:]] or _value(bits[:4]) != VERSION:
        return None
    return dict(rate=RATES[_value(bits[4:6])], sign=-1.0 if bits[6] else 1.0,
                w=[_value(bits[8 + 4*k:12 + 4*k]) for k in range(12)])

def quantise_gains(gains):
    """512 gains -> integer steps below the largest (0..31, 3 dB each) and the gains they stand
    for. The encoder whitens with these, so encoder and decoder use identical gains."""
    g = np.log2(gains)
    q = np.clip(np.round((g.max() - g)/GSTEP), 0, 2**GBITS - 1).astype(int)
    return q, 2.0**(g.max() - GSTEP*q)

def gain_bits(q):
    body = [b for v in q.ravel() for b in _field(v, GBITS)]
    return body + crc16(body)

def parse_gains(bits):
    """Returns (gains relative to the largest, crc_ok). The gains are used even if the CRC fails:
    a wrong cell then damages one band in one quarter second, not the whole clip."""
    n = codec.NBLK*codec.NB*GBITS
    q = np.array([_value(bits[k:k + GBITS]) for k in range(0, n, GBITS)]).reshape(codec.NBLK, codec.NB)
    return 2.0**(-GSTEP*q), crc16(bits[:n]) == [int(b) for b in bits[n:n + 16]]

def cell_origins():
    """(y, x) of the 4032 ring cells: four rows of 252 cells on each side, clockwise from the top,
    outer row first. 8 px next to each finder stay white."""
    span = range(ORIGIN + 8, CANVAS - ORIGIN - 8, CELL)
    out = []
    for k in range(4):
        out += [(4 + CELL*k, x) for x in span]
    for k in range(4):
        out += [(y, CANVAS - 8 - CELL*k) for y in span]
    for k in range(4):
        out += [(CANVAS - 8 - CELL*k, x) for x in span]
    for k in range(4):
        out += [(y, 4 + CELL*k) for y in span]
    return out

def cell_mask(n, seed=1):
    """Fixed pseudo-random bits (the same in every implementation: a 32-bit LCG)."""
    s, out = seed, []
    for _ in range(n):
        s = (s*1103515245 + 12345) & 0xFFFFFFFF
        out.append((s >> 16) & 1)
    return out

def cell_plan(n):
    """For each ring cell: ('h', header bit) or ('g', gain-message bit). The 384 header cells are
    spread evenly around the ring; the others carry the gain message in order, wrapping round."""
    nh = HBITS*HCOPIES
    hpos = {(k*n)//nh: k % HBITS for k in range(nh)}
    glen = codec.NBLK*codec.NB*GBITS + 16
    plan, g = [], 0
    for i in range(n):
        if i in hpos:
            plan.append(("h", hpos[i]))
        else:
            plan.append(("g", g % glen)); g += 1
    return plan


# ---------------------------------------------------------------- frame
def frame(rgb, hbits, gbits):
    """Draw the finders and the ring over the outer 20 px of the border (the inner 8 px of the
    border keep the picture's colour at luma 128 and separate the ring from the data)."""
    out = np.array(rgb, np.uint8)
    out[:20], out[-20:], out[:, :20], out[:, -20:] = 255, 255, 255, 255
    cells = cell_origins(); mask = cell_mask(len(cells)); plan = cell_plan(len(cells))
    for (y, x), m, (kind, k) in zip(cells, mask, plan):
        if (hbits[k] if kind == "h" else gbits[k]) ^ m:
            out[y:y + CELL, x:x + CELL] = 0
    f = (255*(1 - np.kron(FINDER, np.ones((MOD, MOD), int))))[..., None].astype(np.uint8)
    n = 7*MOD
    for y in (0, CANVAS - n):
        for x in (0, CANVAS - n):
            out[y:y + n, x:x + n] = f
    return out

def read_ring(L):
    """L: canonical 1080 x 1080 luma -> (header bits, gain-message bits), each by a soft vote
    over the copies of that bit (the middle 2 x 2 px of every cell is read)."""
    cells = cell_origins(); mask = cell_mask(len(cells)); plan = cell_plan(len(cells))
    soft = {"h": np.zeros(HBITS), "g": np.zeros(codec.NBLK*codec.NB*GBITS + 16)}
    for (y, x), m, (kind, k) in zip(cells, mask, plan):
        v = L[y + 1:y + 3, x + 1:x + 3].mean() - THRESH            # white > 0
        soft[kind][k] += -v if m else v
    return (soft["h"] < 0).astype(int).tolist(), (soft["g"] < 0).astype(int).tolist()     # black = 1


# ---------------------------------------------------------------- encoder
def encode_image(x, fs_in, rate=RATE):
    """Audio -> framed 1080 x 1080 RGB image that decode_image can read with no other input."""
    xp = prepare(x, fs_in, rate)
    X = analyze(xp)
    q, gains = quantise_gains(whiten(X)[1])
    env, R, _ = normalize(pack(X/_expand(gains)), DR)
    ft = features(xp, rate); pal = palette(ft)
    w = shapes.weights(ft)
    cells, under, tmap = shapes.build("droplets", w)
    centre, sign = brightness_for(pal)
    a_env = sign*A_ENV
    shift = float(np.round(centre - 128.0 - a_env*np.median(env)))
    Y, clip = rasterize(modulate(env, R, cells, a_env, A_RES, shift=shift, smooth=SMOOTH), under)
    cb, cr = look_droplets(Y, ft, pal, cells, tmap)
    rgb = frame(colorize(Y, cb, cr), header_bits(rate, sign, w), gain_bits(q))
    return rgb, dict(x=xp, rate=rate, w=w, sign=sign, clip=clip, key=ft["key"], gains=gains)


# ---------------------------------------------------------------- register: find the image in a picture
def _row_candidates(B, m_lo, m_hi):
    """Rows of B (True = dark) scanned for dark-light-DARK-light-dark runs in ratio 1:1:3:1:1."""
    H, W = B.shape
    out = []
    for y in range(H):
        row = B[y]
        starts = np.concatenate([[0], np.flatnonzero(row[1:] != row[:-1]) + 1])
        if len(starts) < 5:
            continue
        lens = np.diff(np.concatenate([starts, [W]]))
        i = np.flatnonzero(row[starts[:-4]])
        b1, w1, b2, w2, b3 = (lens[i + k] for k in range(5))
        m = (w1 + b2 + w2)/5.0
        ok = ((m >= m_lo) & (m <= m_hi) & (np.abs(w1 - m) <= 0.5*m + 0.5) & (np.abs(w2 - m) <= 0.5*m + 0.5)
              & (np.abs(b2 - 3*m) <= m + 0.5) & (b1 >= 0.5*m) & (b3 >= 0.5*m))
        for k in np.flatnonzero(ok):
            out.append((y, starts[i[k] + 2] + lens[i[k] + 2]/2.0, m[k]))
    return out

def _column_check(B, y, x, m):
    """The same 1:3:1 test down the column through a row candidate; returns the vertical centre."""
    H = B.shape[0]
    if not B[y, x]:
        return None
    y0 = y
    while y0 > 0 and B[y0 - 1, x]:
        y0 -= 1
    y1 = y
    while y1 < H - 1 and B[y1 + 1, x]:
        y1 += 1
    b2 = y1 - y0 + 1
    a, wu = y0 - 1, 0
    while a >= 0 and not B[a, x]:
        wu += 1; a -= 1
    c, wd = y1 + 1, 0
    while c < H and not B[c, x]:
        wd += 1; c += 1
    if a < 0 or c >= H:
        return None                                                # no dark ring above or below
    mv = (wu + b2 + wd)/5.0
    if (abs(wu - mv) <= 0.5*mv + 0.5 and abs(wd - mv) <= 0.5*mv + 0.5 and abs(b2 - 3*mv) <= mv + 0.5
            and abs(mv - m) <= 0.35*m + 0.5):
        return y0 + b2/2.0, mv
    return None

def _refine(L, x0, y0, m):
    """Sub-pixel centre of a finder's dark 3x3-module core from its four edges (edge coordinates)."""
    def centre(line, c0):
        c = int(round(c0)); n = len(line)
        if not (0 < c < n - 1) or line[c] >= THRESH:
            return None
        lo = c
        while lo > 0 and line[lo - 1] < THRESH:
            lo -= 1
        hi = c
        while hi < n - 1 and line[hi + 1] < THRESH:
            hi += 1
        if lo == 0 or hi == n - 1:
            return None
        left = (lo - 1) + (line[lo - 1] - THRESH)/(line[lo - 1] - line[lo])      # crossing, pixel centres
        right = hi + (line[hi] - THRESH)/(line[hi] - line[hi + 1])
        return (left + right)/2 + 0.5
    r = max(1, int(round(1.0*m)))
    xs = [centre(L[y], x0) for y in range(int(round(y0)) - r, int(round(y0)) + r + 1) if 0 <= y < L.shape[0]]
    ys = [centre(L[:, x], y0) for x in range(int(round(x0)) - r, int(round(x0)) + r + 1) if 0 <= x < L.shape[1]]
    xs = [v for v in xs if v is not None]; ys = [v for v in ys if v is not None]
    return (float(np.median(xs)) if xs else x0 + 0.5), (float(np.median(ys)) if ys else y0 + 0.5)

def find_finders(L, scale_range=(0.7, 2.2)):
    """Four finder centres (x, y) in edge coordinates, ordered TL, TR, BL, BR, or None."""
    B = L < THRESH
    cands = []
    for y, xc, m in _row_candidates(B, MOD*scale_range[0], MOD*scale_range[1]):
        v = _column_check(B, y, int(xc), m)
        if v is not None:
            cands.append((xc, v[0], (m + v[1])/2))
    clusters = []                                                   # [sum x, sum y, sum m, votes]
    for x, y, m in cands:
        for c in clusters:
            if abs(c[0]/c[3] - x) <= 1.5*m and abs(c[1]/c[3] - y) <= 1.5*m:
                c[0] += x; c[1] += y; c[2] += m; c[3] += 1
                break
        else:
            clusters.append([x, y, m, 1])
    P = [(c[0]/c[3], c[1]/c[3], c[2]/c[3], c[3]) for c in clusters if c[3] >= 3]
    best = None
    span = (FC[1] - FC[0])/MOD                                      # 263 modules between finder centres
    for a in P:
        for b in P:
            side = b[0] - a[0]
            if side <= 0 or abs(b[1] - a[1]) > 0.02*side + 3:
                continue
            m = side/span
            if abs(a[2] - m) > 0.3*m or abs(b[2] - m) > 0.3*m:
                continue
            tol = 0.02*side + 3
            c = [p for p in P if abs(p[0] - a[0]) <= tol and abs(p[1] - (a[1] + side)) <= tol]
            d = [p for p in P if abs(p[0] - b[0]) <= tol and abs(p[1] - (b[1] + side)) <= tol]
            if c and d:
                c, d = max(c, key=lambda p: p[3]), max(d, key=lambda p: p[3])
                score = a[3] + b[3] + c[3] + d[3]
                if best is None or score > best[0]:
                    best = (score, [a, b, c, d])
    if best is None:
        return None
    return [_refine(L, p[0] - 0.5, p[1] - 0.5, p[2]) for p in best[1]]

def _lanczos(n_out, n_in, a, b, lobes=3):
    """Weights (n_out, n_in): output pixel i (centre i + 0.5) reads the source around a*(i + 0.5) + b."""
    s = max(1.0, a)
    W = np.zeros((n_out, n_in))
    for i in range(n_out):
        xs = a*(i + 0.5) + b
        j = np.arange(int(np.floor(xs - lobes*s)), int(np.ceil(xs + lobes*s)) + 1)
        t = (j + 0.5 - xs)/s
        w = np.where(np.abs(t) < lobes, np.sinc(t)*np.sinc(t/lobes), 0.0)
        np.add.at(W[i], np.clip(j, 0, n_in - 1), w/w.sum())
    return W

def register(img):
    """Any upright picture containing the image -> canonical 1080 x 1080 luma, plus the fit
    x_picture = ax*u + bx, y_picture = ay*v + by (u, v: canonical edge coordinates).
    Returns (None, None) when the four finders are not found."""
    L = luma(img)
    f = find_finders(L)
    if f is None:
        return None, None
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = f
    d = FC[1] - FC[0]
    ax = ((x1 - x0) + (x3 - x2))/(2*d); ay = ((y2 - y0) + (y3 - y1))/(2*d)
    bx = np.mean([x0 - ax*FC[0], x1 - ax*FC[1], x2 - ax*FC[0], x3 - ax*FC[1]])
    by = np.mean([y0 - ay*FC[0], y1 - ay*FC[0], y2 - ay*FC[1], y3 - ay*FC[1]])
    Lc = _lanczos(CANVAS, L.shape[0], ay, by) @ L @ _lanczos(CANVAS, L.shape[1], ax, bx).T
    return Lc, dict(ax=float(ax), bx=float(bx), ay=float(ay), by=float(by))


# ---------------------------------------------------------------- decoder
def decode_canonical(Lc, gains=None):
    """Canonical luma -> (audio normalised to its peak, info) or (None, None) if the header fails.
    gains: for diagnostics only, the encoder's gains instead of the ones read from the ring."""
    hb, gb = read_ring(Lc)
    h = parse_header(hb)
    if h is None:
        return None, None
    g_ring, ok = parse_gains(gb)
    cells, under, _ = shapes.build("droplets", h["w"])
    sl = slice(ORIGIN, ORIGIN + INNER)
    P = (Lc - under)[sl, sl].reshape(G, BMOD, G, BMOD).mean(axis=(1, 3)) - 128.0
    E = P.reshape(SIDE, T, SIDE, T).mean(axis=(1, 3))
    P = P - envelope_field(E, SMOOTH)
    V = dctn(P.reshape(SIDE, T, SIDE, T).transpose(0, 2, 1, 3).reshape(NT, T, T)[cells], axes=(1, 2), norm="ortho")
    R = np.clip(V[:, _ZR, _ZC]/A_RES, -1, 1)
    e = E.reshape(-1)[cells]
    lp = (e - np.median(e))/(h["sign"]*A_ENV)*DR/2                 # level of each tile, up to a constant
    y = synthesize(unpack(R*(2.0**lp)[:, None])*_expand(g_ring if gains is None else gains))
    pk = np.abs(y).max()
    return (y/(pk*1.02) if pk > 0 else y), dict(h, gains_ok=bool(ok))

def decode_image(img):
    """Picture -> (audio, info). info is None when the image is not found or its header is unreadable."""
    Lc, fit = register(img)
    if Lc is None:
        return None, None
    y, h = decode_canonical(Lc)
    if y is None:
        return None, None
    return y, dict(h, **fit)

def snr_gain(x, y):
    """SNR after scaling y to fit x best (the image does not carry the absolute level)."""
    n = min(len(x), len(y)); x = x[:n]; y = y[:n]
    g = float(np.dot(x, y)/np.dot(y, y))
    return float(10*np.log10(np.sum(x**2)/np.sum((x - g*y)**2)))
