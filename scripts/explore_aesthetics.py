"""EXPERIMENTAL - aesthetic levers that are not part of the format.

Nothing here changes a2i/codec.py. Each lever is an optional parameter of encode_x/decode_x,
so its audio cost can be measured against the current format before anything is adopted:

  centre   luma value given to the median tile (None = current mapping, 128 + A_ENV*env)
  dr       envelope range in log2 units (current 9)
  smooth   envelope drawn as a smooth, tile-mean-preserving field instead of 16 px blocks
           (0 = blocks; 1 = linear blend across the whole tile); the decoder rebuilds the
           same field from the decoded tile means and subtracts it before reading the AC slots
  nblk     whitening time blocks (current 32)
  layout   any codec layout, or "disc_in": a round spiral from the rim to the centre

Colour (free: the decoder never reads it) adds a perceptual hue wheel, colour drawn in true
polar coordinates (round rings, no 16 px steps) and a hue that follows the harmony.

Run:  python scripts/explore_aesthetics.py     (lever table, step ladder, sheets in out/explore_*.png)
"""
import contextlib
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from _common import OUT, save_json
from a2i import codec
from a2i.codec import NT, T, SIDE, G, BMOD, CANVAS, INNER, ORIGIN, _ZR, _ZC
from a2i.looks import colorize, hue_cbcr, tiles_to_canvas, lerp_hue
from a2i.features import features, tile_frame
from a2i.channel import transmit, snr
from a2i.testsignals import PIECES
from render_options import palette, per_tile, render, look_record

KEY = [("lossless", None, False, 0.0), ("screenshot 0.15 px", 75, True, 0.15), ("worst", 60, True, 0.5)]
SL = slice(ORIGIN, ORIGIN + INNER)


# ---------------------------------------------------------------- layout
def disc_layout(inward=True):
    """Round (Archimedean) spiral, one tile per turn. cells[t] = grid cell of tile t.
    NOTE: uses float atan2/hypot; a shipped format would need an integer table instead."""
    r, c = np.divmod(np.arange(NT), SIDE)
    x = c - (SIDE - 1)/2; y = r - (SIDE - 1)/2
    th = (np.arctan2(x, -y) % (2*np.pi))/(2*np.pi)          # 0 at 12 o'clock, clockwise
    key = np.floor(np.hypot(x, y) - th) + th                # turn number + angle
    order = np.lexsort((np.arange(NT), key))                # centre -> rim
    return order[::-1].copy() if inward else order

def get_layout(name):
    """A codec layout name, "disc_in", or an explicit permutation (see scripts/shapes.py)."""
    if isinstance(name, np.ndarray):
        return name
    return disc_layout(True) if name == "disc_in" else codec.layout(name)


# ---------------------------------------------------------------- smooth envelope
def _interp_matrix(w):
    """U (G x SIDE): module <- tile-centre values; neighbours are blended linearly over the
    fraction w of a tile centred on the tile boundary (w=1: plain linear interpolation)."""
    U = np.zeros((G, SIDE))
    p = (np.arange(G) + 0.5)/T - 0.5                        # module position in tile-centre units
    t0 = np.clip(np.floor(p).astype(int), 0, SIDE - 2)
    u = np.clip(p - t0, 0, 1)
    b = np.clip((u - (1 - w)/2)/w, 0, 1)
    U[np.arange(G), t0] = 1 - b; U[np.arange(G), t0 + 1] += b
    return U

_FIELD = {}
def envelope_field(E, w):
    """E: (SIDE, SIDE) tile means in luma units -> (G, G) module plane with exactly those tile means."""
    if not w:
        return np.kron(E, np.ones((T, T)))
    if w not in _FIELD:
        U = _interp_matrix(w)
        K = U.reshape(SIDE, T, SIDE).mean(1)                # tile mean of the interpolant
        _FIELD[w] = (U, np.linalg.inv(K))
    U, Ki = _FIELD[w]
    return U @ (Ki @ E @ Ki.T) @ U.T


# ---------------------------------------------------------------- experimental codec
@contextlib.contextmanager
def _patched(**kw):
    old = {k: getattr(codec, k) for k in kw}
    for k, v in kw.items():
        setattr(codec, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(codec, k, v)

def encode_x(x, layout="spiral_in", a_env=60.0, a_res=75.0, centre=None, dr=9.0, smooth=0.0,
             nblk=32, under=None):
    x = codec.prepare(x)
    cells = get_layout(layout)
    with _patched(DR=dr, NBLK=nblk):
        W, gains = codec.whiten(codec.analyze(x))
        env, R, ref = codec.normalize(codec.pack(W))
    shift = 0.0 if centre is None else centre - 128.0 - a_env*float(np.median(env))
    V = np.zeros((NT, T, T)); V[:, _ZR, _ZC] = a_res*R
    tex = idctn(V, axes=(1, 2), norm="ortho")
    placed = np.empty_like(tex); placed[cells] = tex
    P = placed.reshape(SIDE, SIDE, T, T).transpose(0, 2, 1, 3).reshape(G, G)
    E = np.empty(NT); E[cells] = a_env*env + shift
    P = P + envelope_field(E.reshape(SIDE, SIDE), smooth)
    Yun = np.full((CANVAS, CANVAS), 128.0)
    Yun[SL, SL] = 128 + np.kron(P, np.ones((BMOD, BMOD))) + (0 if under is None else under[SL, SL])
    clip = float(((Yun < 0) | (Yun > 255)).mean()*(CANVAS/INNER)**2)
    side = dict(gains=gains, ref=ref, cells=cells, a_env=a_env, a_res=a_res, shift=shift, dr=dr,
                smooth=smooth, nblk=nblk, under=under)
    return np.clip(Yun, 0, 255), side, dict(x=x, env=env, clip=clip)

def decode_x(Y, s):
    Y = codec.luma(Y)
    inner = Y[SL, SL] - 128.0 - (0 if s["under"] is None else s["under"][SL, SL])
    P = inner.reshape(G, BMOD, G, BMOD).mean(axis=(1, 3))
    E = P.reshape(SIDE, T, SIDE, T).mean(axis=(1, 3))
    res = P - envelope_field(E, s["smooth"])
    grid = res.reshape(SIDE, T, SIDE, T).transpose(0, 2, 1, 3).reshape(NT, T, T)[s["cells"]]
    V = dctn(grid, axes=(1, 2), norm="ortho")
    env = np.clip((E.reshape(-1)[s["cells"]] - s["shift"])/s["a_env"], -1, 1)
    R = np.clip(V[:, _ZR, _ZC]/s["a_res"], -1, 1)
    with _patched(DR=s["dr"], NBLK=s["nblk"]):
        return codec.synthesize(codec.unwhiten(codec.unpack(codec.denormalize(env, R, s["ref"])), s["gains"]))


# ---------------------------------------------------------------- measures
def chroma_radius(rgb):
    """Mean chroma magnitude sqrt(Cb^2 + Cr^2) over the data square (a pure primary is ~127)."""
    a = rgb[SL, SL].astype(float); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(np.hypot(-0.168736*r - 0.331264*g + 0.5*b, 0.5*r - 0.418688*g - 0.081312*b).mean())

def measure(X, colour=None, **kw):
    """Mean over pieces of SNR per KEY condition (+ clip %, + delivered chroma if colour is given)."""
    per, clips, chroma = [], [], []
    for x in X.values():
        Y, side, info = encode_x(x, **kw)
        img = np.round(Y) if colour is None else colour(Y, side, info)
        per.append([snr(info["x"], decode_x(transmit(img, q, s, r), side)) for _, q, s, r in KEY])
        clips.append(info["clip"])
        if colour is not None:
            chroma.append(chroma_radius(img))
    return np.mean(per, 0), 100*float(np.mean(clips)), (float(np.mean(chroma)) if chroma else None), np.array(per)


# ---------------------------------------------------------------- colour: "harmony"
def tonal_centroid(ft, pal, sigma_frames=10):
    """Per STFT frame: where the sounding notes sit on the circle of fifths (angle, relative
    to the key) and how concentrated they are (0 = noise, ~0.64 = a clean triad, 1 = one note)."""
    ch = gaussian_filter1d(ft["chroma"], sigma_frames, axis=1, mode="nearest")
    ang = 2*np.pi*((7*(np.arange(12) - pal["rel"])) % 12)/12
    z = (ch*np.exp(1j*ang)[:, None]).sum(0)/(ch.sum(0) + 1e-12)
    return (np.angle(z) - np.pi/4)/(2*np.pi), np.abs(z)      # 0 turns = tonic major triad

def oklch_cbcr(L, C, h_turn):
    """Chroma offsets (Cb, Cr) of the OKLCH colour (L, C, hue). Hue in turns, 0 = red.
    OKLCH is perceptually even, so equal C looks equally vivid at every hue."""
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

_TMAP = []
def disc_time_map():
    """Tile-time coordinate (0 = first tile) of every canvas pixel for the disc_in layout,
    continuous along the spiral, so colour can be drawn round instead of per 16 px tile."""
    if not _TMAP:
        r, c = np.divmod(np.arange(NT), SIDE)
        x = c - (SIDE - 1)/2; y = r - (SIDE - 1)/2
        th = (np.arctan2(x, -y) % (2*np.pi))/(2*np.pi)
        keys = np.sort(np.floor(np.hypot(x, y) - th) + th)
        p = (np.arange(CANVAS) - ORIGIN + 0.5)/(T*BMOD) - SIDE/2
        xx, yy = np.meshgrid(p, p)
        thp = (np.arctan2(xx, -yy) % (2*np.pi))/(2*np.pi)
        _TMAP.append(NT - 1 - np.interp(np.floor(np.hypot(xx, yy) - thp) + thp, keys, np.arange(NT)))
    return _TMAP[0]

def look_rings(Y, ft, pal, cells, span=0.45, vivid=1.0, loud_shift=0.06, blur=5.0, tsig=10, tmap=None):
    """Colour that follows the time path continuously (round rings for the disc_in layout).
    tmap       = tile-time of every canvas pixel; None = the disc_in layout
    hue        = key hue + span * harmony (tonal centroid) + a small loudness shift
    vividness  = how clearly tonal the moment is, times loudness
    span 0.45  = related colours around the key hue; 1.0 = each chord gets its own hue
    tsig       = time smoothing of the harmony in 16 ms frames (2 shows single notes)"""
    if tmap is None:
        assert np.array_equal(cells, disc_layout(True)), "look_rings needs disc_in or an explicit tmap"
        tmap = disc_time_map()
    t = per_tile(ft, pal)
    j = np.clip(tile_frame(np.arange(NT)), 0, ft["S"].shape[1] - 1)
    turn, mag = tonal_centroid(ft, pal, tsig)
    loud = gaussian_filter1d(t["loud"], 6, mode="nearest")
    hue = pal["h0"] + span*turn[j] + pal["sgn"]*loud_shift*(loud - 0.5)
    v = vivid*(0.45 + 0.55*np.clip(mag[j]/0.64, 0, 1))*(0.4 + 0.6*loud)
    cb_t, cr_t = oklch_cbcr(0.62, 0.03 + 0.17*v, hue)
    return (gaussian_filter(np.interp(tmap, np.arange(NT), cb_t), blur),
            gaussian_filter(np.interp(tmap, np.arange(NT), cr_t), blur))

def render_x(x, chroma_fn=look_rings, chroma_kw=None, **kw):
    """Experimental render: returns rgb, Y, side, info, features."""
    Y, side, info = encode_x(x, **kw)
    ft = features(info["x"]); pal = palette(ft)
    cb, cr = chroma_fn(Y, ft, pal, side["cells"], **(chroma_kw or {}))
    return colorize(Y, cb, cr), Y, side, info, ft

def run(X, chroma_fn, chroma_kw=None, **kw):
    """Colour render + measurement of one configuration on every piece in X."""
    out = dict(snr={}, chroma={}, clip={}, img={})
    for pk, x in X.items():
        rgb, Y, side, info, _ = render_x(x, chroma_fn, chroma_kw, **kw)
        out["snr"][pk] = [float(snr(info["x"], decode_x(transmit(rgb, q, s, r), side))) for _, q, s, r in KEY]
        out["chroma"][pk] = chroma_radius(rgb); out["clip"][pk] = 100*info["clip"]; out["img"][pk] = rgb
    out["mean"] = np.mean(list(out["snr"].values()), 0)
    return out

def closeup(rgb, at=150, size=200, zoom=2):
    """1:1 crop enlarged without smoothing, so single modules stay visible."""
    c = slice(at, at + size)
    return np.asarray(Image.fromarray(rgb[c, c]).resize((size*zoom, size*zoom), Image.NEAREST))


FREE = dict(layout="disc_in", centre=128, smooth=1.0, dr=6.0)       # the candidate, grain unchanged

if __name__ == "__main__":
    from figures import sheet
    X = {pk: fn() for pk, fn in PIECES.items()}
    base = dict(layout="spiral_in", a_env=60.0, a_res=75.0)
    levers = [
        ("current format (spiral_in, 60/75)",        {}),
        ("round spiral layout (disc_in)",            dict(layout="disc_in")),
        ("median tile at luma 128",                  dict(centre=128)),
        ("median tile at luma 112",                  dict(centre=112)),
        ("envelope range 7 log2 (42 dB)",            dict(dr=7.0)),
        ("envelope range 6 log2 (36 dB)",            dict(dr=6.0)),
        ("envelope range 5 log2 (30 dB)",            dict(dr=5.0)),
        ("smooth envelope, blend 0.5 tile",          dict(smooth=0.5)),
        ("smooth envelope, blend 1.0 tile",          dict(smooth=1.0)),
        ("whitening: 8 time blocks (2 s)",           dict(nblk=8)),
        ("whitening: 1 time block (static)",         dict(nblk=1)),
        ("grain amplitude 60 (60/60)",               dict(a_res=60.0)),
        ("grain amplitude 45 (60/45)",               dict(a_res=45.0)),
    ]
    print("1. Single levers, grey, mean of pieces A/B/C - audio SNR (dB): lossless / screenshot 0.15 px / worst")
    results = dict(levers={}, ladder={}, variants={})
    ref = None
    for name, kw in levers:
        m, clip, _, _ = measure(X, **{**base, **kw})
        ref = m if ref is None else ref
        d = m - ref
        results["levers"][name] = dict(snr=m.tolist(), clip=clip)
        print(f"  {name:<40} {m[0]:5.2f} / {m[1]:5.2f} / {m[2]:5.2f}   "
              f"delta {d[0]:+5.2f} / {d[1]:+5.2f} / {d[2]:+5.2f}   clip {clip:4.2f} %")

    ladder = [
        ("1 current Record look",          look_record, dict(layout="spiral_in")),
        ("2 + luma centred on 128",        look_record, dict(layout="spiral_in", centre=128)),
        ("3 + smooth envelope",            look_record, dict(layout="spiral_in", centre=128, smooth=1.0)),
        ("4 + round spiral",               look_record, dict(layout="disc_in", centre=128, smooth=1.0)),
        ("5 + round harmony colour",       look_rings,  dict(layout="disc_in", centre=128, smooth=1.0)),
        ("6 + envelope range 36 dB",       look_rings,  dict(FREE)),
        ("7 + grain 62 (was 75)",          look_rings,  dict(FREE, a_res=62.0)),
        ("8 + loud = dark",                look_rings,  dict(FREE, a_res=62.0, a_env=-60.0)),
    ]
    print("\n2. Step ladder, colour, mean of A/B/C - SNR (dB) lossless / screenshot 0.15 px / worst;"
          " delivered chroma radius; clip %")
    L = {}
    for name, fn, kw in ladder:
        L[name] = r = run(X, fn, **kw)
        results["ladder"][name] = dict(snr=r["snr"], chroma=r["chroma"], clip=r["clip"])
        print(f"  {name:<28} {r['mean'][0]:5.2f} / {r['mean'][1]:5.2f} / {r['mean'][2]:5.2f}   "
              f"chroma {np.mean(list(r['chroma'].values())):4.1f}   clip {np.mean(list(r['clip'].values())):4.2f} %"
              f"   screenshot per piece " + " ".join(f"{pk} {v[1]:.2f}" for pk, v in r["snr"].items()))
    sheet(OUT / "explore_ladder_A.png",
          [(f"{n}: {L[n]['snr']['A'][1]:.1f} dB", L[n]["img"]["A"]) for n, _, _ in ladder], 4, 400,
          "Piece A, one change at a time (dB = audio SNR after a simulated screenshot, 0.15 px)")

    old, new, calm = L[ladder[0][0]], L[ladder[5][0]], L[ladder[6][0]]
    cells = []
    for pk in X:
        cells += [(f"{pk} today: {old['snr'][pk][1]:.1f} dB, colour {old['chroma'][pk]:.0f}", old["img"][pk]),
                  (f"{pk} candidate: {new['snr'][pk][1]:.1f} dB, colour {new['chroma'][pk]:.0f}", new["img"][pk]),
                  (f"{pk} + grain 62: {calm['snr'][pk][1]:.1f} dB, colour {calm['chroma'][pk]:.0f}", calm["img"][pk]),
                  (f"{pk} today, close-up (pixels x2)", closeup(old["img"][pk])),
                  (f"{pk} candidate, close-up", closeup(new["img"][pk]))]
    sheet(OUT / "explore_before_after.png", cells, 5, 400,
          "Today (Record) vs candidate: centred luma, smooth envelope, round spiral, round harmony colour")
    for pk in X:
        Image.fromarray(new["img"][pk]).save(OUT / f"explore_candidate_{pk}.png")

    variants = [
        ("A related hues (span 0.45)",        "A", look_rings, {},                    dict(FREE)),
        ("A one hue per chord (span 1.0)",    "A", look_rings, dict(span=1.0),        dict(FREE)),
        ("A loud = dark",                     "A", look_rings, {},                    dict(FREE, a_env=-60.0)),
        ("A grain 55",                        "A", look_rings, {},                    dict(FREE, a_res=55.0)),
        ("C luma centre 128",                 "C", look_rings, {},                    dict(FREE)),
        ("C luma centre 165, loud = dark",    "C", look_rings, {},                    dict(FREE, centre=165, a_env=-60.0)),
        ("C centre 165 + single notes",       "C", look_rings, dict(span=1.0, tsig=2, blur=3.0), dict(FREE, centre=165, a_env=-60.0)),
        ("B loud = dark",                     "B", look_rings, {},                    dict(FREE, a_env=-60.0)),
    ]
    print("\n3. Taste switches, colour, single piece - SNR (dB) lossless / screenshot 0.15 px / worst; chroma; clip %")
    cells = []
    for name, pk, fn, ckw, kw in variants:
        r = run({pk: X[pk]}, fn, ckw, **kw)
        v = r["snr"][pk]
        results["variants"][name] = dict(snr=v, chroma=r["chroma"][pk], clip=r["clip"][pk])
        print(f"  {name:<34} {v[0]:5.2f} / {v[1]:5.2f} / {v[2]:5.2f}   chroma {r['chroma'][pk]:4.1f}   clip {r['clip'][pk]:4.2f} %")
        cells.append((f"{name}: {v[1]:.1f} dB", r["img"][pk]))
    sheet(OUT / "explore_variants.png", cells, 4, 400, "Taste switches (dB = audio SNR after a simulated screenshot, 0.15 px)")
    save_json("explore_aesthetics.json", results)
