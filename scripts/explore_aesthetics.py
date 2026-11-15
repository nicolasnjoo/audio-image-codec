"""EXPERIMENTAL - aesthetic levers measured against the format.

Centred brightness, the smooth envelope and the decodable ripple were adopted and now live in
a2i/codec.py and a2i/shapes.py. This script keeps the measurements that led there, and the
levers that were measured but NOT adopted:

  dr       envelope range in log2 units (format: 9 = 54 dB)
  nblk     whitening time blocks (format: 32)
  a_res    grain amplitude (format: 75)
  layout   any codec layout, "disc_in" (round spiral) or an explicit permutation

Run:  python scripts/explore_aesthetics.py     (lever table, step ladder, sheets in out/explore_*.png)
"""
import contextlib
import numpy as np
from PIL import Image
from _common import OUT, save_json
from a2i import codec, shapes
from a2i.codec import INNER, ORIGIN
from a2i.looks import colorize
from a2i.features import features
from a2i.channel import transmit, snr
from a2i.testsignals import PIECES
from render_options import palette, look_record, look_droplets

KEY = [("lossless", None, False, 0.0), ("screenshot 0.15 px", 75, True, 0.15), ("worst", 60, True, 0.5)]
SL = slice(ORIGIN, ORIGIN + INNER)


# ---------------------------------------------------------------- layout
def disc_layout(inward=True):
    """Round spiral, one tile per turn (the "disc" landscape of a2i.shapes)."""
    cells = shapes.build("disc", ripple=0.0)[0]
    return cells if inward else cells[::-1].copy()

def get_layout(name):
    """A codec layout name, "disc_in", or an explicit permutation."""
    if isinstance(name, np.ndarray):
        return name
    return disc_layout(True) if name == "disc_in" else codec.layout(name)


# ---------------------------------------------------------------- format + the two unadopted constants
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
    """codec.encode_luma with the envelope range and the whitening blocks made variable."""
    with _patched(DR=dr, NBLK=nblk):
        Y, side, info = codec.encode_luma(x, cells=get_layout(layout), a_env=a_env, a_res=a_res,
                                          centre=centre, smooth=smooth, under=under)
    side.update(dr=dr, nblk=nblk)
    return Y, side, info

def decode_x(Y, s):
    with _patched(DR=s["dr"], NBLK=s["nblk"]):
        return codec.decode(Y, s)


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


# ---------------------------------------------------------------- colour along the time path
def look_rings(Y, ft, pal, cells, tmap=None, **kw):
    """render_options.look_droplets on the round spiral (or any layout, given its tmap).
    Options: span (0.45 related hues, 1.0 one hue per chord), tsig (2 shows single notes)."""
    if tmap is None:
        assert np.array_equal(cells, disc_layout(True)), "look_rings needs disc_in or an explicit tmap"
        tmap = shapes.build("disc", ripple=0.0)[2]
    return look_droplets(Y, ft, pal, cells, tmap, **kw)

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


FREE = dict(layout="disc_in", centre=128, smooth=1.0)       # the adopted changes on a round spiral

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
        ("5 + round harmony colour",       look_rings,  dict(FREE)),
        ("6 option: envelope 36 dB",       look_rings,  dict(FREE, dr=6.0)),
        ("7 option: grain 62 (was 75)",    look_rings,  dict(FREE, a_res=62.0)),
        ("8 option: loud = dark",          look_rings,  dict(FREE, a_env=-60.0)),
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

    old, new, calm = L[ladder[0][0]], L[ladder[4][0]], L[ladder[6][0]]
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
