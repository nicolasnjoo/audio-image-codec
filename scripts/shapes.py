"""EXPERIMENTAL - shapes that are generated from the music instead of chosen from a list.

Principle: a layout is only an ordering of the 4096 tiles. Define a landscape h(x, y) over the
image and let time run along its contour lines (inward spiral: key = floor(h - angle) + angle,
tiles sorted by key). The circle is the landscape "distance from the centre"; the square
spiral is the same thing with a square distance. Any other landscape gives another shape, and
every ordering is still a permutation, so the audio capacity is unchanged.

The landscape is built from a 7-byte descriptor of the music (12 pitch-class weights, brightness,
tonality; 4 bits each). The decoder would read those 7 bytes from the header and rebuild the
identical ordering, so nothing here is a preset: the shape is a function of the notes.

  flower    the scale drawn as a star: one lobe per pitch class on a 12-hour clock (tonic at
            12), lobe length = how much that note is used; bright music -> sharp lobes
  droplets  one drop per prominent note at its clock position, size = its weight; the drops
            merge outwards, and every drop plays the same moment at once
  chladni   the figure of a vibrating plate: each pitch class excites one plate mode (the mode
            whose frequency is closest to that note), weighted by how much the note is used

NOTE: float sort keys; a shipped format needs an integer specification (roadmap step 10).
Run:  python scripts/shapes.py      (measurements + out/shapes_*.png)
"""
import numpy as np
from PIL import Image
from _common import OUT, save_json
from a2i import codec
from a2i.codec import NT, SIDE, T, BMOD, CANVAS, ORIGIN
from a2i.looks import colorize, _K
from a2i.features import features
from a2i.channel import transmit, snr
from a2i.testsignals import PIECES, _note, FS, NS
from render_options import palette
from explore_aesthetics import encode_x, decode_x, look_rings, chroma_radius, oklch_cbcr, FREE, KEY

TAU = 2*np.pi


# ---------------------------------------------------------------- music -> 7 bytes
def descriptor(ft):
    """12 pitch-class weights (tonic first), brightness, tonality: integers 0..15 each."""
    cm = np.roll(ft["chroma"].sum(1), -ft["tonic"])
    return dict(w=np.round(15*np.sqrt(cm/cm.max())).astype(int).tolist(),
                bright=int(np.clip(round(15*(ft["brightness"] - 300)/2500), 0, 15)),
                tonal=int(np.clip(round(15*ft["tonality"]), 0, 15)))


# ---------------------------------------------------------------- geometry
def tile_xy():
    r, c = np.divmod(np.arange(NT), SIDE)
    return c - (SIDE - 1)/2, r - (SIDE - 1)/2               # tile units, origin at the centre

def pixel_xy():
    p = (np.arange(CANVAS) - ORIGIN + 0.5)/(T*BMOD) - SIDE/2
    return np.meshgrid(p, p)

def angle(x, y):
    return (np.arctan2(x, -y) % TAU)/TAU                    # turns, 0 at 12 o'clock, clockwise


# ---------------------------------------------------------------- landscapes: f(x, y) -> (h, angle)
def shape_disc(d):
    return lambda x, y: (np.hypot(x, y), angle(x, y))

def shape_flower(d):
    w = np.array(d["w"])/15
    kappa = 6 + 2.3*d["bright"]                             # lobe sharpness: dark = round, bright = spiky
    depth = 0.3 + 0.06*d["tonal"]                           # lobe length: noisy = shallow, tonal = deep
    def lobes(th):
        return sum(w[k]*np.exp(kappa*(np.cos(TAU*(th - k/12)) - 1)) for k in range(12))
    peak = lobes(np.arange(720)/720).max()
    def f(x, y):
        th = angle(x, y)
        return np.hypot(x, y)/(1 + depth*lobes(th)/peak), th
    return f

def shape_droplets(d, ring=13.0, beta=0.45):
    w = np.array(d["w"])
    ks = [int(k) for k in np.argsort(-w, kind="stable")[:6] if w[k] >= 8]       # prominent notes
    cx = np.array([ring*np.sin(TAU*k/12) for k in ks]); cy = np.array([-ring*np.cos(TAU*k/12) for k in ks])
    rad = np.array([1.5 + 5*w[k]/15 for k in ks])
    def raw(x, y):
        dist = np.stack([np.hypot(x - a, y - b) - r for a, b, r in zip(cx, cy, rad)])
        j = dist.argmin(0)
        return -np.log(np.exp(-beta*dist).sum(0))/beta, angle(x - cx[j], y - cy[j])
    h0 = raw(*tile_xy())[0].min()
    def f(x, y):
        h, th = raw(x, y)
        return h - h0, th
    return f

def plate_modes(nmax=8):
    """For each pitch class k: the plate mode (m, n) whose frequency sqrt(m^2 + n^2), folded into
    one octave above the (1, 1) mode, is closest to k semitones (simplest mode wins ties)."""
    out = []
    for k in range(12):
        best = None
        for n in range(1, nmax + 1):
            for m in range(1, n + 1):
                st = (12*np.log2(np.hypot(m, n)/np.sqrt(2))) % 12
                err = min(abs(st - k), 12 - abs(st - k)) + 0.03*(m + n)
                if best is None or err < best[0]:
                    best = (err, m, n)
        out.append(best[1:])
    return out

def shape_chladni(d, bands=34):
    w = (np.array(d["w"])/15)**2                            # the main notes dominate
    s = 0.25 + 0.25*d["bright"]/15                          # dark = broad forms, bright = finer forms
    modes = plate_modes()
    def H(x, y):
        u, v = np.pi*s*x/(SIDE/2), np.pi*s*y/(SIDE/2)
        return sum(w[k]*(np.cos(m*u)*np.cos(n*v) + np.cos(n*u)*np.cos(m*v)) for k, (m, n) in enumerate(modes))
    Hs = np.sort(H(*tile_xy()))
    return lambda x, y: (bands*np.interp(H(x, y), Hs, np.arange(NT)/NT), angle(x, y))

SHAPES = {"disc": shape_disc, "flower": shape_flower, "droplets": shape_droplets, "chladni": shape_chladni}


# ---------------------------------------------------------------- landscape -> layout, time map
def spiral_key(h, th):
    return np.floor(h - th) + th

def layout_from(f):
    """cells[t] = grid cell of tile t; time starts on the highest contour and ends on the lowest."""
    key = spiral_key(*f(*tile_xy()))
    order = np.lexsort((np.arange(NT), key))
    return order[::-1].copy(), key[order]


# ---------------------------------------------------------------- brightness that suits the hue
def best_luma(pal, head=16.0):
    """Luma at which the key hue has the most room inside the RGB cube (yellow: high, blue: low)."""
    cb, cr = oklch_cbcr(0.62, 0.15, pal["h0"])
    d = np.array([_K[i, 0]*cb + _K[i, 1]*cr for i in range(3)])/np.hypot(cb, cr)
    ys = np.arange(40, 216)
    room = [min(((255 - head - y)/v if v > 0 else (y - head)/-v) for v in d if abs(v) > 1e-9) for y in ys]
    return int(ys[int(np.argmax(room))])

def auto_brightness(pal):
    """Raise the median tile towards the luma the hue needs (never below 128, at most 165) and,
    when raised, draw loud moments dark so that they move towards mid grey instead of clipping."""
    centre = float(np.clip(best_luma(pal), 128, 165))
    return dict(centre=centre, a_env=-60.0 if centre > 128 else 60.0)


# ---------------------------------------------------------------- render + measure
def render_shape(x, shape, ripple=0.0, auto=True, chroma_kw=None, **kw):
    """ripple: amplitude (grey levels) of a light/dark wave drawn along the contour lines. It is a
    known function of the layout, so the decoder subtracts it; it costs only luma headroom."""
    xp = codec.prepare(x)
    ft = features(xp); pal = palette(ft); d = descriptor(ft)
    f = SHAPES[shape](d)
    cells, keys = layout_from(f)
    h, th = f(*pixel_xy())
    tmap = NT - 1 - np.interp(spiral_key(h, th), keys, np.arange(NT))    # tile-time of every pixel
    under = ripple*np.cos(TAU*(h - th)) if ripple else None
    cfg = {**FREE, **(auto_brightness(pal) if auto else {}), **kw, "layout": cells, "under": under}
    Y, side, info = encode_x(x, **cfg)
    cb, cr = look_rings(Y, ft, pal, cells, tmap=tmap, **(chroma_kw or {}))
    return colorize(Y, cb, cr), Y, side, info, d

def measure_shape(X, shape, **kw):
    per, ch, img = [], [], {}
    for pk, x in X.items():
        rgb, Y, side, info, d = render_shape(x, shape, **kw)
        per.append([float(snr(info["x"], decode_x(transmit(rgb, q, s, r), side))) for _, q, s, r in KEY])
        ch.append(chroma_radius(rgb)); img[pk] = rgb
    return np.array(per), ch, img

def clip_of(freqs, kind="pad", seed=1):
    """Small synthetic clips used only to show how the shape reacts to different notes."""
    rng = np.random.default_rng(seed); t = np.arange(NS)/FS; x = np.zeros(NS)
    if kind == "pad":                                       # sustained chord, dark
        for f0 in freqs:
            x += _note(t, f0, 0.5, n_h=5, roll=2.0, rng=rng)
        x *= 0.6 + 0.4*np.sin(2*np.pi*0.25*t)**2
    else:                                                   # plucked arpeggio, bright
        step = FS//8
        for j, k in enumerate(range(0, NS, step)):
            n = min(2*step, NS - k); tt = np.arange(n)/FS
            x[k:k+n] += np.exp(-tt*10)*_note(tt, freqs[j % len(freqs)], 0.4, n_h=10, roll=0.8, rng=rng)
    return x/(np.abs(x).max()*1.02)

C4 = 261.6256
def hz(*semitones):
    return [C4*2**(s/12) for s in semitones]

EXTRA = {                                                   # name: (notes, kind)
    "one note (C)":            (hz(0), "pad"),
    "major triad":             (hz(0, 4, 7), "pad"),
    "diminished 7th":          (hz(0, 3, 6, 9), "pluck"),
    "whole-tone scale":        (hz(0, 2, 4, 6, 8, 10), "pluck"),
    "major scale":             (hz(0, 2, 4, 5, 7, 9, 11), "pluck"),
    "chromatic cluster":       (hz(0, 1, 2, 3, 4, 5), "pad"),
}

RIPPLE = 14.0

if __name__ == "__main__":
    from figures import sheet
    X = {pk: fn() for pk, fn in PIECES.items()}
    results = dict(ripple={}, shapes={})

    print("1. Contour ripple (flower shape), colour, mean of pieces A/B/C - audio SNR (dB)\n"
          "   lossless / screenshot 0.15 px / worst; delivered chroma radius")
    for label, kw in [("no ripple, fixed brightness 128", dict(auto=False)),
                      ("no ripple, brightness by hue",    dict()),
                      ("ripple 8",                        dict(ripple=8.0)),
                      ("ripple 14",                       dict(ripple=14.0)),
                      ("ripple 22",                       dict(ripple=22.0))]:
        per, ch, _ = measure_shape(X, "flower", **kw)
        m = per.mean(0); results["ripple"][label] = dict(snr=per.tolist(), chroma=ch)
        print(f"  {label:<32} {m[0]:5.2f} / {m[1]:5.2f} / {m[2]:5.2f}   chroma {np.mean(ch):4.1f}"
              f"   screenshot per piece " + " ".join(f"{pk} {p[1]:.2f}" for pk, p in zip(X, per)))

    print(f"\n2. Shapes generated from the music (ripple {RIPPLE:g}, brightness by hue), colour, mean of A/B/C")
    cells = []
    imgs = {}
    for name in SHAPES:
        per, ch, imgs[name] = measure_shape(X, name, ripple=RIPPLE)
        m = per.mean(0); results["shapes"][name] = dict(snr=per.tolist(), chroma=ch)
        print(f"  {name:<9} {m[0]:5.2f} / {m[1]:5.2f} / {m[2]:5.2f}   chroma {np.mean(ch):4.1f}"
              f"   screenshot per piece " + " ".join(f"{pk} {p[1]:.2f}" for pk, p in zip(X, per)))
        for pk in X:
            Image.fromarray(imgs[name][pk]).save(OUT / f"shape_{name}_{pk}.png")
    for pk, x in X.items():
        print(f"  descriptor {pk}: {descriptor(features(codec.prepare(x)))}")
    sheet(OUT / "shapes_pieces.png",
          [(f"{pk} {name}: {results['shapes'][name]['snr'][i][1]:.1f} dB", imgs[name][pk])
           for i, pk in enumerate(X) for name in SHAPES], 4, 400,
          "Shapes generated from each piece's notes (dB = audio SNR after a simulated screenshot, 0.15 px)")

    clips = {label: clip_of(notes, kind) for label, (notes, kind) in EXTRA.items()}
    sheet(OUT / "shapes_gallery.png",
          [(f"{name}: {label}", render_shape(x, name, ripple=RIPPLE)[0])
           for name in ("flower", "droplets", "chladni") for label, x in clips.items()], 6, 300,
          "Same three generators fed with different notes (six synthetic clips)")
    save_json("shapes.json", results)
