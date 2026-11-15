"""EXPERIMENTAL - other shapes generated from the music (droplets was adopted: a2i/shapes.py).

Same principle as a2i.shapes: a landscape over the image, time runs along its contour lines.
These two landscapes also read brightness and tonality from the descriptor (7 bytes in all):

  flower    the scale drawn as a star: one lobe per pitch class on a 12-hour clock (tonic at
            12), lobe length = how much that note is used; bright music -> sharp lobes
  chladni   the figure of a vibrating plate: each pitch class excites one plate mode (the mode
            whose frequency is closest to that note), weighted by how much the note is used

Run:  python scripts/explore_shapes.py      (measurements + out/shapes_*.png)
"""
import numpy as np
from PIL import Image
from _common import OUT, save_json
from a2i import codec
from a2i.codec import NT, SIDE
from a2i.looks import colorize
from a2i.features import features
from a2i.channel import transmit, snr
from a2i.shapes import TAU, RIPPLE, descriptor, tile_xy, angle, disc, droplets, layout_from, fields_from
from a2i.testsignals import PIECES, _note, FS, NS
from render_options import palette, look_droplets, brightness_for, SMOOTH
from explore_aesthetics import chroma_radius, KEY


# ---------------------------------------------------------------- landscapes: f(x, y) -> (h, angle)
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

SHAPES = {"disc": lambda d: disc(), "droplets": lambda d: droplets(d["w"]),
          "flower": shape_flower, "chladni": shape_chladni}


# ---------------------------------------------------------------- render + measure
def render_shape(x, shape, ripple=RIPPLE, auto=True, chroma_kw=None, **kw):
    """Like render_options.render_droplets, for any landscape in SHAPES."""
    ft = features(codec.prepare(x)); pal = palette(ft); d = descriptor(ft)
    cells, under, tmap = fields_from(SHAPES[shape](d), ripple)
    centre, sign = brightness_for(pal) if auto else (128.0, 1.0)
    cfg = dict(a_env=sign*60.0, a_res=75.0, centre=centre, smooth=SMOOTH); cfg.update(kw)
    Y, side, info = codec.encode_luma(x, cells=cells, under=under, **cfg)
    cb, cr = look_droplets(Y, ft, pal, cells, tmap, **(chroma_kw or {}))
    return colorize(Y, cb, cr), Y, side, info, d

def measure_shape(X, shape, **kw):
    per, ch, img = [], [], {}
    for pk, x in X.items():
        rgb, Y, side, info, d = render_shape(x, shape, **kw)
        per.append([float(snr(info["x"], codec.decode(transmit(rgb, q, s, r), side))) for _, q, s, r in KEY])
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

if __name__ == "__main__":
    from figures import sheet
    X = {pk: fn() for pk, fn in PIECES.items()}
    results = dict(ripple={}, shapes={})

    print("1. Contour ripple (droplets), colour, mean of pieces A/B/C - audio SNR (dB)\n"
          "   lossless / screenshot 0.15 px / worst; delivered chroma radius")
    for label, kw in [("no ripple, fixed brightness 128", dict(ripple=0.0, auto=False)),
                      ("no ripple, brightness by hue",    dict(ripple=0.0)),
                      ("ripple 8 (adopted)",              dict(ripple=8.0)),
                      ("ripple 14",                       dict(ripple=14.0)),
                      ("ripple 22",                       dict(ripple=22.0))]:
        per, ch, _ = measure_shape(X, "droplets", **kw)
        m = per.mean(0); results["ripple"][label] = dict(snr=per.tolist(), chroma=ch)
        print(f"  {label:<32} {m[0]:5.2f} / {m[1]:5.2f} / {m[2]:5.2f}   chroma {np.mean(ch):4.1f}"
              f"   screenshot per piece " + " ".join(f"{pk} {p[1]:.2f}" for pk, p in zip(X, per)))

    print(f"\n2. Shapes generated from the music (ripple {RIPPLE:g}, brightness by hue), colour, mean of A/B/C")
    imgs = {}
    for name in SHAPES:
        per, ch, imgs[name] = measure_shape(X, name)
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
          f"Shapes generated from each piece's notes, ripple {RIPPLE:g} (dB = audio SNR after a simulated screenshot, 0.15 px)")

    clips = {label: clip_of(notes, kind) for label, (notes, kind) in EXTRA.items()}
    sheet(OUT / "shapes_gallery.png",
          [(f"{name}: {label}", render_shape(x, name)[0])
           for name in ("droplets", "flower", "chladni") for label, x in clips.items()], 6, 300,
          f"Same generators fed with different notes (six synthetic clips), ripple {RIPPLE:g}")
    save_json("shapes.json", results)
