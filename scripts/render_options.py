"""Render the looks for the three pieces and measure them (section 7F, 7G).
The four original looks use the first format. Droplets (the adopted default) uses the format
additions: centred brightness, smooth envelope, a layout generated from the notes, ripple."""
import json, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from _common import OUT, compare, save_json
from a2i import shapes
from a2i.codec import encode_luma, decode, layout, prepare, CANVAS, INNER, NT, FS
from a2i.looks import colorize, hue_cbcr, tiles_to_canvas, lerp_hue, oklch_cbcr, best_luma
from a2i.features import features, tile_frame, tonal_centroid
from a2i.channel import transmit, snr
from a2i.testsignals import PIECES

A_ENV, A_RES = 60, 75
KEY = [("lossless", None, False, 0.0), ("screenshot", 75, True, 0.15), ("worst", 60, True, 0.5)]

def palette(ft):
    rel = (ft["tonic"] + (3 if ft["mode"] == "minor" else 0)) % 12
    h0 = ((7*rel) % 12)/12 + 0.04
    sgn = 1 if ft["mode"] == "major" else -1
    spread = 0.10 + 0.12*np.clip((ft["brightness"] - 400)/2600, 0, 1)
    sat = float(np.clip(0.45 + 0.55*ft["tonality"], 0.45, 1.0))
    return dict(h0=h0 % 1, h1=(h0 + sgn*spread) % 1, sgn=sgn, sat=sat, rel=rel)

def per_tile(ft, pal):
    j = np.clip(tile_frame(np.arange(NT), ft["hop"]), 0, ft["S"].shape[1] - 1)
    L = 20*np.log10(ft["rms"] + 1e-9); Ln = np.clip((L - (L.max() - 42))/42, 0, 1)
    pc = ft["chroma"].argmax(0)
    d = (7*(pc - pal["rel"])) % 12; dsig = ((d + 6) % 12) - 6
    return dict(loud=Ln[j], dist=dsig[j], onset=ft["flux"][j])

def border_fill(pal, r=28):
    return hue_cbcr(pal["h0"], r*pal["sat"])

def look_record(Y, ft, pal, cells):
    t = per_tile(ft, pal)
    hue = lerp_hue(pal["h0"], pal["h1"], 0.6*t["loud"]) + 0.022*t["dist"]
    rad = 72*pal["sat"]*(0.2 + 0.8*t["loud"]) + 18*t["onset"]
    cb_t, cr_t = hue_cbcr(hue, rad); fb, fr = border_fill(pal)
    return tiles_to_canvas(cb_t, cells, 5, fb), tiles_to_canvas(cr_t, cells, 5, fr)

def look_duotone(Y, ft, pal, cells):
    tt = Y/255.0
    shadow = (pal["h1"] + 0.45) % 1
    hue = lerp_hue(shadow, pal["h1"], np.clip((tt - 0.2)/0.6, 0, 1))
    rad = 80*pal["sat"]*np.sqrt(np.clip(4*tt*(1 - tt), 0, 1))
    return hue_cbcr(hue, rad)

def look_mandala(Y, ft, pal, cells):
    S = ft["S"]; f = ft["f"]; nfr = S.shape[1]
    rb = 256; fr_ = 50*(5000/50)**(np.arange(rb)/(rb - 1))
    idx = np.clip(np.searchsorted(f, fr_), 0, len(f) - 1)
    Ed = 20*np.log10(S[idx] + 1e-9); row = Ed.mean(1, keepdims=True)
    wgt = np.clip((row - (row.max() - 45))/45, 0.2, 1.0)
    E = np.clip((Ed - row + 8)/18, 0, 1)**0.8 * wgt
    E = gaussian_filter(E, (2.0, 1.5))
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS] - (CANVAS - 1)/2
    rho = np.hypot(xx, yy)/(INNER/2)
    u = (np.arctan2(xx, -yy)/(2*np.pi)) % 1.0
    e = map_coordinates(E, [np.clip(rho, 0, 1)*(rb - 1), u*(nfr - 1)], order=1, mode="nearest")
    e *= np.clip((1.04 - rho)/0.08, 0, 1)
    hue = lerp_hue(pal["h0"], pal["h1"], np.clip(rho, 0, 1)) + 0.06*np.sin(2*np.pi*u)
    rad = pal["sat"]*(10 + 100*e)
    cb, cr = hue_cbcr(hue, rad)
    return gaussian_filter(cb, 1.5), gaussian_filter(cr, 1.5)

def look_poster(Y, ft, pal, cells):
    t = per_tile(ft, pal)
    glow = tiles_to_canvas(t["loud"], cells, 38, fill=0.0)
    glow = (glow - glow.min())/(np.ptp(glow) + 1e-9)
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]/CANVAS
    diag = np.clip(0.5*(xx + yy), 0, 1)
    hue = lerp_hue(pal["h0"], pal["h1"], diag) + 0.05*(glow - 0.5)
    rad = 70*pal["sat"]*(0.35 + 0.65*glow)
    return hue_cbcr(hue, rad)

def look_droplets(Y, ft, pal, cells, tmap, span=0.45, vivid=1.0, loud_shift=0.06, blur=5.0, tsig=10):
    """Colour that follows the time path continuously, so it has no 16 px steps.
    tmap       tile-time of every canvas pixel (from a2i.shapes)
    hue        key hue + span * harmony (tonal centroid on the circle of fifths) + a small
               loudness shift. span 0.45 keeps related colours around the key hue.
    vividness  how clearly tonal the moment is, times loudness. Perceptual (OKLCH) wheel."""
    t = per_tile(ft, pal)
    j = np.clip(tile_frame(np.arange(NT), ft["hop"]), 0, ft["S"].shape[1] - 1)
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

LOOKS = {
    "Record":  ("spiral_in", look_record),
    "Mandala": ("spiral_in", look_mandala),
    "Duotone": ("hilbert",   look_duotone),
    "Poster":  ("hilbert",   look_poster),
}
DEFAULT_LOOK = "Droplets"
ALL_LOOKS = list(LOOKS) + [DEFAULT_LOOK]
SMOOTH = 1.0                                # adopted envelope blend width (tiles)
ENV_RANGE = 6.0                             # adopted envelope range: 6 log2 units = 36 dB
RATE = 32000                                # adopted audio rate for real music (8.064 s per image)

def render_droplets(x, a_env=A_ENV, a_res=A_RES, ripple=shapes.RIPPLE, shape="droplets",
                    hue_brightness=True, chroma_kw=None, dr=ENV_RANGE, weights=None,
                    fs_in=FS, rate=FS):
    """The adopted look. The side dict holds only header-sized fields for the layout: the shape
    name, the 12 note weights and the ripple amplitude; decode() rebuilds the rest.
    fs_in: sample rate of x. rate: audio rate carried by the image (the clip lasts NS/rate s)."""
    ft = features(prepare(x, fs_in, rate), rate); pal = palette(ft)
    spec = dict(name=shape, w=(shapes.weights if weights is None else weights)(ft), ripple=ripple)
    cells, under, tmap = shapes.build(**spec)
    centre, sign = brightness_for(pal) if hue_brightness else (128.0, 1.0)
    Y, side, info = encode_luma(x, fs_in, cells=cells, a_env=sign*a_env, a_res=a_res,
                                centre=centre, smooth=SMOOTH, under=under, dr=dr, fs=rate)
    side.update(cells=None, under=None, shape=spec)
    cb, cr = look_droplets(Y, ft, pal, cells, tmap, **(chroma_kw or {}))
    return colorize(Y, cb, cr), Y, side, info, ft

def render(x, name, a_env=A_ENV, a_res=A_RES, fs_in=FS, rate=FS):
    """Encode audio x with look `name`. Returns rgb (uint8), float luma Y, side, info, features.
    fs_in: sample rate of x. rate: audio rate carried by the image (defaults: 16 kHz, as the
    synthetic test pieces are 16 kHz signals; real music uses RATE)."""
    if name == DEFAULT_LOOK:
        return render_droplets(x, a_env, a_res, fs_in=fs_in, rate=rate)
    lay, chroma_fn = LOOKS[name]
    cells = layout(lay)
    Y, side, info = encode_luma(x, fs_in, cells=cells, a_env=a_env, a_res=a_res, fs=rate)
    ft = features(info["x"], rate); pal = palette(ft)
    cb, cr = chroma_fn(Y, ft, pal, cells)
    return colorize(Y, cb, cr), Y, side, info, ft

# Reference numbers (section 7F/7G): screenshot 0.15 px, mean of 3 pieces.
REF_F = {"Record": (16.48, 16.58, -0.11), "Mandala": (16.53, 16.58, -0.06),
         "Duotone": (16.39, 16.59, -0.20), "Poster": (16.50, 16.59, -0.09)}
REF_KEYS = {"A": "A minor", "B": "D minor", "C": "B minor"}
REF_BPM = {"A": 80, "B": 188, "C": 66}

if __name__ == "__main__":
    results, meta = {}, {}
    for pk, fn in PIECES.items():
        x = fn()
        for name in ALL_LOOKS:
            rgb, Y, side, info, ft = render(x, name)
            pal = palette(ft)
            meta[pk] = dict(key=ft["key"], bpm=round(ft["bpm"]), tonality=round(ft["tonality"], 2),
                            brightness=round(ft["brightness"]), h0=round(pal["h0"], 3))
            Image.fromarray(rgb).save(OUT / f"look_{name}_{pk}.png")
            col = [snr(info["x"], decode(transmit(rgb, q, s, r), side)) for _, q, s, r in KEY]
            gry = [snr(info["x"], decode(transmit(np.round(Y), q, s, r), side)) for _, q, s, r in KEY]
            results[f"{name}/{pk}"] = dict(colour=col, grey=gry)
            print(f"{name:<8} {pk}  colour " + " ".join(f"{v:5.1f}" for v in col) +
                  "   grey " + " ".join(f"{v:5.1f}" for v in gry))
    print("features:", json.dumps(meta))

    rows = []
    for name in ALL_LOOKS:
        rc, rg, rd = REF_F.get(name, (None, None, None))      # Droplets is new: no reference values
        ref = (lambda v: v) if name in REF_F else (lambda v: None)
        c = np.array([results[f"{name}/{pk}"]["colour"] for pk in PIECES])   # (piece, condition)
        g = np.array([results[f"{name}/{pk}"]["grey"] for pk in PIECES])
        rows += [(f"{name}: colour, screenshot 0.15 px", c[:, 1].mean(), rc),
                 (f"{name}: grey, screenshot 0.15 px", g[:, 1].mean(), rg),
                 (f"{name}: cost (colour - grey)", c[:, 1].mean() - g[:, 1].mean(), rd),
                 (f"{name}: colour, lossless", c[:, 0].mean(), ref(30.5)),
                 (f"{name}: grey, lossless", g[:, 0].mean(), None),
                 (f"{name}: colour, worst", c[:, 2].mean(), ref(7.3))]
        if name == "Duotone":
            rows.append(("Duotone: cost on worst piece", (c[:, 1] - g[:, 1]).min(), -0.35))
    table = compare("7F. Looks at 60/75, mean of pieces A/B/C - audio SNR (dB)", rows)

    print("\n7G. Features (reference in brackets)")
    for pk in PIECES:
        k_ok = "ok" if meta[pk]["key"] == REF_KEYS[pk] else "<-- DIFFERS"
        b_ok = "ok" if meta[pk]["bpm"] == REF_BPM[pk] else "<-- DIFFERS"
        print(f"  {pk}: key {meta[pk]['key']} [{REF_KEYS[pk]}] {k_ok};  "
              f"tempo {meta[pk]['bpm']} BPM [{REF_BPM[pk]}] {b_ok} (tempo is unreliable, unused)")
    save_json("looks.json", dict(results=results, meta=meta, table=table))
