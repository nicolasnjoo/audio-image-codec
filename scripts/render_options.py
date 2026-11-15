"""Render the four looks for the three pieces and measure them (section 7F, 7G)."""
import json, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from _common import OUT, compare, save_json
from a2i.codec import encode_luma, decode, layout, CANVAS, INNER, NT
from a2i.looks import colorize, hue_cbcr, tiles_to_canvas, lerp_hue
from a2i.features import features, tile_frame
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
    j = np.clip(tile_frame(np.arange(NT)), 0, ft["S"].shape[1] - 1)
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

LOOKS = {
    "Record":  ("spiral_in", look_record),
    "Mandala": ("spiral_in", look_mandala),
    "Duotone": ("hilbert",   look_duotone),
    "Poster":  ("hilbert",   look_poster),
}

def render(x, name, a_env=A_ENV, a_res=A_RES):
    """Encode audio x with look `name`. Returns rgb (uint8), float luma Y, side, info, features."""
    lay, chroma_fn = LOOKS[name]
    cells = layout(lay)
    Y, side, info = encode_luma(x, cells=cells, a_env=a_env, a_res=a_res)
    ft = features(info["x"]); pal = palette(ft)
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
        for name in LOOKS:
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
    for name, (rc, rg, rd) in REF_F.items():
        c = np.array([results[f"{name}/{pk}"]["colour"] for pk in PIECES])   # (piece, condition)
        g = np.array([results[f"{name}/{pk}"]["grey"] for pk in PIECES])
        rows += [(f"{name}: colour, screenshot 0.15 px", c[:, 1].mean(), rc),
                 (f"{name}: grey, screenshot 0.15 px", g[:, 1].mean(), rg),
                 (f"{name}: cost (colour - grey)", c[:, 1].mean() - g[:, 1].mean(), rd),
                 (f"{name}: colour, lossless", c[:, 0].mean(), 30.5),
                 (f"{name}: grey, lossless", g[:, 0].mean(), None),
                 (f"{name}: colour, worst", c[:, 2].mean(), 7.3)]
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
