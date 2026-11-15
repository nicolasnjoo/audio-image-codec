"""Section 7C: aesthetic levers in grey, mean of pieces A/B/C.
Columns: lossless / screenshot 0.15 px / worst case. Also saves grey images of piece A
for every lever setting (out/grey_*.png); figures.py builds the contact sheets."""
import numpy as np
from PIL import Image
from _common import OUT, compare, save_json
from a2i.codec import encode_luma, decode, layout
from a2i.channel import transmit, snr
from a2i.testsignals import PIECES

KEY = [("lossless", None, False, 0.0), ("screenshot 0.15 px", 75, True, 0.15), ("worst", 60, True, 0.5)]
LAYOUTS = ["raster", "serpentine", "hilbert", "spiral_in"]
CONTRAST = [(30, 105), (45, 90), (60, 75), (75, 60), (90, 45)]
GRAIN = [0.0, 0.5, 1.0]

REF_LAYOUT = {"raster": (26.8, 16.6, 7.4), "serpentine": (26.8, 16.7, 7.5),
              "hilbert": (26.8, 16.6, 7.5), "spiral_in": (26.8, 16.7, 7.5)}
REF_CONTRAST = {(30, 105): (26.8, 16.6, 7.4), (45, 90): (28.3, 16.8, 7.4), (60, 75): (29.3, 16.6, 7.3),
                (75, 60): (29.9, 15.8, 7.0), (90, 45): (30.0, 14.4, 6.4)}
REF_CLIP = {(30, 105): 0.71, (60, 75): 0.35}
REF_GRAIN = {0.0: (29.3, 16.6, 7.3), 0.5: (30.0, 16.4, 7.2), 1.0: (30.4, 16.2, 7.2)}

_X = {pk: fn() for pk, fn in PIECES.items()}
_cache = {}

def measure(lay, a_env, a_res, grain):
    """Returns (mean SNR per KEY condition, mean clip fraction, per-piece SNR array)."""
    k = (lay, a_env, a_res, grain)
    if k not in _cache:
        per, clips = [], []
        for pk, x in _X.items():
            Y, side, info = encode_luma(x, cells=layout(lay), a_env=a_env, a_res=a_res, grain=grain)
            per.append([snr(info["x"], decode(transmit(np.round(Y), q, s, r), side)) for _, q, s, r in KEY])
            clips.append(info["clip"])
            if pk == "A":
                Image.fromarray(np.round(Y).astype(np.uint8)).save(
                    OUT / f"grey_{lay}_{a_env}_{a_res}_g{grain:g}_A.png")
        per = np.array(per)
        _cache[k] = (per.mean(0), float(np.mean(clips)), per)
    return _cache[k]

def _rows(label, m, ref):
    return [(f"{label}: {name}", m[i], ref[i]) for i, (name, *_) in enumerate(KEY)]

if __name__ == "__main__":
    out = {}
    rows = []
    for lay in LAYOUTS:
        m, _, _ = measure(lay, 30, 105, 0.0)
        rows += _rows(lay, m, REF_LAYOUT[lay])
    out["layout"] = compare("7C. Layout (30/105, grey) - audio SNR (dB), mean of 3 pieces", rows)

    rows, clip_rows = [], []
    for ae, ar in CONTRAST:
        m, clip, _ = measure("raster", ae, ar, 0.0)
        rows += _rows(f"{ae}/{ar}", m, REF_CONTRAST[(ae, ar)])
        clip_rows.append((f"{ae}/{ar}: clipped modules (%)", 100*clip, REF_CLIP.get((ae, ar))))
    out["contrast"] = compare("7C. Contrast A_ENV/A_RES (raster, grey) - audio SNR (dB), mean of 3 pieces", rows)
    out["clip"] = compare("7C. Clipped modules (%), mean of 3 pieces", clip_rows, tol=0.02, unit="%")

    rows = []
    for a in GRAIN:
        m, _, _ = measure("raster", 60, 75, a)
        rows += _rows(f"grain alpha {a:g}", m, REF_GRAIN[a])
    out["grain"] = compare("7C. Dynamic grain (60/75, raster, grey) - audio SNR (dB), mean of 3 pieces", rows)
    save_json("levers.json", out)
