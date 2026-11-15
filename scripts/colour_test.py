"""Section 7D: the colour fix. Piece A, raster, 30/105, saturated stress chroma field
hue_cbcr(0.1 + 0.6 x + 0.2 y, radius), x and y = pixel position / canvas size."""
import numpy as np
from _common import compare, save_json
from a2i.codec import encode_luma, decode, luma, CANVAS
from a2i.looks import colorize, colorize_naive, hue_cbcr
from a2i.channel import transmit, snr
from a2i.testsignals import piece_a

KEY = [("lossless", None, False, 0.0), ("screenshot 0.15 px", 75, True, 0.15)]

def stress_field(radius):
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]/CANVAS
    return hue_cbcr(0.1 + 0.6*xx + 0.2*yy, radius)

if __name__ == "__main__":
    Y, side, info = encode_luma(piece_a())
    x = info["x"]
    cb80, cr80 = stress_field(80.0)
    cb45, cr45 = stress_field(45.0)
    variants = [
        ("grey (np.round)",                               np.round(Y),                                   (30.5, 18.1)),
        ("old: no gamut control, clip, truncate, r=80",   colorize_naive(Y, cb80, cr80),                 (12.0, 8.7)),
        ("gamut-safe only (nb=1, plain round), r=80",     colorize(Y, cb80, cr80, nb=1, exact=False),    (None, None)),
        ("gamut + 6 px neighbourhood (plain round), r=80", colorize(Y, cb80, cr80, exact=False),         (None, None)),
        ("colorize (head 16, nb 6, luma-exact), r=80",    colorize(Y, cb80, cr80),                       (None, None)),
        ("colorize (head 16, nb 6, luma-exact), r=45",    colorize(Y, cb45, cr45),                       (None, 18.0)),
    ]
    rows, extra = [], {}
    for label, img, refs in variants:
        for (cname, q, s, r), ref in zip(KEY, refs):
            rows.append((f"{label[:34]:<34} {cname[:10]}", snr(x, decode(transmit(img, q, s, r), side)), ref))
        if img.ndim == 3:
            inside = (Y >= 16) & (Y <= 239)          # pixels that are allowed to carry colour
            f = img.astype(float)[inside]
            extra[label] = dict(max_luma_err=float(np.abs(luma(img) - Y).max()),
                                out_of_headroom=float(((f < 15) | (f > 240)).any(-1).mean()))
    print("variants (label truncated in the table):")
    for label, _, _ in variants:
        print(f"  {label}")
    table = compare("7D. Colour fix, piece A, 30/105 - audio SNR (dB)", rows)
    print("\n  lossless image vs target: max luma error (grey levels); share of colourable pixels")
    print("  (16 <= Y <= 239) with a channel outside the 16-level headroom:")
    for label, e in extra.items():
        print(f"  {label:<48} max |dY| {e['max_luma_err']:6.3f}   outside {100*e['out_of_headroom']:5.2f} %")
    save_json("colour_test.json", dict(table=table, extra=extra))
