"""Section 7E: loss budget at the recommended settings (Record look: spiral_in, 60/75)."""
import numpy as np
from _common import compare, save_json
from a2i.codec import decode
from a2i.channel import transmit, snr, segsnr
from a2i.testsignals import PIECES
from render_options import render

STAGES = [  # name, q, screenshot, reg, reference A / B / C / mean
    ("float luma (no channel)",          "float", None, None, (32.5, 21.3, 37.8, 30.5)),
    ("lossless PNG",                     None, False, 0.0,    (32.5, 21.3, 37.7, 30.5)),
    ("JPEG q85 file",                    85,   False, 0.0,    (23.6, 19.7, 24.4, 22.6)),
    ("screenshot q75, perfect alignment", 75,  True,  0.0,    (19.0, 16.6, 19.5, 18.4)),
    ("screenshot q75, 0.15 px",          75,   True,  0.15,   (16.7, 15.3, 17.4, 16.5)),
    ("screenshot q75, 0.35 px",          75,   True,  0.35,   (11.1, 10.9, 11.5, 11.2)),
    ("q60 screenshot, 0.5 px (worst)",   60,   True,  0.50,   (7.0, 7.3, 7.4, 7.3)),
]

if __name__ == "__main__":
    enc = {pk: render(fn(), "Record") for pk, fn in PIECES.items()}
    rows, seg_rows, clip = [], [], {}
    for name, q, s, r, ref in STAGES:
        vals, segs = [], []
        for pk, (rgb, Y, side, info, _) in enc.items():
            y = decode(Y, side) if q == "float" else decode(transmit(rgb, q, s, r), side)
            vals.append(snr(info["x"], y)); segs.append(segsnr(info["x"], y))
            clip[pk] = info["clip"]
        for pk, v, rf in zip(list(enc) + ["mean"], vals + [np.mean(vals)], ref):
            rows.append((f"{name}: {pk}", v, rf))
        seg_rows.append((name, segs + [float(np.mean(segs))]))
    table = compare("7E. Loss budget, Record look (spiral_in, 60/75) - audio SNR (dB)", rows)
    print("\n  segmental SNR (dB), pieces A / B / C / mean (no reference values exist):")
    for name, v in seg_rows:
        print(f"  {name:<36} " + " / ".join(f"{a:5.1f}" for a in v))
    print("  clipped modules (%): " + ", ".join(f"{pk} {100*c:.2f}" for pk, c in clip.items()))
    save_json("loss_budget.json", dict(table=table, segsnr=dict(seg_rows), clip=clip))
