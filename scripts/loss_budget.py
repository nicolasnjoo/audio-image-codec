"""Loss budget per channel stage. Section 7E is the Record look (spiral_in, 60/75) against
its reference values; the adopted default look (Droplets) is measured the same way."""
import numpy as np
from _common import compare, save_json
from a2i.codec import decode
from a2i.channel import transmit, snr, segsnr
from a2i.testsignals import PIECES
from render_options import render, DEFAULT_LOOK

STAGES = [  # name, q, screenshot, reg, Record reference A / B / C / mean
    ("float luma (no channel)",          "float", None, None, (32.5, 21.3, 37.8, 30.5)),
    ("lossless PNG",                     None, False, 0.0,    (32.5, 21.3, 37.7, 30.5)),
    ("JPEG q85 file",                    85,   False, 0.0,    (23.6, 19.7, 24.4, 22.6)),
    ("screenshot q75, perfect alignment", 75,  True,  0.0,    (19.0, 16.6, 19.5, 18.4)),
    ("screenshot q75, 0.15 px",          75,   True,  0.15,   (16.7, 15.3, 17.4, 16.5)),
    ("screenshot q75, 0.35 px",          75,   True,  0.35,   (11.1, 10.9, 11.5, 11.2)),
    ("q60 screenshot, 0.5 px (worst)",   60,   True,  0.50,   (7.0, 7.3, 7.4, 7.3)),
]

def budget(look):
    """Returns {stage: (snr A, B, C, mean)}, {stage: (segsnr ...)}, {piece: clip fraction}."""
    enc = {pk: render(fn(), look) for pk, fn in PIECES.items()}
    out, seg = {}, {}
    for name, q, s, r, _ in STAGES:
        vals, segs = [], []
        for pk, (rgb, Y, side, info, _) in enc.items():
            y = decode(Y, side) if q == "float" else decode(transmit(rgb, q, s, r), side)
            vals.append(float(snr(info["x"], y))); segs.append(segsnr(info["x"], y))
        out[name] = vals + [float(np.mean(vals))]; seg[name] = segs + [float(np.mean(segs))]
    return out, seg, {pk: e[3]["clip"] for pk, e in enc.items()}

def show(title, out, seg, clip):
    print(f"\n{title}\n  {'stage':<36} " + "       A       B       C    mean")
    for name, v in out.items():
        print(f"  {name:<36} " + " ".join(f"{a:7.2f}" for a in v))
    print("  segmental SNR (dB), A / B / C / mean:")
    for name, v in seg.items():
        print(f"  {name:<36} " + " / ".join(f"{a:5.1f}" for a in v))
    print("  clipped modules (%): " + ", ".join(f"{pk} {100*c:.2f}" for pk, c in clip.items()))

if __name__ == "__main__":
    rec, rseg, rclip = budget("Record")
    rows = [(f"{name}: {pk}", v, rf) for (name, *_, ref) in STAGES
            for pk, v, rf in zip(list(PIECES) + ["mean"], rec[name], ref)]
    table = compare("7E. Loss budget, Record look (spiral_in, 60/75) - audio SNR (dB)", rows)
    show("Record: segmental SNR and clipping (no reference values exist)", {}, rseg, rclip)

    new, nseg, nclip = budget(DEFAULT_LOOK)
    show(f"Loss budget, {DEFAULT_LOOK} look (adopted default) - audio SNR (dB)", new, nseg, nclip)
    print(f"\n{DEFAULT_LOOK} minus Record, mean of A/B/C (dB):")
    for name in new:
        print(f"  {name:<36} {new[name][3] - rec[name][3]:+6.2f}")
    save_json("loss_budget.json", dict(table=table, record=rec, record_segsnr=rseg, record_clip=rclip,
                                       default_look=DEFAULT_LOOK, default=new, default_segsnr=nseg,
                                       default_clip=nclip))
