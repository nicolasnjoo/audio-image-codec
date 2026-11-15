"""EXPERIMENTAL - feasibility of a second, finer layer (not built, not the format).

Today every 2x2 px module carries one number (its mean). The four pixels of a module could
carry three more numbers as patterns with zero mean, which the present decoder does not see.
That would quadruple the capacity (for example 32 kHz stereo for the full 16 s), but such
detail does not survive a rescale. So it would be a second tier: present when the image file
itself is shared, gone after a screenshot, where the decoder falls back to today's audio.

This script only measures feasibility, with stand-in data in the fine layer (values drawn
from the real coefficients, so the statistics match) and grey images:
  (a) what the fine layer costs the base layer, and (b) how well the fine layer itself survives.

usage:  python scripts/explore_fine_layer.py NAME=file.wav [...] [--at 60]
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))    # also works under python -I
import numpy as np
from _common import save_json
from a2i import codec, shapes
from a2i.codec import ORIGIN, INNER, G, decode
from a2i.channel import transmit, snr
from real_music import load, excerpt
from render_options import render, DEFAULT_LOOK, ENV_RANGE

SL = slice(ORIGIN, ORIGIN + INNER)
PATTERNS = [np.array([[1, -1], [1, -1]]), np.array([[1, 1], [-1, -1]]), np.array([[1, -1], [-1, 1]])]
CHANNELS = [("lossless PNG", None, False, 0.0), ("JPEG q95 file", 95, False, 0.0), ("JPEG q90 file", 90, False, 0.0),
            ("JPEG q85 file", 85, False, 0.0), ("screenshot 0.15 px", 75, True, 0.15), ("worst case", 60, True, 0.5)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="NAME=path.wav")
    ap.add_argument("--at", default="60")
    args = ap.parse_args()
    clips = {}
    for spec in args.files:
        name, path = spec.split("=", 1)
        fs, a = load(path)
        for t0 in (float(t) for t in args.at.split(",")):
            clips[f"{name}@{t0:g}"] = excerpt(a, fs, t0)
    print(f"{len(clips)} excerpts, grey, mean. SNR in dB.")
    print("fine amplitude | base layer: " + " / ".join(c[0] for c in CHANNELS) +
          " | fine layer: lossless / q95 / q90 / q85 / screenshot | clipped px %")
    out = {}
    for amp in (0, 15, 30, 45):
        base, fine, clipped = [], [], []
        for x in clips.values():
            rgb, Y, side, info, _ = render(x, DEFAULT_LOOK)
            R = codec.normalize(codec.pack(codec.whiten(codec.analyze(info["x"]))[0]), ENV_RANGE)[1].ravel()
            planes = np.random.default_rng(1).choice(R, size=(3, G, G))      # stand-in data, real statistics
            under = shapes.build(**side["shape"])[1]                         # the ripple, known to the decoder
            Y2 = Y.copy(); Y2[SL, SL] += amp*sum(np.kron(p, h) for p, h in zip(planes, PATTERNS))
            clipped.append(100*((Y2 < 0) | (Y2 > 255)).mean()); Y2 = np.round(np.clip(Y2, 0, 255))
            b, f = [], []
            for _, q, s, r in CHANNELS:
                L = transmit(Y2, q, s, r)
                b.append(float(snr(info["x"], decode(L, side))))
                if amp:
                    blk = (L - under)[SL, SL].reshape(G, 2, G, 2)
                    est = np.stack([(blk*h[None, :, None, :]).sum(axis=(1, 3))/4 for h in PATTERNS])/amp
                    f.append(float(10*np.log10(planes.var()/np.mean((planes - est)**2))))
            base.append(b); fine.append(f)
        b = np.mean(base, 0); out[amp] = dict(base=b.tolist(), clipped=float(np.mean(clipped)))
        line = f"  {amp:3d} grey levels | " + " / ".join(f"{v:5.2f}" for v in b)
        if amp:
            f = np.mean(fine, 0); out[amp]["fine"] = f.tolist()
            line += " | " + " / ".join(f"{v:5.1f}" for v in f[:5])
        print(line + f" | {np.mean(clipped):.2f}")
    save_json("fine_layer.json", out)
