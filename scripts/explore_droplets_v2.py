"""EXPERIMENTAL - a more stable droplet shape (prototype, not the format).

Two changes to how the 12 pitch-class weights are derived; the landscape itself is unchanged:

  fixed clock   C is always at 12 o'clock. Today the clock starts at the detected key note,
                so the same notes give a rotated shape whenever the key detector changes its
                mind (E minor and G major are the same notes, a quarter turn apart).
  stand-out     weight = how far a note rises above the middle of the mix: (energy - median)
                / (strongest - median), 0 below the median. Today it is sqrt(energy/strongest),
                which is 0.45..1 for every note of a dense mix, so six drops nearly always.
                A drop is drawn for weight >= 0.2 (3 of 15).

usage:  python scripts/explore_droplets_v2.py NAME=file.wav [...] [--at 0,60,120]
        (without files: the three synthetic pieces)
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))    # also works under python -I
import numpy as np
from _common import OUT, save_json
from a2i import codec, shapes
from a2i.codec import decode
from a2i.channel import transmit, snr
from a2i.features import NAMES
from a2i.testsignals import PIECES
from render_options import render_droplets
from figures import sheet

FLOOR_V2 = 3                                                 # of 15: weight 0.2

def standout_weights(ft):
    """12 integers 0..15 in fixed pitch order (C first): how far each note stands out."""
    cm = ft["chroma"].sum(1)
    med = np.median(cm)
    return np.round(15*np.clip((cm - med)/(cm.max() - med + 1e-30), 0, 1)).astype(int).tolist()

shapes.LANDSCAPES["droplets_v2"] = lambda w: shapes.droplets(w, floor=FLOOR_V2)

def drops_now(ft):
    w = np.array(shapes.descriptor(ft)["w"])
    ks = [int(k) for k in np.argsort(-w, kind="stable")[:6] if w[k] >= 8]
    return [NAMES[(k + ft["tonic"]) % 12] for k in ks]       # back to absolute note names

def drops_v2(ft):
    w = np.array(standout_weights(ft))
    return [NAMES[int(k)] for k in np.argsort(-w, kind="stable")[:6] if w[k] >= FLOOR_V2]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="NAME=path.wav")
    ap.add_argument("--at", default="0,60,120")
    args = ap.parse_args()
    starts = [float(t) for t in args.at.split(",")]
    groups = {}
    if args.files:
        from real_music import load, excerpt
        for spec in args.files:
            name, path = spec.split("=", 1)
            fs, a = load(path)
            groups[name] = {f"{name} @ {t0:g} s": excerpt(a, fs, t0) for t0 in starts}
        out_dir, tag = OUT / "real", "real"
    else:
        groups = {f"piece {pk}": {f"piece {pk}": fn()} for pk, fn in PIECES.items()}
        out_dir, tag = OUT, "synthetic"
    out_dir.mkdir(exist_ok=True)

    res, cells = {}, []
    print(f"{'clip':<18} {'key':<10} now: drops (clock starts at the key note)      prototype: drops (C at 12 o'clock)       "
          "SNR after screenshot now / prototype (dB)")
    for gname, clips in groups.items():
        row_now, row_new = [], []
        for cid, x in clips.items():
            a = render_droplets(x)
            b = render_droplets(x, shape="droplets_v2", weights=standout_weights)
            ft = a[4]
            s = [float(snr(r[3]["x"], decode(transmit(r[0], 75, True, 0.15), r[2]))) for r in (a, b)]
            dn, dv = drops_now(ft), drops_v2(ft)
            res[cid] = dict(key=ft["key"], now=dn, prototype=dv, snr_now=s[0], snr_prototype=s[1],
                            weights_prototype=standout_weights(ft))
            print(f"{cid:<18} {ft['key']:<10} {len(dn)}: {' '.join(dn):<34} {len(dv)}: {' '.join(dv):<34} {s[0]:.2f} / {s[1]:.2f}")
            row_now.append((f"{cid}, now: {len(dn)} drops", a[0]))
            row_new.append((f"{cid}, new: {' '.join(dv)}", b[0]))
        cells += row_now + row_new
    n = np.array([[r["snr_now"], r["snr_prototype"]] for r in res.values()])
    print(f"mean SNR after screenshot: now {n[:, 0].mean():.2f} dB, prototype {n[:, 1].mean():.2f} dB;"
          f" drops per clip: now {np.mean([len(r['now']) for r in res.values()]):.1f}, prototype"
          f" {np.mean([len(r['prototype']) for r in res.values()]):.1f}")
    ncol = 2*len(next(iter(groups.values())))
    sheet(out_dir / f"droplets_v2_{tag}.png", cells, ncol, 300,
          "Droplets: today (left half) and the prototype (right half): fixed clock, drops only for notes that stand out")
    save_json(f"droplets_v2_{tag}.json", res)
