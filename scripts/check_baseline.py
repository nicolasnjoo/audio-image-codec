"""Section 7A (MDCT perfect reconstruction) and 7B (grey baseline, piece A, raster, 30/105)."""
import numpy as np
from _common import compare, save_json
from a2i.mdct import mdct, imdct
from a2i.codec import encode_luma, decode
from a2i.channel import transmit, CONDITIONS, snr, segsnr
from a2i.testsignals import piece_a

REF_B = [30.5, 25.6, 18.1, 11.6, 7.5]

def mdct_pr(M, n=40000, seed=0):
    x = np.random.default_rng(seed).standard_normal(n)
    y = imdct(mdct(x, M), M, n)
    return 10*np.log10(np.var(x)/np.var(x - y))

if __name__ == "__main__":
    print("7A. MDCT perfect reconstruction on white noise (requirement: >= 250 dB)")
    pr = {}
    for M in (64, 128, 256):
        pr[M] = float(mdct_pr(M))
        print(f"  M = {M:<4} {pr[M]:7.1f} dB  {'ok' if pr[M] >= 250 else '<-- BELOW 250 dB'}")

    Y, side, info = encode_luma(piece_a())
    x = info["x"]
    rows, seg = [], {}
    for (name, q, s, r), ref in zip(CONDITIONS, REF_B):
        y = decode(transmit(np.round(Y), q, s, r), side)
        rows.append((name, snr(x, y), ref)); seg[name] = segsnr(x, y)
    res = compare("7B. Grey baseline, piece A, raster, A_ENV/A_RES 30/105 - audio SNR (dB)", rows)
    print("  segmental SNR (dB): " + " | ".join(f"{k}: {v:.1f}" for k, v in seg.items()))
    print(f"  clipped modules: {100*info['clip']:.2f} %")
    save_json("check_baseline.json", dict(mdct_pr=pr, baseline=res, segsnr=seg, clip=info["clip"]))
