"""EXPERIMENTAL - what better audio quality would cost (listening examples, not the format).

The image holds 258 048 numbers, one per MDCT coefficient. That count is fixed by the smallest
module that survives a phone screenshot, so better audio has to come from one of two places:

  real      a higher sample rate or stereo, paid for with a shorter clip
              16 kHz mono 16.1 s (now) | 24 kHz mono 10.8 s | 32 kHz mono 8.1 s | 16 kHz stereo 8.1 s
  rebuilt   the decoder recreates what was not sent from what was sent, steered by a few
            hundred small numbers. Here: the 8-16 kHz band is rebuilt from the 3-7 kHz band and
            shaped by 512 loudness values (8 bands x 64 time blocks). Plausible, not exact.

usage:  python scripts/explore_quality.py NAME=file.wav [...] [--at 60]
Writes WAVs to out/quality/ (git-ignored: they contain the music).
"""
import argparse, pathlib, sys, wave
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))    # also works under python -I
from math import gcd
import numpy as np
from scipy.signal import resample_poly
from _common import OUT, save_json
from a2i.codec import NS, decode
from a2i.channel import transmit, snr
from a2i.mdct import mdct, imdct
from render_options import render, DEFAULT_LOOK
from real_music import load

QDIR = OUT / "quality"
MB, HB0, NBAND, BLK = 128, 64, 8, 63        # 32 kHz MDCT: 125 Hz bins, 4 ms hop; rebuild from bin 64 (8 kHz)
SRC0 = 24                                   # the rebuilt band is copied from bins 24..55 (3-7 kHz)

def resample(x, fs, rate):
    g = gcd(int(fs), int(rate))
    return x if fs == rate else resample_poly(x, rate//g, fs//g, axis=0)

def write_wav(path, x, rate, gain=1.0):
    x = np.clip(np.asarray(x)*gain, -1, 1)
    pcm = np.round(x*32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1]); w.setsampwidth(2); w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())

def through(x, rate):
    """The whole pipeline (Droplets look) for NS samples of mono audio at `rate`.
    Returns the prepared reference, the decode from the lossless image, and the decode after
    the simulated screenshot (q75, 0.15 px)."""
    rgb, Y, side, info, _ = render(x, DEFAULT_LOOK, fs_in=rate, rate=rate)
    return info["x"], decode(rgb, side), decode(transmit(rgb, 75, True, 0.15), side)

# ---------------------------------------------------------------- rebuilt highs
def band_energy(X):
    nb = X.shape[0]//BLK
    return (X[:nb*BLK, HB0:].reshape(nb, BLK, NBAND, (MB - HB0)//NBAND)**2).mean(axis=(1, 3))

def highs_envelope(x32):
    """What the encoder would send: 64 x 8 = 512 loudness values for 8-16 kHz."""
    return band_energy(mdct(x32, MB))

def rebuild_highs(y16, env):
    """Decoder side: 16 kHz audio -> 32 kHz audio whose 8-16 kHz band is a copy of its own
    3-7 kHz band, scaled per band and time block to the loudness the encoder measured."""
    X = mdct(resample_poly(y16, 2, 1), MB)
    X[:, HB0:] = 0
    X[:, 64:96] = X[:, SRC0:SRC0 + 32]; X[:, 96:128] = X[:, SRC0:SRC0 + 32]
    g = np.minimum(np.sqrt(env/(band_energy(X) + 1e-20)), 10.0)        # at most +20 dB
    n = g.shape[0]*BLK
    X[:n, HB0:] *= np.repeat(np.repeat(g, BLK, 0), (MB - HB0)//NBAND, 1)
    X[n:, HB0:] = 0
    return imdct(X, MB, 2*len(y16))

def fine_error_db(y, ref):
    """Median |level error| of the rebuilt band on a finer grid than the 512 values
    (500 Hz x 32 ms cells), over cells that are not near-silent."""
    def cells(x):
        X = mdct(x, MB); n = X.shape[0]//8*8
        return (X[:n, HB0:].reshape(n//8, 8, 16, 4)**2).mean(axis=(1, 3))
    a, b = cells(y), cells(ref)
    ok = b > b.max()*1e-5
    return float(np.median(np.abs(10*np.log10((a[ok] + 1e-20)/(b[ok] + 1e-20)))))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="NAME=path.wav")
    ap.add_argument("--at", default="60", help="start time in seconds")
    args = ap.parse_args()
    t0 = float(args.at); QDIR.mkdir(exist_ok=True)
    res = {}
    for spec in args.files:
        name, path = spec.split("=", 1)
        fs, a = load(path)
        a = a if a.ndim == 2 else np.stack([a, a], 1)
        src = a[int(t0*fs):int(t0*fs) + int(np.ceil((NS/16000 + 0.2)*fs))]       # a little over 16.128 s
        mono = src.mean(1)
        tag = f"{name}_{int(t0):03d}s"; r = res[tag] = {}

        # what the source contains that 16 kHz mono cannot carry
        m32 = resample(mono, fs, 32000); X = mdct(m32, MB)
        s32 = resample((src[:, 0] - src[:, 1])/2, fs, 32000)
        r["energy_above_8k_dB"] = float(10*np.log10((X[:, HB0:]**2).sum()/(X**2).sum()))
        r["side_vs_mid_dB"] = float(10*np.log10((s32**2).sum()/(m32**2).sum()))

        g0 = 0.97/np.abs(src).max()
        write_wav(QDIR / f"{tag}_0_source_stereo.wav", src[:int(NS/16000*fs)], fs, g0)

        def save(label, rate, ref, lossless, shot, seconds):
            gain = 0.97/np.abs(ref).max()
            write_wav(QDIR / f"{tag}_{label}_clean.wav", ref, rate, gain)
            write_wav(QDIR / f"{tag}_{label}_screenshot.wav", shot, rate, gain)
            r[label] = dict(rate=rate, seconds=seconds, snr_lossless=float(snr(ref.ravel(), lossless.ravel())),
                            snr_screenshot=float(snr(ref.ravel(), shot.ravel())))
            print(f"  {label:<34} {seconds:5.1f} s   lossless {r[label]['snr_lossless']:5.1f} dB   after screenshot {r[label]['snr_screenshot']:5.2f} dB")

        print(f"{tag}: {r['energy_above_8k_dB']:.1f} dB of the energy is above 8 kHz; the left-right difference is {r['side_vs_mid_dB']:.1f} dB relative to the sum")
        # 1-3: mono at three sample rates (same image, shorter clip)
        dec16 = None
        for label, rate in (("1_now_16k_mono", 16000), ("2_real_24k_mono", 24000), ("2b_real_25k6_mono", 25600),
                            ("3_real_32k_mono", 32000)):
            ref, lo, sh = through(resample(mono, fs, rate)[:NS], rate)
            save(label, rate, ref, lo, sh, NS/rate)
            if rate == 16000:
                ref16, dec16 = ref, (lo, sh)

        # 4: true stereo at 16 kHz: sum and difference, half the time each
        L, R = (resample(src[:, c], fs, 16000)[:NS//2] for c in (0, 1))
        ref, lo, sh = through(np.concatenate([(L + R)/2, (L - R)/2]), 16000)
        st = lambda y: np.stack([y[:NS//2] + y[NS//2:], y[:NS//2] - y[NS//2:]], 1)
        save("4_real_16k_stereo", 16000, st(ref), st(lo), st(sh), NS/2/16000)

        # 5: rebuilt highs on top of today's 16 kHz mono
        x16 = resample(mono, fs, 16000)[:NS]
        scale = ref16.max()/x16.max() if x16.max() > 0 else 1.0          # the codec's own normalisation
        true32 = resample(mono, fs, 32000)[:2*NS]*scale
        env = highs_envelope(true32)
        hi_lo, hi_sh = rebuild_highs(dec16[0], env), rebuild_highs(dec16[1], env)
        gain = 0.97/np.abs(true32).max()
        write_wav(QDIR / f"{tag}_5_rebuilt_highs_32k_mono_target.wav", true32, 32000, gain)
        write_wav(QDIR / f"{tag}_5_rebuilt_highs_32k_mono_clean.wav", hi_lo, 32000, gain)
        write_wav(QDIR / f"{tag}_5_rebuilt_highs_32k_mono_screenshot.wav", hi_sh, 32000, gain)
        r["5_rebuilt_highs"] = dict(values=int(env.size), fine_error_dB_lossless=fine_error_db(hi_lo, true32),
                                    fine_error_dB_screenshot=fine_error_db(hi_sh, true32))
        print(f"  {'5_rebuilt_highs_32k_mono':<34} {NS/16000:5.1f} s   {env.size} extra values; level error of the rebuilt band on a "
              f"finer grid: median {r['5_rebuilt_highs']['fine_error_dB_lossless']:.1f} dB (lossless), "
              f"{r['5_rebuilt_highs']['fine_error_dB_screenshot']:.1f} dB (after screenshot)")
    save_json("quality.json", res)
