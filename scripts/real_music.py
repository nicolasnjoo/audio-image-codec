"""Real music through the whole pipeline (roadmap step 7, the audio half).

usage:  python scripts/real_music.py NAME=file.wav [NAME=file.wav ...] [--at 0,60,120]

Each file is a PCM WAV at any sample rate, mono or stereo (decode MP3 and the like first, for
example with ffmpeg). For every file and every start time, 16.128 s are cut out, prepared
(mono, 16 kHz, peak-normalised) and sent through the simulated channel with the default look
(Droplets) and with the previous default (Record). SNR is measured against the prepared
signal, so the deliberate loss of stereo and of everything above 8 kHz is not counted.

Writes images, decoded WAVs, spectrogram figures and JSON to out/real/ (git-ignored: the
images and WAVs contain the music, so they must not be committed or published).
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))    # also works under python -I
import numpy as np
from PIL import Image
from scipy.io import wavfile
from _common import OUT, save_json
from a2i import codec
from a2i.codec import FS, NS, decode
from a2i.channel import transmit, CONDITIONS, snr, segsnr
from a2i.shapes import descriptor
from render_options import render, render_droplets, DEFAULT_LOOK
from loss_budget import STAGES
from explore_aesthetics import chroma_radius, KEY
from figures import sheet, write_wav, degradation_figure

REAL = OUT / "real"

def load(path):
    fs, a = wavfile.read(path)
    if a.dtype.kind == "i":
        a = a.astype(float)/float(2**(8*a.dtype.itemsize - 1))
    return fs, a.astype(float)

def excerpt(a, fs, t0):
    """16.128 s (plus a little, prepare() trims) from t0 seconds, prepared for the codec."""
    n = int(np.ceil(NS*fs/FS)) + fs//10
    return codec.prepare(a[int(t0*fs):int(t0*fs) + n], fs)

def stages(rgb, Y, side, x):
    out, seg = [], []
    for name, q, s, r, _ in STAGES:
        y = decode(Y, side) if q == "float" else decode(transmit(rgb, q, s, r), side)
        out.append(float(snr(x, y))); seg.append(segsnr(x, y))
    return out, seg

def key3(rgb, side, x):
    return [float(snr(x, decode(transmit(rgb, q, s, r), side))) for _, q, s, r in KEY]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="NAME=path.wav")
    ap.add_argument("--at", default="0,60,120", help="start times in seconds")
    args = ap.parse_args()
    starts = [float(t) for t in args.at.split(",")]
    REAL.mkdir(exist_ok=True); (REAL / "wav").mkdir(exist_ok=True)

    res, tiles_new, tiles_cmp = {}, [], []
    for spec in args.files:
        name, path = spec.split("=", 1)
        fs, a = load(path)
        print(f"{name}: {a.shape[0]/fs:.1f} s, {fs} Hz, {a.shape[1] if a.ndim == 2 else 1} channel(s)")
        for t0 in starts:
            if (t0 + NS/FS)*fs > a.shape[0]:
                print(f"  {t0:g} s: skipped, the file is too short"); continue
            x = excerpt(a, fs, t0); clip_id = f"{name}_{int(t0):03d}s"
            rgb, Y, side, info, ft = render(x, DEFAULT_LOOK)
            xr = info["x"]
            new, new_seg = stages(rgb, Y, side, xr)
            d = descriptor(ft)
            rgb_r, Y_r, side_r, info_r, _ = render(x, "Record")
            old, old_seg = stages(rgb_r, Y_r, side_r, info_r["x"])
            rgb9, _, side9, info9, _ = render_droplets(x, dr=9.0)    # the first format's range, 54 dB
            dr9 = key3(rgb9, side9, info9["x"])
            rgb14, _, side14, info14, _ = render_droplets(x, ripple=14.0)
            rip14 = key3(rgb14, side14, info14["x"])
            res[clip_id] = dict(
                key=ft["key"], brightness=round(ft["brightness"]), tonality=round(ft["tonality"], 2),
                weights=d["w"], drops=int(min(6, sum(w >= 8 for w in d["w"]))),
                centre_shift=side["shift"], loud_is_dark=bool(side["a_env"] < 0),
                droplets=new, droplets_segsnr=new_seg, droplets_clip=100*info["clip"],
                droplets_chroma=chroma_radius(rgb),
                record=old, record_segsnr=old_seg, record_clip=100*info_r["clip"],
                record_chroma=chroma_radius(rgb_r), envelope_54dB=dr9, ripple_14=rip14)
            Image.fromarray(rgb).save(REAL / f"{clip_id}_{DEFAULT_LOOK}.png")
            Image.fromarray(rgb_r).save(REAL / f"{clip_id}_Record.png")
            write_wav(REAL / "wav" / f"{clip_id}_0_original.wav", xr)
            decoded = []
            for i, (cname, q, s, r) in enumerate(CONDITIONS, 1):
                y = decode(transmit(rgb, q, s, r), side)
                slug = cname.replace(",", "").replace(".", "p").replace(" ", "_")
                write_wav(REAL / "wav" / f"{clip_id}_{i}_{slug}.wav", y)
                decoded.append((cname, y, snr(xr, y)))
            degradation_figure(REAL / f"{clip_id}_spectrogram.png", clip_id, xr, decoded,
                               title=f"{name}, from {t0:g} s: spectrogram after each channel condition "
                                     f"({DEFAULT_LOOK} look, 60/75)")
            tiles_new.append((f"{name} @ {t0:g} s: {new[4]:.1f} dB, {ft['key']}", rgb))
            tiles_cmp += [(f"{name} @ {t0:g} s, Record: {old[4]:.1f} dB", rgb_r),
                          (f"{name} @ {t0:g} s, {DEFAULT_LOOK}: {new[4]:.1f} dB", rgb)]
            print(f"  {t0:5g} s  key {ft['key']:<9} drops {res[clip_id]['drops']}  "
                  f"{DEFAULT_LOOK} shot 0.15 px {new[4]:5.2f} dB, worst {new[6]:5.2f} dB, clip {100*info['clip']:.2f} %"
                  f" | Record {old[4]:5.2f} / {old[6]:5.2f} dB, clip {100*info_r['clip']:.2f} %")

    ids = list(res)
    def table(title, field, names):
        print(f"\n{title}\n  {'clip':<16} " + " ".join(f"{n:>9}" for n in names))
        rows = np.array([res[c][field] for c in ids])
        for c, row in zip(ids, rows):
            print(f"  {c:<16} " + " ".join(f"{v:9.2f}" for v in row))
        print(f"  {'mean':<16} " + " ".join(f"{v:9.2f}" for v in rows.mean(0)))
        print(f"  {'minimum':<16} " + " ".join(f"{v:9.2f}" for v in rows.min(0)))
        return rows
    short = ["float", "lossless", "q85", "shot 0.00", "shot 0.15", "shot 0.35", "worst"]
    n = table(f"{DEFAULT_LOOK} (default): audio SNR (dB) per channel stage", "droplets", short)
    table(f"{DEFAULT_LOOK}: segmental SNR (dB)", "droplets_segsnr", short)
    o = table("Record (previous default): audio SNR (dB)", "record", short)
    print(f"\n{DEFAULT_LOOK} minus Record, mean over the clips (dB): " +
          ", ".join(f"{s} {v:+.2f}" for s, v in zip(short, (n - o).mean(0))))
    k3 = ["lossless", "shot 0.15", "worst"]
    base = n[:, [1, 4, 6]]
    for label, field in (("envelope range 54 dB", "envelope_54dB"), ("ripple 14", "ripple_14")):
        v = np.array([res[c][field] for c in ids])
        print(f"Option {label} minus default, mean (dB): " + ", ".join(f"{s} {d:+.2f}" for s, d in zip(k3, (v - base).mean(0)))
              + "; worst single clip at shot 0.15: " + f"{(v - base)[:, 1].min():+.2f}")
    print("Clipped modules (%), Droplets / Record: " +
          ", ".join(f"{c} {res[c]['droplets_clip']:.2f}/{res[c]['record_clip']:.2f}" for c in ids))
    print("Delivered chroma radius, Droplets / Record, mean: "
          f"{np.mean([res[c]['droplets_chroma'] for c in ids]):.1f} / {np.mean([res[c]['record_chroma'] for c in ids]):.1f}")

    ncol = max(1, len(starts))
    sheet(REAL / "sheet_real_droplets.png", tiles_new, ncol, 400,
          f"Real music, {DEFAULT_LOOK} look (dB = audio SNR after a simulated screenshot, 0.15 px)")
    sheet(REAL / "sheet_real_compare.png", tiles_cmp, 2*ncol, 300,
          f"Real music: Record (previous default) next to {DEFAULT_LOOK}")
    save_json("real_music.json", res)
