"""Contact sheets, decoded WAVs (16-bit mono 16 kHz) and the spectrogram degradation figure.
WAVs and spectrograms use the adopted default look (Droplets, 60/75).
Run render_options.py first if you only want the 12 look PNGs; this script re-renders
what it needs, so it can also be run on its own."""
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import stft
from _common import OUT, save_json
from a2i.codec import encode_luma, decode, layout, FS
from a2i.channel import transmit, CONDITIONS, snr, segsnr
from a2i.testsignals import PIECES
from render_options import render, LOOKS, ALL_LOOKS, DEFAULT_LOOK

INK, PAPER = (30, 30, 30), (250, 250, 248)

def write_wav(path, x, rate=FS):
    pcm = np.round(np.clip(x, -1, 1)*32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(rate)); w.writeframes(pcm.tobytes())

def sheet(path, cells, ncols, thumb, title=None, pad=14, label_h=34):
    """cells: list of (label, image array or None). Writes a labelled contact sheet."""
    font = ImageFont.load_default(size=20); tfont = ImageFont.load_default(size=26)
    nrows = -(-len(cells)//ncols); top = 52 if title else pad
    W = pad + ncols*(thumb + pad); H = top + nrows*(thumb + label_h + pad)
    im = Image.new("RGB", (W, H), PAPER); d = ImageDraw.Draw(im)
    if title:
        d.text((pad, 12), title, fill=INK, font=tfont)
    for i, (label, a) in enumerate(cells):
        x0 = pad + (i % ncols)*(thumb + pad); y0 = top + (i//ncols)*(thumb + label_h + pad)
        d.text((x0, y0 + 4), label, fill=INK, font=font)
        if a is not None:
            t = Image.fromarray(a).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
            im.paste(t, (x0, y0 + label_h))
    im.save(path)

def grey(x, lay="raster", a_env=30, a_res=105, grain=0.0):
    Y, _, _ = encode_luma(x, cells=layout(lay), a_env=a_env, a_res=a_res, grain=grain)
    return np.round(Y).astype(np.uint8)

def spectrogram_db(x, rate=FS):
    n = int(round(512*rate/FS))                              # 32 ms window at any rate
    f, t, Z = stft(x, rate, nperseg=n, noverlap=3*n//4)
    return f, t, np.abs(Z)

def degradation_figure(path, pk, x, decoded, title=None, rate=FS):
    """decoded: list of (label, signal, snr dB). Original + 5 conditions, magma, 0 Hz - rate/2."""
    f, t, S0 = spectrogram_db(x, rate); ref = S0.max()
    panels = [(f"original (prepared: {rate/1000:g} kHz mono)", x, None)] + decoded
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, (label, y, s) in zip(axes.ravel(), panels):
        S = spectrogram_db(y, rate)[2]
        img = ax.imshow(20*np.log10(S/ref + 1e-9), origin="lower", aspect="auto", cmap="magma",
                        vmin=-80, vmax=0, extent=(t[0], t[-1], 0, f[-1]/1000), interpolation="nearest")
        ax.set_title(label if s is None else f"{label} - SNR {s:.1f} dB", fontsize=11, loc="left")
        ax.set_ylim(0, rate/2000)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=3, colors="#555555", labelsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (kHz)", fontsize=10)
    cb = fig.colorbar(img, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("magnitude (dB re. loudest bin of the original)", fontsize=10)
    cb.outline.set_visible(False)
    fig.suptitle(title or f"Piece {pk}: spectrogram after each channel condition ({DEFAULT_LOOK} look, 60/75)",
                 fontsize=13, x=0.01, ha="left")
    fig.savefig(path, dpi=110); plt.close(fig)

if __name__ == "__main__":
    X = {pk: fn() for pk, fn in PIECES.items()}

    # 1. looks: 15 PNGs + contact sheets (Droplets first: it is the default)
    order = [DEFAULT_LOOK] + list(LOOKS)
    lay = {n: (LOOKS[n][0] if n in LOOKS else "layout from the notes") for n in order}
    looks = {}
    for pk, x in X.items():
        for name in ALL_LOOKS:
            looks[name, pk] = render(x, name)
            Image.fromarray(looks[name, pk][0]).save(OUT / f"look_{name}_{pk}.png")
    keys = {pk: looks["Record", pk][4]["key"] for pk in X}
    sheet(OUT / "sheet_looks_A.png", [(f"{n} ({lay[n]})", looks[n, "A"][0]) for n in order], 3, 540,
          "Piece A, five looks, A_ENV/A_RES 60/75 (Droplets is the default)")
    sheet(OUT / "sheet_looks_all.png",
          [(f"{n} - piece {pk} ({keys[pk]})", looks[n, pk][0]) for pk in X for n in order], 5, 400,
          "Five looks x three pieces, A_ENV/A_RES 60/75 (rows: pieces A, B, C)")

    # 2. lever sheets (grey, piece A)
    sheet(OUT / "sheet_layouts.png",
          [(lay, grey(X["A"], lay)) for lay in ("raster", "serpentine", "hilbert", "spiral_in")], 4, 400,
          "Layouts, grey, piece A, A_ENV/A_RES 30/105")
    sheet(OUT / "sheet_contrast.png",
          [(f"A_ENV/A_RES {ae}/{ar}", grey(X["A"], "raster", ae, ar))
           for ae, ar in ((30, 105), (45, 90), (60, 75), (75, 60), (90, 45))], 5, 400,
          "Contrast, grey, piece A, raster layout")
    sheet(OUT / "sheet_grain.png",
          [(f"grain alpha {a:g}", grey(X["A"], "raster", 60, 75, a)) for a in (0.0, 0.5, 1.0)], 3, 400,
          "Dynamic grain, grey, piece A, raster, 60/75")

    # 3. decoded WAVs + spectrogram degradation figures
    (OUT / "wav").mkdir(exist_ok=True)
    table = {}
    for pk in X:
        rgb, Y, side, info, _ = looks[DEFAULT_LOOK, pk]
        x = info["x"]; write_wav(OUT / "wav" / f"{pk}_0_original.wav", x)
        decoded = []
        for i, (name, q, s, r) in enumerate(CONDITIONS, 1):
            y = decode(transmit(rgb, q, s, r), side)
            slug = name.replace(",", "").replace(".", "p").replace(" ", "_")
            write_wav(OUT / "wav" / f"{pk}_{i}_{slug}.wav", y)
            decoded.append((name, y, snr(x, y)))
            table[f"{pk}/{name}"] = dict(snr=float(snr(x, y)), segsnr=segsnr(x, y),
                                         clipped_samples=int((np.abs(y) > 1).sum()))
        degradation_figure(OUT / f"spectrogram_degradation_{pk}.png", pk, x, decoded)
        print(f"piece {pk}: " + " | ".join(f"{n}: {s:.1f} dB" for n, _, s in decoded))
    save_json("figures.json", table)
    print("wrote", len(list(OUT.glob("look_*.png"))), "look PNGs,", len(list(OUT.glob("sheet_*.png"))),
          "contact sheets,", len(list((OUT / "wav").glob("*.wav"))), "WAVs,",
          len(list(OUT.glob("spectrogram_*.png"))), "spectrogram figures")
