"""Deterministic DSP features used only for colour (never transmitted, no ML)."""
import numpy as np
from scipy.signal import stft

FS = 16000
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
HOP = 256

def features(x):
    f, _, Z = stft(x, FS, nperseg=4096, noverlap=4096-HOP)
    S = np.abs(Z)                                   # (bins, frames); frame j ~ sample j*HOP
    band = (f >= 55) & (f <= 4000)
    pc = (np.round(12*np.log2(f[band]/261.6256)) % 12).astype(int)
    chroma = np.zeros((12, S.shape[1]))
    np.add.at(chroma, pc, S[band]**2)
    cm = chroma.sum(1); cm = cm/cm.sum()
    scores = [(np.corrcoef(cm, np.roll(KK_MAJ, k))[0, 1], k, "major") for k in range(12)] + \
             [(np.corrcoef(cm, np.roll(KK_MIN, k))[0, 1], k, "minor") for k in range(12)]
    _, tonic, mode = max(scores)
    lS = np.log1p(S[(f > 30) & (f < 7500)]*50)
    flux = np.maximum(0, np.diff(lS, axis=1)).sum(0); flux = np.r_[0, flux]
    fr = FS/HOP; o = flux - flux.mean()
    ac = np.correlate(o, o, "full")[len(o)-1:]
    lags = np.arange(int(fr*60/180), int(fr*60/60)+1)
    bpm = 60*fr/lags[np.argmax(ac[lags])]           # unreliable: octave errors, unused
    sb = (f > 60) & (f < 7500)
    Sb = S[sb] + 1e-9
    centroid = (f[sb, None]*Sb).sum(0)/Sb.sum(0)
    flatness = np.exp(np.log(Sb).mean(0))/Sb.mean(0)
    rms = np.sqrt((S**2).mean(0))
    active = rms > rms.max()*0.02
    return dict(f=f, S=S, chroma=chroma, tonic=tonic, mode=mode, key=f"{NAMES[tonic]} {mode}",
                bpm=float(bpm), flux=flux/(flux.max()+1e-9), centroid=centroid,
                flatness=flatness, rms=rms,
                tonality=float(1 - np.median(flatness[active])),
                brightness=float(np.median(centroid[active])))

def tile_frame(t):
    """Tile t holds MDCT coefficients starting near sample 63*t."""
    return (63*np.asarray(t))//HOP
