"""Deterministic DSP features used only for colour (never transmitted, no ML)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft

FS = 16000
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
HOP = 256

def features(x, fs=FS):
    """fs: sample rate of x. Window (256 ms) and hop (16 ms) are kept in time, and all band
    limits are in Hz, so the features of the same music do not depend on the rate."""
    hop = int(round(HOP*fs/FS))
    f, _, Z = stft(x, fs, nperseg=16*hop, noverlap=15*hop)
    S = np.abs(Z)                                   # (bins, frames); frame j ~ sample j*hop
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
    fr = fs/hop; o = flux - flux.mean()
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
                flatness=flatness, rms=rms, hop=hop, fs=fs,
                tonality=float(1 - np.median(flatness[active])),
                brightness=float(np.median(centroid[active])))

def tonal_centroid(ft, rel, sigma_frames=10):
    """Per STFT frame: where the sounding notes sit on the circle of fifths, as an angle in
    turns relative to the major triad on pitch class `rel` (0 = that triad), and how
    concentrated they are (0 = noise, ~0.64 = a clean triad, 1 = a single note)."""
    ch = gaussian_filter1d(ft["chroma"], sigma_frames, axis=1, mode="nearest")
    ang = 2*np.pi*((7*(np.arange(12) - rel)) % 12)/12
    z = (ch*np.exp(1j*ang)[:, None]).sum(0)/(ch.sum(0) + 1e-12)
    return (np.angle(z) - np.pi/4)/(2*np.pi), np.abs(z)

def tile_frame(t, hop=HOP):
    """Tile t holds MDCT coefficients starting near sample 63*t; hop = STFT hop in samples."""
    return (63*np.asarray(t))//hop
