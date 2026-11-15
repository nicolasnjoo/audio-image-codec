"""Three deterministic synthetic pieces with different keys, tempi and timbres."""
import numpy as np
FS, NS = 16000, 258048

def _note(t, f, amp, n_h=8, roll=1.4, rng=None):
    ph = rng.uniform(0, 6.3, n_h) if rng is not None else np.zeros(n_h)
    return sum(amp/h**roll*np.sin(2*np.pi*f*h*t + ph[h-1])
               for h in range(1, n_h+1) if np.max(f)*h < 7800)

def piece_a():
    """A-major-ish progression (A, G, F, E), kick every 250 ms, offbeat hats."""
    rng = np.random.default_rng(3); t = np.arange(NS)/FS; x = np.zeros(NS)
    for i, ch in enumerate([(220,277.2,329.6),(196,246.9,293.7),(174.6,220,261.6),(164.8,207.7,246.9)]):
        sl = slice(i*(NS//4), (i+1)*(NS//4)); tt = t[sl]-t[sl][0]; e = np.exp(-tt*0.35)
        for f in ch:
            x[sl] += e*_note(tt, f, 0.6, rng=rng)
    for k in range(0, NS, FS//4):
        n = min(2000, NS-k); tt = np.arange(n)/FS
        x[k:k+n] += 1.4*np.exp(-tt*28)*np.sin(2*np.pi*58*np.exp(-tt*9)*tt)
    for k in range(FS//8, NS, FS//4):
        n = min(900, NS-k); tt = np.arange(n)/FS
        x[k:k+n] += 0.30*np.exp(-tt*70)*rng.standard_normal(n)
    return x/(np.abs(x).max()*1.02)

def piece_b():
    """D-minor pad and bass, no drums, slow swells, dark timbre."""
    rng = np.random.default_rng(5); t = np.arange(NS)/FS; x = np.zeros(NS)
    prog = [(146.8,174.6,220.0),(116.5,146.8,174.6),(130.8,164.8,196.0),(110.0,138.6,164.8)]
    seg = NS//4
    for i, ch in enumerate(prog):
        sl = slice(i*seg, (i+1)*seg); tt = t[sl]-t[sl][0]
        sw = np.sin(np.pi*np.clip(tt/(seg/FS), 0, 1))**0.7
        for f in ch:
            x[sl] += sw*_note(tt, f*(1+0.002*np.sin(2*np.pi*0.3*tt)), 0.5, n_h=6, roll=2.2, rng=rng)
        x[sl] += 0.8*sw*np.sin(2*np.pi*ch[0]/2*tt)
    return x/(np.abs(x).max()*1.02)

def piece_c():
    """E-minor pentatonic arpeggio at 132 BPM, bright pluck, busy hats, claps, kick."""
    rng = np.random.default_rng(9); t = np.arange(NS)/FS; x = np.zeros(NS)
    notes = [329.6, 392.0, 440.0, 493.9, 587.3, 659.3, 587.3, 493.9]
    step = int(FS*60/132/4)
    for j, k in enumerate(range(0, NS, step)):
        n = min(step*2, NS-k); tt = np.arange(n)/FS
        x[k:k+n] += np.exp(-tt*9)*_note(tt, notes[j % 8], 0.35, n_h=10, roll=0.9, rng=rng)
        hn = min(500, NS-k); th = np.arange(hn)/FS
        x[k:k+hn] += 0.22*np.exp(-th*90)*rng.standard_normal(hn)
        if j % 8 == 4:
            cn = min(2500, NS-k); tc = np.arange(cn)/FS
            x[k:k+cn] += 0.5*np.exp(-tc*25)*rng.standard_normal(cn)
        if j % 4 == 0:
            kn = min(2400, NS-k); tk = np.arange(kn)/FS
            x[k:k+kn] += 1.1*np.exp(-tk*25)*np.sin(2*np.pi*52*np.exp(-tk*7)*tk)
    return x/(np.abs(x).max()*1.02)

PIECES = {"A": piece_a, "B": piece_b, "C": piece_c}
