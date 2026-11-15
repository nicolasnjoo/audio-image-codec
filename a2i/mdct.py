import numpy as np

def _basis(M):
    n = np.arange(2*M)[None, :]
    k = np.arange(M)[:, None]
    return np.cos(np.pi/M * (n + 0.5 + M/2) * (k + 0.5))   # (M, 2M)

def window(M):
    n = np.arange(2*M)
    return np.sin(np.pi/(2*M) * (n + 0.5))

def mdct(x, M):
    """x: 1-D. Returns (frames, M). Pads with M zeros in front and enough behind."""
    x = np.concatenate([np.zeros(M), x, np.zeros(2*M - (len(x) % M) + M)])
    nf = (len(x) - M) // M
    C, w = _basis(M), window(M)
    frames = np.stack([x[i*M:i*M+2*M] * w for i in range(nf)])
    return frames @ C.T

def imdct(X, M, n_out=None):
    C, w = _basis(M), window(M)
    y = np.zeros((X.shape[0] + 1) * M + M)
    rec = (X @ C) * (2.0 / M) * w
    for i in range(X.shape[0]):
        y[i*M:i*M+2*M] += rec[i]
    y = y[M:]
    return y[:n_out] if n_out else y
