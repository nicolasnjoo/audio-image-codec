import numpy as np
import pytest
from a2i.mdct import mdct, imdct, window

@pytest.mark.parametrize("M", [64, 128, 256])
def test_perfect_reconstruction(M):
    x = np.random.default_rng(0).standard_normal(40000)
    y = imdct(mdct(x, M), M, len(x))
    assert 10*np.log10(np.var(x)/np.var(x - y)) >= 250

@pytest.mark.parametrize("M", [64, 128, 256])
def test_window_is_power_complementary(M):
    w = window(M)
    assert np.allclose(w[:M]**2 + w[M:]**2, 1.0, atol=1e-14)

def test_critically_sampled():
    assert mdct(np.zeros(64*100), 64).shape[1] == 64
