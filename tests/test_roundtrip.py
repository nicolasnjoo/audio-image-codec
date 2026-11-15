import numpy as np
import pytest
from a2i import codec
from a2i.channel import snr
from a2i.testsignals import piece_a, PIECES

LAYOUTS = ["raster", "serpentine", "hilbert", "spiral_in", "spiral_out"]

@pytest.fixture(scope="module")
def xa():
    return piece_a()

def test_format_constants():
    assert codec.NS == 258048 and codec.NT == 4096 and codec.AC == 63
    assert codec.NT*codec.AC == codec.F*codec.M          # every coefficient is carried
    assert codec.ORIGIN == 28 and codec.G == 512 and codec.SIDE == 64

@pytest.mark.parametrize("name", LAYOUTS)
def test_layout_is_permutation(name):
    cells = codec.layout(name)
    assert cells.shape == (codec.NT,)
    assert np.array_equal(np.sort(cells), np.arange(codec.NT))

def test_spiral_in_starts_on_the_outer_edge_and_ends_in_the_centre():
    cells = codec.layout("spiral_in")
    r0, c0 = divmod(int(cells[0]), codec.SIDE); r1, c1 = divmod(int(cells[-1]), codec.SIDE)
    assert min(r0, c0, codec.SIDE - 1 - r0, codec.SIDE - 1 - c0) == 0
    assert (r1, c1) == (codec.SIDE//2 - 1, codec.SIDE//2 - 1)

def test_zigzag_covers_all_ac_slots():
    assert codec.ZZ[0] == (0, 0) and len(set(codec.ZZ)) == 64

def test_no_channel_roundtrip_piece_a_60_75(xa):
    """Float luma straight into the decoder: >= 30 dB at the recommended amplitudes."""
    for lay in ("raster", "spiral_in"):
        Y, side, info = codec.encode_luma(xa, cells=codec.layout(lay), a_env=60, a_res=75)
        assert snr(info["x"], codec.decode(Y, side)) >= 30.0

def test_lossless_rounded_grey_roundtrip_piece_a_60_75(xa):
    Y, side, info = codec.encode_luma(xa, cells=codec.layout("spiral_in"), a_env=60, a_res=75)
    assert snr(info["x"], codec.decode(np.round(Y), side)) >= 30.0

def test_stage_inverses_are_exact(xa):
    """Everything except normalize/rasterize clipping is an exact inverse pair."""
    x = codec.prepare(xa)
    X = codec.analyze(x)
    W, gains = codec.whiten(X)
    C = codec.pack(W)
    assert np.array_equal(codec.unpack(C), W)
    assert np.allclose(codec.unwhiten(W, gains), X, rtol=1e-12, atol=1e-12)
    env, R, ref = codec.normalize(C)
    assert np.abs(R).max() <= 1.0 and env.min() >= -1.0 and env.max() == 1.0
    cells = codec.layout("hilbert")
    P = codec.modulate(env, R, cells, 60, 75)
    Yun = np.full((codec.CANVAS, codec.CANVAS), 128.0)
    o, n = codec.ORIGIN, codec.INNER
    Yun[o:o+n, o:o+n] = 128 + np.kron(P, np.ones((codec.BMOD, codec.BMOD)))   # unclipped raster
    env2, R2 = codec.demodulate(Yun, cells, 60, 75)
    assert np.allclose(env2, env, atol=1e-10) and np.allclose(R2, R, atol=1e-10)
    y = codec.synthesize(X)
    assert snr(x[:-codec.M], y[:-codec.M]) >= 250      # exact, except ...
    assert snr(x, y) >= 40                             # ... the last 64 samples (frame F+1 is not kept)

def test_prepare_shape_peak_and_resampling():
    rng = np.random.default_rng(1)
    for fs, n, ch in ((16000, 1000, 1), (44100, 44100*20, 2), (48000, 48000*3, 1)):
        x = rng.standard_normal((n, ch)) if ch == 2 else rng.standard_normal(n)
        y = codec.prepare(x, fs)
        assert y.shape == (codec.NS,)
        assert abs(np.abs(y).max() - 1/1.02) < 1e-12
    assert not codec.prepare(np.zeros(100)).any()

def test_pieces_are_deterministic_and_prepared():
    for fn in PIECES.values():
        a, b = fn(), fn()
        assert np.array_equal(a, b) and a.shape == (codec.NS,)
