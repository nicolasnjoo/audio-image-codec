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

# ---------------------------------------------------------------- format additions
def _unclipped(P, under=None):
    Y = np.full((codec.CANVAS, codec.CANVAS), 128.0)
    o, n = codec.ORIGIN, codec.INNER
    Y[o:o+n, o:o+n] = 128 + np.kron(P, np.ones((codec.BMOD, codec.BMOD)))
    if under is not None:
        Y[o:o+n, o:o+n] += under[o:o+n, o:o+n]
    return Y

def test_defaults_still_produce_the_first_format(xa):
    Y, side, info = codec.encode_luma(xa, cells=codec.layout("spiral_in"), a_env=60, a_res=75)
    assert side["shift"] == 0.0 and side["smooth"] == 0.0 and side["under"] is None
    env, R, _ = codec.normalize(codec.pack(codec.whiten(codec.analyze(info["x"]))[0]))
    tiles = Y[28:1052, 28:1052].reshape(64, 16, 64, 16)
    cells = codec.layout("spiral_in")
    means = tiles.mean(axis=(1, 3)).reshape(-1)[cells]
    unclipped = ((tiles.min(axis=(1, 3)) > 0) & (tiles.max(axis=(1, 3)) < 255)).reshape(-1)[cells]
    assert unclipped.mean() > 0.9
    assert np.abs(means - (128 + 60*env))[unclipped].max() < 1e-9   # uncentred: mean = 128 + A_ENV*env

def test_smooth_envelope_and_ripple_are_exact_inverses(xa):
    x = codec.prepare(xa)
    env, R, ref = codec.normalize(codec.pack(codec.whiten(codec.analyze(x))[0]))
    cells = codec.layout("hilbert")
    under = 8*np.cos(np.add.outer(np.arange(codec.CANVAS), np.arange(codec.CANVAS))/9.0)
    for a_env in (60.0, -60.0):                                 # loud = bright and loud = dark
        P = codec.modulate(env, R, cells, a_env, 75, shift=-21.0, smooth=1.0)
        env2, R2 = codec.demodulate(_unclipped(P, under), cells, a_env, 75, shift=-21.0, smooth=1.0, under=under)
        assert np.allclose(env2, env, atol=1e-9) and np.allclose(R2, R, atol=1e-9)

def test_smooth_envelope_keeps_every_tile_mean():
    E = np.random.default_rng(0).uniform(-60, 60, (codec.SIDE, codec.SIDE))
    for w in (0.5, 1.0):
        means = codec.envelope_field(E, w).reshape(codec.SIDE, codec.T, codec.SIDE, codec.T).mean(axis=(1, 3))
        assert np.abs(means - E).max() < 1e-9
    assert np.array_equal(codec.envelope_field(E, 0), np.kron(E, np.ones((codec.T, codec.T))))

def test_centre_puts_the_median_tile_on_the_requested_luma(xa):
    for centre, a_env in ((128, 60.0), (165, -60.0)):           # a bright centre goes with loud = dark
        Y, side, info = codec.encode_luma(xa, cells=codec.layout("raster"), a_env=a_env, a_res=75,
                                          centre=centre, smooth=1.0)
        assert side["shift"] == round(side["shift"])            # one integer for the header
        assert abs(128 + a_env*np.median(info["env"]) + side["shift"] - centre) <= 0.5
        assert snr(info["x"], codec.decode(Y, side)) >= 30.0
