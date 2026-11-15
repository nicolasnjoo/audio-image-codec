"""The experimental harness is the codec plus two variable constants: check that it stays so."""
import numpy as np
from a2i import codec
from a2i.channel import snr
from a2i.testsignals import piece_a
from a2i.codec import envelope_field
from explore_aesthetics import disc_layout, encode_x, decode_x, FREE

def test_disc_layout_is_a_permutation_from_rim_to_centre():
    cells = disc_layout(True)
    assert np.array_equal(np.sort(cells), np.arange(codec.NT))
    r0, c0 = divmod(int(cells[0]), codec.SIDE); r1, c1 = divmod(int(cells[-1]), codec.SIDE)
    assert max(abs(r0 - 31.5), abs(c0 - 31.5)) == 31.5          # first tile is a corner tile
    assert max(abs(r1 - 31.5), abs(c1 - 31.5)) == 0.5           # last tile touches the centre

def test_smooth_envelope_keeps_every_tile_mean():
    E = np.random.default_rng(0).uniform(-60, 60, (codec.SIDE, codec.SIDE))
    for w in (0.5, 1.0):
        S = envelope_field(E, w)
        means = S.reshape(codec.SIDE, codec.T, codec.SIDE, codec.T).mean(axis=(1, 3))
        assert np.abs(means - E).max() < 1e-9

def test_levers_off_equals_the_format():
    x = piece_a()
    Y0, _, _ = codec.encode_luma(x, cells=codec.layout("spiral_in"), a_env=60, a_res=75)
    Y1, side, info = encode_x(x)
    assert np.allclose(Y0, Y1, atol=1e-9)
    assert np.allclose(decode_x(Y1, side), codec.decode(Y0, dict(side, cells=side["cells"], grain=0.0)), atol=1e-9)

def test_candidate_roundtrip_without_channel():
    x = piece_a()
    Y, side, info = encode_x(x, **FREE)
    assert snr(info["x"], decode_x(Y, side)) >= 30.0
    assert snr(info["x"], decode_x(np.round(Y), side)) >= 30.0
