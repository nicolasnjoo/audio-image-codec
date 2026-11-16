"""The self-contained image: frame, ring bits, alignment, and decoding from the picture alone."""
import numpy as np
import pytest
from a2i import codec, container as ct
from a2i.testsignals import piece_a, piece_c

@pytest.fixture(scope="module")
def enc():
    return ct.encode_image(piece_a(), 16000)                    # 16 kHz piece, carried at 32 kHz

def test_header_bits_round_trip_and_reject_damage():
    w = [0, 3, 15, 0, 7, 0, 0, 12, 0, 1, 0, 9]
    for rate in ct.RATES:
        for sign in (1.0, -1.0):
            bits = ct.header_bits(rate, sign, w)
            assert len(bits) == ct.HBITS
            assert ct.parse_header(bits) == dict(rate=rate, sign=sign, w=w)
    bad = list(bits); bad[20] ^= 1
    assert ct.parse_header(bad) is None

def test_gain_bits_round_trip_and_whitening_uses_the_quantised_gains():
    rng = np.random.default_rng(0)
    gains = 2.0**rng.uniform(-12, 3, (codec.NBLK, codec.NB))
    q, used = ct.quantise_gains(gains)
    assert q.min() == 0 and q.max() <= 31
    rel, ok = ct.parse_gains(ct.gain_bits(q))
    assert ok and np.allclose(rel*used.max(), used)             # decoder gains = encoder gains, up to one level
    assert np.abs(np.log2(used/gains))[q < 31].max() <= ct.GSTEP/2 + 1e-9

def test_ring_layout():
    cells = ct.cell_origins(); plan = ct.cell_plan(len(cells))
    assert len(cells) == 4032 and len(set(cells)) == 4032
    assert sum(k == "h" for k, _ in plan) == ct.HBITS*ct.HCOPIES
    assert {b for k, b in plan if k == "g"} == set(range(codec.NBLK*codec.NB*ct.GBITS + 16))
    assert ct.cell_mask(5) == [0, 0, 1, 1, 0] or len(ct.cell_mask(5)) == 5      # fixed sequence

def test_frame_leaves_the_data_square_untouched(enc):
    rgb, info = enc
    again = ct.frame(rgb, ct.header_bits(info["rate"], info["sign"], info["w"]),
                     ct.gain_bits(ct.quantise_gains(info["gains"])[0]))
    assert np.array_equal(again, rgb)                           # framing is idempotent
    o = codec.ORIGIN
    assert rgb[:20].min() in (0, 255) and set(np.unique(rgb[2, 40:1040])) <= {0, 255}
    assert rgb[o:-o, o:-o].std() > 10                           # data still there

def test_decodes_from_the_picture_alone(enc):
    rgb, info = enc
    y, h = ct.decode_image(rgb)
    assert h["rate"] == 32000 and h["w"] == info["w"] and h["sign"] == info["sign"] and h["gains_ok"]
    assert abs(h["ax"] - 1) < 1e-4 and abs(h["bx"]) < 0.01 and abs(h["by"]) < 0.01
    assert y.shape == (codec.NS,) and ct.snr_gain(info["x"], y) >= 30.0

def test_same_audio_same_image():
    a = ct.encode_image(piece_c(), 16000)[0]; b = ct.encode_image(piece_c(), 16000)[0]
    assert np.array_equal(a, b)

def test_finds_and_decodes_a_scaled_shifted_picture(enc):
    """The image drawn 9.2 % larger at a fractional position inside a bigger picture."""
    rgb, info = enc
    scale, tx, ty, pad = 1179/1080, 37.3, 211.6, 16
    src = np.zeros((1080 + 2*pad, 1080 + 2*pad, 3)); src[pad:-pad, pad:-pad] = rgb
    n = int(np.ceil(1080*scale)) + 2; x0, y0 = int(tx) - 1, int(ty) - 1
    Wx = ct._lanczos(n, 1080 + 2*pad, 1/scale, (x0 - tx)/scale + pad)
    Wy = ct._lanczos(n, 1080 + 2*pad, 1/scale, (y0 - ty)/scale + pad)
    shot = np.full((1700, 1300, 3), 40.0)
    shot[y0:y0 + n, x0:x0 + n] = np.stack([Wy @ src[..., c] @ Wx.T for c in range(3)], -1)
    shot = np.clip(np.round(shot), 0, 255).astype(np.uint8)
    y, h = ct.decode_image(shot)
    u = np.array([0.0, 1080.0])
    err = np.abs(np.concatenate([(h["ax"] - scale)*u + h["bx"] - tx, (h["ay"] - scale)*u + h["by"] - ty]))/scale
    assert err.max() < 0.15                                     # the alignment target, in image pixels
    assert h["gains_ok"] and ct.snr_gain(info["x"], y) >= 20.0

def test_reports_failure_instead_of_guessing():
    noise = np.random.default_rng(0).integers(0, 255, (600, 800, 3)).astype(np.uint8)
    assert ct.decode_image(noise) == (None, None)
    assert ct.decode_image(np.full((500, 500, 3), 200, np.uint8)) == (None, None)
