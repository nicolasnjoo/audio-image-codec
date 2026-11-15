"""The audio sample rate is a format parameter: the image always holds NS samples."""
import numpy as np
from scipy.signal import resample_poly
from a2i import codec
from a2i.channel import snr
from a2i.features import features, tile_frame, HOP
from a2i.testsignals import piece_a, piece_c
from render_options import render, DEFAULT_LOOK, RATE

def test_prepare_resamples_to_the_requested_rate():
    x = piece_a()                                               # 16 kHz, 16.128 s
    y = codec.prepare(x, 16000, 32000)
    assert y.shape == (codec.NS,) and abs(np.abs(y).max() - 1/1.02) < 1e-12
    assert np.array_equal(codec.prepare(x, 16000, 16000), codec.prepare(x))   # default unchanged
    assert codec.prepare(np.zeros(1000), 44100, 25600).shape == (codec.NS,)

def test_features_keep_their_meaning_at_another_rate():
    for piece in (piece_a, piece_c):
        x16 = codec.prepare(piece())
        x32 = resample_poly(x16, 2, 1)[:codec.NS]               # the same music, first 8.064 s, at 32 kHz
        a = features(x16[:codec.NS//2]); b = features(x32, 32000)
        assert a["hop"] == HOP and b["hop"] == 2*HOP
        assert a["key"] == b["key"]
        assert abs(a["brightness"] - b["brightness"]) < 0.05*a["brightness"]
        assert abs(a["S"].shape[1] - b["S"].shape[1]) <= 1      # same number of 16 ms frames
    assert tile_frame(1000, 2*HOP) == 63*1000//(2*HOP)

def test_default_look_at_the_adopted_rate():
    assert RATE == 32000
    rgb, Y, side, info, ft = render(piece_a(), DEFAULT_LOOK, fs_in=16000, rate=RATE)
    assert side["fs"] == RATE and info["x"].shape == (codec.NS,) and ft["fs"] == RATE
    y = codec.decode(Y, side)
    assert y.shape == (codec.NS,) and snr(info["x"], y) >= 30.0
    assert snr(info["x"], codec.decode(rgb, side)) >= 30.0
