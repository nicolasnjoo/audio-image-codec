import numpy as np
import pytest
from a2i.codec import encode_luma, luma, layout, CANVAS
from a2i.looks import colorize, colorize_naive, hue_cbcr, lerp_hue
from a2i.testsignals import piece_a

@pytest.fixture(scope="module")
def Y():
    return encode_luma(piece_a(), cells=layout("spiral_in"), a_env=60, a_res=75)[0]

def stress(radius):
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]/CANVAS
    return hue_cbcr(0.1 + 0.6*xx + 0.2*yy, radius)

@pytest.mark.parametrize("radius", [45.0, 80.0, 200.0])
def test_luma_preserved_within_0p1(Y, radius):
    rgb = colorize(Y, *stress(radius))
    assert rgb.dtype == np.uint8 and rgb.shape == (CANVAS, CANVAS, 3)
    assert np.abs(luma(rgb) - Y).max() <= 0.1

@pytest.mark.parametrize("radius", [45.0, 80.0, 200.0])
def test_never_clips(Y, radius):
    """Coloured pixels stay inside the 16-grey-level headroom (+/-1 for integer rounding);
    pixels whose luma is already outside it stay grey."""
    rgb = colorize(Y, *stress(radius)).astype(float)
    inside = (Y >= 16) & (Y <= 239)
    assert rgb[inside].min() >= 15 and rgb[inside].max() <= 240
    outside = ~inside
    assert np.abs(rgb[outside] - Y[outside][:, None]).max() <= 1.0

def test_zero_chroma_is_grey(Y):
    rgb = colorize(Y, np.zeros_like(Y), np.zeros_like(Y)).astype(float)
    assert np.abs(rgb - Y[..., None]).max() <= 1.0

def test_naive_method_does_damage_luma(Y):
    """Regression guard: the old method is kept only to show what colorize fixes."""
    assert np.abs(luma(colorize_naive(Y, *stress(80.0))) - Y).max() > 5

def test_hue_convention_and_wrap_aware_lerp():
    for h, (r, g, b) in ((0.0, (1, 0, 0)), (1/3, (0, 1, 0)), (2/3, (0, 0, 1))):
        cb, cr = hue_cbcr(h, 1.0)
        d = np.array([1.402*cr, -0.344136*cb - 0.714136*cr, 1.772*cb])
        assert np.argmax(d) == np.argmax([r, g, b])
    assert abs(lerp_hue(0.95, 0.05, 0.5) % 1.0) < 1e-12        # goes through red, not through cyan
    assert abs(lerp_hue(0.2, 0.4, 0.5) - 0.3) < 1e-12
