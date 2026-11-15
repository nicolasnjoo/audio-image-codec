"""Music-generated layouts: always a permutation, deterministic, and exactly decodable."""
import hashlib, io
import numpy as np
import pytest
from PIL import Image
from a2i import codec
from a2i.channel import snr
from a2i.features import features
from a2i.testsignals import PIECES, piece_a
from explore_aesthetics import decode_x
from shapes import SHAPES, descriptor, layout_from, plate_modes, render_shape

@pytest.fixture(scope="module")
def descriptors():
    return {pk: descriptor(features(codec.prepare(fn()))) for pk, fn in PIECES.items()}

def test_descriptor_is_fourteen_4bit_numbers(descriptors):
    for d in descriptors.values():
        vals = d["w"] + [d["bright"], d["tonal"]]
        assert len(vals) == 14 and all(isinstance(v, int) and 0 <= v <= 15 for v in vals)
        assert max(d["w"]) == 15

@pytest.mark.parametrize("shape", list(SHAPES))
def test_every_shape_is_a_permutation(shape, descriptors):
    for d in descriptors.values():
        cells, keys = layout_from(SHAPES[shape](d))
        assert np.array_equal(np.sort(cells), np.arange(codec.NT))
        assert np.all(np.diff(keys) >= 0)

def test_shape_depends_on_the_notes(descriptors):
    a = layout_from(SHAPES["flower"](descriptors["A"]))[0]
    c = layout_from(SHAPES["flower"](descriptors["C"]))[0]
    assert not np.array_equal(a, c)

def test_plate_modes_follow_the_membrane_frequencies():
    modes = plate_modes()
    assert len(modes) == 12 and modes[0] == (1, 1) and modes[7] == (3, 3)   # unison, fifth
    assert all(1 <= m <= n <= 8 for m, n in modes)

def test_same_audio_same_image_and_exact_decode():
    x = piece_a()
    a = render_shape(x, "flower", ripple=14.0)
    b = render_shape(x, "flower", ripple=14.0)
    png = []
    for rgb in (a[0], b[0]):
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG"); png.append(buf.getvalue())
    assert png[0] == png[1] and hashlib.sha256(png[0]).digest() == hashlib.sha256(png[1]).digest()
    _, Y, side, info, _ = a
    assert snr(info["x"], decode_x(Y, side)) >= 30.0          # ripple is subtracted exactly
