"""Layouts generated from the music: always a permutation, deterministic, exactly decodable,
and rebuildable by the decoder from header-sized fields alone."""
import hashlib, io
import numpy as np
import pytest
from PIL import Image
from a2i import codec
from a2i.channel import snr
from a2i.features import features
from a2i.shapes import RIPPLE, build, descriptor, layout_from
from a2i.testsignals import PIECES, piece_a
from explore_shapes import SHAPES, plate_modes, render_shape
from render_options import render, DEFAULT_LOOK

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
    a = build("droplets", descriptors["A"]["w"])[0]
    c = build("droplets", descriptors["C"]["w"])[0]
    assert not np.array_equal(a, c)

def test_ripple_has_the_adopted_amplitude(descriptors):
    cells, under, tmap = build("droplets", descriptors["A"]["w"])
    assert RIPPLE == 8.0 and abs(np.abs(under).max() - RIPPLE) < 0.01
    assert under.shape == tmap.shape == (codec.CANVAS, codec.CANVAS)
    assert tmap.min() >= 0 and tmap.max() <= codec.NT - 1

def test_plate_modes_follow_the_membrane_frequencies():
    modes = plate_modes()
    assert len(modes) == 12 and modes[0] == (1, 1) and modes[7] == (3, 3)   # unison, fifth
    assert all(1 <= m <= n <= 8 for m, n in modes)

def test_default_look_decodes_from_header_sized_fields_only():
    """The side dict of the default look carries no layout array and no ripple image: only the
    shape name, 12 weights and the ripple amplitude. decode() must rebuild the rest."""
    rgb, Y, side, info, _ = render(piece_a(), DEFAULT_LOOK)
    assert side["cells"] is None and side["under"] is None
    spec = side["shape"]
    assert set(spec) == {"name", "w", "ripple"} and len(spec["w"]) == 12 and spec["ripple"] == RIPPLE
    assert side["shift"] == round(side["shift"]) and side["smooth"] == 1.0
    assert snr(info["x"], codec.decode(Y, side)) >= 30.0          # float luma, no channel
    assert snr(info["x"], codec.decode(rgb, side)) >= 30.0        # lossless PNG

def test_experimental_shape_same_audio_same_image_and_exact_decode():
    x = piece_a()
    a = render_shape(x, "flower", ripple=14.0)
    b = render_shape(x, "flower", ripple=14.0)
    png = []
    for rgb in (a[0], b[0]):
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG"); png.append(buf.getvalue())
    assert png[0] == png[1] and hashlib.sha256(png[0]).digest() == hashlib.sha256(png[1]).digest()
    _, Y, side, info, _ = a
    assert snr(info["x"], codec.decode(Y, side)) >= 30.0          # ripple is subtracted exactly
