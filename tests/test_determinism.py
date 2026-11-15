"""The exact same audio must always produce the exact same image, bit for bit."""
import hashlib, io, os, pathlib, subprocess, sys
import numpy as np
import pytest
from PIL import Image
from a2i.testsignals import piece_a
from render_options import render, LOOKS          # scripts/ is on pytest's pythonpath

ROOT = pathlib.Path(__file__).resolve().parents[1]

def png_bytes(rgb):
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "PNG"); return b.getvalue()

@pytest.mark.parametrize("name", list(LOOKS))
def test_encode_twice_identical_png(name):
    a = png_bytes(render(piece_a(), name)[0])
    b = png_bytes(render(piece_a(), name)[0])
    assert a == b
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()

def test_identical_across_fresh_processes():
    """Two new interpreters with different hash seeds must write the same PNG bytes."""
    code = ("import hashlib, io; from PIL import Image; from a2i.testsignals import piece_a; "
            "from render_options import render; b = io.BytesIO(); "
            "Image.fromarray(render(piece_a(), 'Record')[0]).save(b, 'PNG'); "
            "print(hashlib.sha256(b.getvalue()).hexdigest())")
    digests = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed,
                   PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]))
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
        digests.append(out)
    here = hashlib.sha256(png_bytes(render(piece_a(), "Record")[0])).hexdigest()
    assert digests[0] == digests[1] == here

def test_different_audio_gives_different_image():
    x = piece_a(); y = x.copy(); y[1000] += 0.01
    assert not np.array_equal(render(x, "Record")[0], render(y, "Record")[0])
