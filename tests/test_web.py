"""The browser codec (web/a2i.js, run under Node) must agree with the Python reference."""
import json, pathlib, shutil, subprocess
import numpy as np
import pytest
from scipy.signal import resample_poly
from a2i import codec, container as ct, shapes
from a2i.testsignals import piece_c

ROOT = pathlib.Path(__file__).resolve().parents[1]
node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")

def js(code):
    out = subprocess.run(["node", "-e", f"const A = require({json.dumps(str(ROOT / 'web' / 'a2i.js'))}); {code}"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)

@node
def test_tile_order_is_identical_including_symmetric_ties():
    """Equal drops placed symmetrically put whole lines of tiles exactly on a tie; both
    implementations must settle those the same way (this once differed on 173 tiles)."""
    cases = [[0, 7, 7, 0, 0, 0, 5, 0, 15, 4, 0, 0], [15]*12, [15, 0, 0, 15, 0, 0, 15, 0, 0, 15, 0, 0],
             [0]*12, [15, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0], [15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    got = js(f"console.log(JSON.stringify({json.dumps(cases)}.map(w => Array.from(A.buildShape(w, 8, false).cells))));")
    for w, cells in zip(cases, got):
        assert np.array_equal(shapes.build("droplets", w)[0], np.array(cells)), w

@node
def test_header_and_gain_bits_are_identical():
    w = [0, 3, 15, 0, 7, 0, 0, 12, 0, 1, 0, 9]
    q = (np.arange(codec.NBLK*codec.NB)*7 % 32).reshape(codec.NBLK, codec.NB)
    got = js(f"console.log(JSON.stringify([A.headerBits(32000, -1, {json.dumps(w)}), A.gainBits({json.dumps(q.ravel().tolist())})]));")
    assert got[0] == ct.header_bits(32000, -1.0, w) and got[1] == ct.gain_bits(q)

@node
def test_pictures_decode_across_implementations(tmp_path):
    x32 = resample_poly(piece_c(), 2, 1)[:codec.NS]
    rgb_py, info = ct.encode_image(x32, 32000)
    (tmp_path / "x.f32").write_bytes(x32.astype("<f4").tobytes())
    (tmp_path / "py.rgba").write_bytes(np.dstack([rgb_py, np.full(rgb_py.shape[:2], 255, np.uint8)]).tobytes())
    script = str(ROOT / "scripts" / "web_node.js")
    subprocess.run(["node", script, "encode", str(tmp_path / "x.f32"), str(tmp_path / "js.rgba"), str(tmp_path / "e.json")], check=True)
    subprocess.run(["node", script, "decode", str(tmp_path / "py.rgba"), "1080", "1080", str(tmp_path / "y.f32"), str(tmp_path / "d.json")], check=True)
    rgb_js = np.frombuffer((tmp_path / "js.rgba").read_bytes(), np.uint8).reshape(1080, 1080, 4)[..., :3]
    o = codec.ORIGIN
    assert np.abs(ct.luma(rgb_js) - ct.luma(rgb_py))[o:-o, o:-o].max() <= 1.0      # same picture, to one grey level
    y_py, h = ct.decode_image(rgb_js)                                              # JavaScript picture, Python decoder
    y_js = np.frombuffer((tmp_path / "y.f32").read_bytes(), "<f4").astype(float)   # Python picture, JavaScript decoder
    assert h["w"] == info["w"] and h["gains_ok"]
    assert ct.snr_gain(info["x"], y_py) >= 30.0 and ct.snr_gain(info["x"], y_js) >= 30.0
