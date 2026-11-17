"""Check the browser codec (web/a2i.js, run under Node) against the Python reference.

usage:  python scripts/web_check.py [NAME=file.wav ...] [--at 60]
        (without files: the three synthetic pieces, upsampled to 32 kHz)

For each clip: the image made by JavaScript is decoded by Python and the image made by Python
is decoded by JavaScript, both from the exact picture and from a simulated 1179 px wide
screenshot. SNR in dB with best-fit level (container.snr_gain).
"""
import argparse, json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))    # also works under python -I
import numpy as np
from scipy.signal import resample_poly
from _common import ROOT, save_json
from a2i import codec, container as ct
from a2i.testsignals import PIECES
from container_test import screenshot

NODE_SCRIPT = str(ROOT / "scripts" / "web_node.js")

def js_encode(x32, tmp):
    (tmp / "x.f32").write_bytes(np.asarray(x32, "<f4").tobytes())
    subprocess.run(["node", NODE_SCRIPT, "encode", str(tmp / "x.f32"), str(tmp / "js.rgba"), str(tmp / "enc.json")], check=True)
    rgba = np.frombuffer((tmp / "js.rgba").read_bytes(), np.uint8).reshape(1080, 1080, 4)
    return rgba[..., :3].copy(), json.loads((tmp / "enc.json").read_text())

def js_decode(rgb, tmp):
    h, w = rgb.shape[:2]
    rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
    (tmp / "in.rgba").write_bytes(rgba.tobytes())
    subprocess.run(["node", NODE_SCRIPT, "decode", str(tmp / "in.rgba"), str(w), str(h), str(tmp / "y.f32"), str(tmp / "dec.json")], check=True)
    info = json.loads((tmp / "dec.json").read_text())
    if "error" in info:
        return None, info
    return np.frombuffer((tmp / "y.f32").read_bytes(), "<f4").astype(float), info

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="NAME=path.wav")
    ap.add_argument("--at", default="60")
    args = ap.parse_args()
    clips = {}
    if args.files:
        from real_music import load
        for spec in args.files:
            name, path = spec.split("=", 1)
            fs, a = load(path)
            for t0 in (float(t) for t in args.at.split(",")):
                seg = a[int(t0*fs):int((t0 + 8.3)*fs)]
                seg = seg.mean(1) if seg.ndim == 2 else seg
                g = np.gcd(fs, 32000)
                clips[f"{name}@{t0:g}"] = resample_poly(seg, 32000//g, fs//g)[:codec.NS]
    else:
        clips = {f"piece {pk}": resample_poly(fn(), 2, 1)[:codec.NS] for pk, fn in PIECES.items()}
    res = {}
    print(f"{'clip':<16} {'same header':>11} {'luma diff':>10} | {'py->py':>7} {'js->py':>7} {'py->js':>7} {'js->js':>7} | screenshot: {'py->py':>7} {'js->py':>7} {'py->js':>7} {'js->js':>7} | JS ms enc/dec")
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        for k, (cid, x32) in enumerate(clips.items()):
            rgb_py, info = ct.encode_image(x32, 32000)
            rgb_js, einfo = js_encode(x32, tmp)
            ref = info["x"]
            same = (einfo["weights"] == info["w"]) and (einfo["loudIsDark"] == (info["sign"] < 0)) and einfo["key"] == info["key"]
            o = codec.ORIGIN
            dl = np.abs(ct.luma(rgb_py) - ct.luma(rgb_js))[o:-o, o:-o]
            row = []
            for shot_of in (lambda r: r, lambda r: screenshot(r, 1179, 2556, 1179/1080, 0.0, 688.37, 0, "lanczos", k)):
                sp, sj = shot_of(rgb_py), shot_of(rgb_js)
                ypp, _ = ct.decode_image(sp); yjp, _ = ct.decode_image(sj)
                ypj, dinfo = js_decode(sp, tmp); yjj, dinfo2 = js_decode(sj, tmp)
                row += [ct.snr_gain(ref, y) if y is not None else float("nan") for y in (ypp, yjp, ypj, yjj)]
            res[cid] = dict(same_header=bool(same), luma_diff_mean=float(dl.mean()), luma_diff_max=float(dl.max()), snr=row,
                            js_ms=[einfo["ms"], dinfo2.get("ms")], js_key=einfo["key"], py_key=info["key"])
            print(f"{cid:<16} {str(same):>11} {dl.mean():6.3f}/{dl.max():3.0f} | " + " ".join(f"{v:7.2f}" for v in row[:4]) +
                  " |             " + " ".join(f"{v:7.2f}" for v in row[4:]) + f" | {einfo['ms']}/{dinfo2.get('ms')}")
    save_json("web_check.json", res)
