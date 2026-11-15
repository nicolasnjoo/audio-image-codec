"""Shared by the scripts: puts the repo root on sys.path, defines out/, and prints
measured-vs-reference tables (tolerance 0.3 dB, section 7 of the project brief)."""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
TOL = 0.3

def compare(title, rows, tol=TOL, unit="dB"):
    """rows: (label, measured, reference or None). Prints a table, returns the rows as dicts."""
    print(f"\n{title}")
    print(f"  {'item':<46} {'measured':>9} {'ref':>7} {'delta':>7}")
    out = []
    for label, meas, ref in rows:
        if ref is None:
            print(f"  {label:<46} {meas:9.2f} {'-':>7} {'-':>7}")
            out.append(dict(item=label, measured=float(meas), ref=None, ok=None))
            continue
        d = meas - ref
        flag = "" if abs(d) <= tol else f"  <-- outside +/-{tol} {unit}"
        print(f"  {label:<46} {meas:9.2f} {ref:7.2f} {d:+7.2f}{flag}")
        out.append(dict(item=label, measured=float(meas), ref=float(ref), delta=float(d),
                        ok=bool(abs(d) <= tol)))
    n_bad = sum(1 for r in out if r["ok"] is False)
    print(f"  -> {len(out) - n_bad}/{len(out)} within tolerance" if n_bad else "  -> all within tolerance")
    return out

def save_json(name, obj):
    with open(OUT / name, "w") as fh:
        json.dump(obj, fh, indent=1)
