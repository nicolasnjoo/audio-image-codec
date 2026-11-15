# audio-image-codec — a2i

**a2i** turns a ~16 s audio clip into a 1080 × 1080 image that is (1) pleasant to look at,
(2) decodable back to the audio, and (3) degrades gracefully: a worse copy of the image
gives noisier audio, never a decode failure.

The payload is analog. MDCT coefficients are written directly as luma amplitudes
(8 × 8-module orthonormal DCT tiles, 2 × 2 px modules); there is no quantiser, entropy
coder or error correction on the payload. Colour carries no data and is free for styling.
No ML, no third-party audio codec.

Status: Python prototype. `frame` (finder patterns + header) and `register` (alignment)
are not built yet; side information is still passed out of band and alignment error is
simulated as a sub-pixel shift.

## Layout

```
a2i/        mdct.py  codec.py  channel.py  looks.py  features.py  testsignals.py
scripts/    check_baseline.py  levers.py  colour_test.py  render_options.py
            loss_budget.py  figures.py
tests/      test_mdct.py  test_roundtrip.py  test_determinism.py  test_colorize.py
out/        generated images, WAVs, JSON (git-ignored)
src/codec/  earlier empty project skeleton, untouched
```

Pipeline, one function per stage:

- Encoder: `prepare → analyze → whiten → pack → normalize → modulate → rasterize → colorize → frame`
- Decoder: `register → luma → demodulate → denormalize → unpack → unwhiten → synthesize`

## How to run

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q                  # 36 tests, ~30 s

.venv/bin/python scripts/check_baseline.py     # MDCT reconstruction + grey baseline
.venv/bin/python scripts/levers.py             # layout / contrast / grain levers
.venv/bin/python scripts/colour_test.py        # colour fix regression
.venv/bin/python scripts/loss_budget.py        # loss budget at the recommended settings
.venv/bin/python scripts/render_options.py     # four looks x three pieces, 12 PNGs
.venv/bin/python scripts/figures.py            # contact sheets, WAVs, spectrogram figures
```

Each script prints its measurements next to the reference values, flags anything outside
±0.3 dB, and writes a JSON copy to `out/`. Python ≥ 3.10.

## Format (current version)

| Constant | Value |
|---|---|
| Audio | 16 000 Hz mono, 258 048 samples (16.128 s), peak-normalised to 1/1.02 |
| Transform | MDCT, 64 bins, hop 64 (4 ms), sine window, 4032 frames |
| Canvas | 1080 × 1080 px; data square 1024 × 1024 px at offset 28; border neutral (reserved) |
| Module / tile | 2 × 2 px module; 8 × 8 modules per tile; 64 × 64 tiles; 1 DC + 63 AC slots per tile |
| Capacity | 4096 × 63 = 258 048 coefficients = all of them |
| Luma amplitudes | envelope `A_ENV = 60`, texture `A_RES = 75` (recommended) |
| Layouts | `raster`, `serpentine`, `hilbert`, `spiral_in` |

## Current numbers

Measured on 2026-11-15 with Python 3.13.18, numpy 2.5.4, scipy 1.18.1, Pillow 13.0.0
(libjpeg-turbo). Audio SNR in dB on three synthetic test pieces (A, B, C), recommended
settings: Record look, `spiral_in` layout, 60/75. The channel is simulated
(`a2i/channel.py`); no real phone screenshot has been measured yet.

| Condition | A | B | C | Mean |
|---|---|---|---|---|
| Float luma, no channel | 32.5 | 21.3 | 37.8 | 30.5 |
| Lossless PNG | 32.5 | 21.3 | 37.7 | 30.5 |
| JPEG q85 file | 23.6 | 19.7 | 24.4 | 22.6 |
| Screenshot (q75, 1080→1179→1080), perfect alignment | 19.0 | 16.6 | 19.5 | 18.4 |
| Screenshot, 0.15 px alignment error | 16.7 | 15.3 | 17.4 | 16.5 |
| Screenshot, 0.35 px alignment error | 11.1 | 10.9 | 11.5 | 11.2 |
| q60 screenshot, 0.5 px (worst case) | 7.0 | 7.3 | 7.4 | 7.3 |

Colour costs 0.06–0.20 dB after a screenshot (mean of the three pieces, per look).
Piece B's 21 dB lossless ceiling is luma clipping (1.00 % of modules) on tonal material.
The MDCT stage alone reconstructs to > 270 dB except for the final 64 samples (4 ms),
which are not reconstructed because the last half-window frame is not carried.

## Palette wording

The palette is derived from DSP features of the audio (key, mode, spectral centroid,
tonality). The detected key sets the base hue on the circle of fifths. The mode does
**not** make the palette warmer or cooler: it only selects the direction in which the
second hue is rotated away from the base hue (major: +spread, minor: −spread).
