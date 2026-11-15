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
simulated as a sub-pixel shift. The default look is **Droplets**.

## Layout

```
a2i/        mdct.py  codec.py  shapes.py  channel.py  looks.py  features.py  testsignals.py
scripts/    check_baseline.py  levers.py  colour_test.py  render_options.py
            loss_budget.py  figures.py
            explore_aesthetics.py  explore_shapes.py   (experimental, not the format)
tests/      test_mdct.py  test_roundtrip.py  test_determinism.py  test_colorize.py
            test_explore.py  test_shapes.py
out/        generated images, WAVs, JSON (git-ignored)
```

Pipeline, one function per stage:

- Encoder: `prepare → analyze → whiten → pack → normalize → modulate → rasterize → colorize → frame`
- Decoder: `register → luma → demodulate → denormalize → unpack → unwhiten → synthesize`

## How to run

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q                  # 54 tests, ~40 s

.venv/bin/python scripts/check_baseline.py     # MDCT reconstruction + grey baseline
.venv/bin/python scripts/levers.py             # layout / contrast / grain levers
.venv/bin/python scripts/colour_test.py        # colour fix regression
.venv/bin/python scripts/loss_budget.py        # loss budget: Record (reference) and Droplets
.venv/bin/python scripts/render_options.py     # five looks x three pieces, 15 PNGs
.venv/bin/python scripts/figures.py            # contact sheets, WAVs, spectrogram figures
```

Each script prints its measurements next to the reference values where they exist, flags
anything outside ±0.3 dB, and writes a JSON copy to `out/`. Python ≥ 3.10.

## Format

| Constant | Value |
|---|---|
| Audio | 16 000 Hz mono, 258 048 samples (16.128 s), peak-normalised to 1/1.02 |
| Transform | MDCT, 64 bins, hop 64 (4 ms), sine window, 4032 frames |
| Canvas | 1080 × 1080 px; data square 1024 × 1024 px at offset 28; border neutral (reserved) |
| Module / tile | 2 × 2 px module; 8 × 8 modules per tile; 64 × 64 tiles; 1 DC + 63 AC slots per tile |
| Capacity | 4096 × 63 = 258 048 coefficients = all of them |
| Luma amplitudes | envelope `A_ENV = 60`, texture `A_RES = 75` |
| Envelope range | 9 log2 units (54 dB) below the loudest tile |

Adopted on top of the first format (used by Droplets; every codec function default still
reproduces the first format, so the four original looks are byte-identical to before):

| Addition | What it does | Side information |
|---|---|---|
| Centred brightness | The median tile is drawn at a chosen luma instead of wherever the music puts it (was 151–169) | `shift`, one integer |
| Smooth envelope | Tile brightness is drawn as a smooth field that keeps every tile mean; the decoder rebuilds the field from the tile means and subtracts it | none (constant, blend 1.0 tile) |
| Layout from the notes | Tiles are ordered along the contour lines of a landscape built from the notes (`a2i/shapes.py`) | 12 pitch-class weights, 4 bits each = 6 bytes |
| Contour ripple | A light/dark wave of ±8 grey levels along the contours; the decoder knows it and subtracts it | none (constant) |

Encoder policy, not a decoder constant: the centre luma follows the key hue (128–165), and
when it is raised, loud moments are drawn dark (`A_ENV` negative).

**Droplets:** one drop per prominent pitch class, placed on a 12-hour clock (tonic at 12
o'clock, one hour per semitone), size = how much that note is used. The drops merge
outwards. Time starts on the outermost contour and ends inside the drops. Colour follows the
time path continuously; hue = key hue plus the harmony (tonal centroid on the circle of
fifths).

Not yet solved: the ordering uses floating-point sort keys. It is deterministic on one
machine (tested), but a bit-exact cross-platform format needs an integer definition.

## Current numbers

Measured on 2026-11-15 with Python 3.13.18, numpy 2.5.4, scipy 1.18.1, Pillow 13.0.0
(libjpeg-turbo). Audio SNR in dB on three synthetic test pieces (A, B, C), Droplets look,
60/75, decoded from the header-sized side information. The channel is simulated
(`a2i/channel.py`); no real music and no real phone screenshot has been measured yet.

| Condition | A | B | C | Mean | Record, mean |
|---|---|---|---|---|---|
| Float luma, no channel | 37.3 | 28.3 | 41.6 | 35.7 | 30.5 |
| Lossless PNG | 37.3 | 28.3 | 41.6 | 35.7 | 30.5 |
| JPEG q85 file | 24.1 | 24.0 | 24.7 | 24.3 | 22.6 |
| Screenshot (q75, 1080→1179→1080), perfect alignment | 19.5 | 19.4 | 19.8 | 19.6 | 18.4 |
| Screenshot, 0.15 px alignment error | 17.3 | 17.3 | 17.5 | 17.4 | 16.5 |
| Screenshot, 0.35 px alignment error | 11.3 | 11.7 | 11.7 | 11.6 | 11.2 |
| q60 screenshot, 0.5 px (worst case) | 7.1 | 7.7 | 7.6 | 7.4 | 7.3 |

"Record" is the previous default (first format); its numbers reproduce the project's
reference values within 0.05 dB. Clipped modules with Droplets: A 0.01 %, B 0.33 %,
C 0.05 % (Record: 0.02 %, 1.00 %, 0.02 %). Delivered colour (mean chroma radius; a pure
primary is ~127): 47 with Droplets, 34 with Record.

The MDCT stage alone reconstructs to > 270 dB except for the final 64 samples (4 ms),
which are not reconstructed because the last half-window frame is not carried.

## Measured but not adopted

`scripts/explore_aesthetics.py` and `scripts/explore_shapes.py` keep the measurements behind
the choices above. Audio SNR in dB after the simulated screenshot (q75, 0.15 px), colour,
mean of pieces A/B/C.

| Step, cumulative from the Record look (16.48) | SNR |
|---|---|
| Median tile at luma 128 | 17.38 |
| + smooth envelope | 17.40 |
| + round spiral layout | 17.30 |
| + colour along the time path, hue follows the harmony | 17.30 |

| Option on top of that | SNR |
|---|---|
| Envelope range 36 dB instead of 54 dB (more contrast on beats) | 17.16 |
| Grain amplitude 62 instead of 75 (calmer texture, more colour) | 16.73 |
| Loud moments drawn dark | 17.35 |

| Droplets, ripple amplitude | 0 | 8 (adopted) | 14 | 22 |
|---|---|---|---|---|
| SNR | 17.38 | 17.38 | 17.28 | 17.06 |

Other landscapes at ripple 8: round spiral 17.26, `flower` (the scale as a star) 17.39,
`chladni` (vibrating-plate figure) 17.22. They read two more numbers from the music
(brightness, tonality).

## Palette wording

The palette is derived from DSP features of the audio (key, mode, spectral centroid,
tonality). The detected key sets the base hue on the circle of fifths. The mode does
**not** make the palette warmer or cooler: it only selects the direction in which the
second hue is rotated away from the base hue (major: +spread, minor: −spread).
