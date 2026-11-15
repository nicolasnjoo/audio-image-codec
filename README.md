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
            loss_budget.py  figures.py  real_music.py
            explore_aesthetics.py  explore_shapes.py  explore_droplets_v2.py
            explore_quality.py  explore_fine_layer.py   (experimental, not the format)
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
.venv/bin/python -m pytest -q                  # 55 tests, ~45 s

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
| Envelope range | 6 log2 units (36 dB) below the loudest tile in Droplets; 9 (54 dB) in the first format |

Adopted on top of the first format (used by Droplets; every codec function default still
reproduces the first format, so the four original looks are byte-identical to before):

| Addition | What it does | Side information |
|---|---|---|
| Centred brightness | The median tile is drawn at a chosen luma instead of wherever the music puts it (was 151–169) | `shift`, one integer |
| Smooth envelope | Tile brightness is drawn as a smooth field that keeps every tile mean; the decoder rebuilds the field from the tile means and subtracts it | none (constant, blend 1.0 tile) |
| Layout from the notes | Tiles are ordered along the contour lines of a landscape built from the notes (`a2i/shapes.py`) | 12 pitch-class weights, 4 bits each = 6 bytes |
| Contour ripple | A light/dark wave of ±8 grey levels along the contours; the decoder knows it and subtracts it | none (constant) |
| Envelope range 36 dB | The dark-to-bright swing of a tile covers 36 dB of loudness instead of 54 dB, so beats show more | none (constant) |

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
(`a2i/channel.py`); no real phone screenshot has been measured yet. Real music is below.

| Condition | A | B | C | Mean | Record, mean |
|---|---|---|---|---|---|
| Float luma, no channel | 32.0 | 28.0 | 40.8 | 33.6 | 30.5 |
| Lossless PNG | 32.0 | 28.0 | 40.8 | 33.6 | 30.5 |
| JPEG q85 file | 23.7 | 23.9 | 24.7 | 24.1 | 22.6 |
| Screenshot (q75, 1080→1179→1080), perfect alignment | 19.1 | 19.3 | 19.7 | 19.4 | 18.4 |
| Screenshot, 0.15 px alignment error | 17.0 | 17.3 | 17.5 | 17.3 | 16.5 |
| Screenshot, 0.35 px alignment error | 11.3 | 11.7 | 11.6 | 11.5 | 11.2 |
| q60 screenshot, 0.5 px (worst case) | 7.1 | 7.7 | 7.5 | 7.4 | 7.3 |

"Record" is the previous default (first format); its numbers reproduce the project's
reference values within 0.05 dB. Clipped modules with Droplets: A 0.02 %, B 0.40 %,
C 0.10 % (Record: 0.02 %, 1.00 %, 0.02 %).

The MDCT stage alone reconstructs to > 270 dB except for the final 64 samples (4 ms),
which are not reconstructed because the last half-window frame is not carried.

## Real music

`scripts/real_music.py NAME=file.wav ... --at 0,60,120` runs any PCM WAV through the same
pipeline (decode MP3 first, for example with ffmpeg). Measured on twelve 16.128 s excerpts:
four commercial songs (MP3, 44.1 or 48 kHz stereo), starting at 0, 60 and 120 s. Droplets
look. SNR is against the prepared signal (mono, 16 kHz), so the deliberate loss of stereo
and of everything above 8 kHz is not counted. No audio is stored in this repository;
`out/` is git-ignored.

| Condition | Mean of 12 | Minimum | Synthetic mean |
|---|---|---|---|
| Lossless PNG | 37.6 | 32.5 | 33.6 |
| JPEG q85 file | 23.6 | 22.9 | 24.1 |
| Screenshot, perfect alignment | 19.0 | 18.3 | 19.4 |
| Screenshot, 0.15 px alignment error | 17.0 | 16.4 | 17.3 |
| Screenshot, 0.35 px alignment error | 11.4 | 11.0 | 11.5 |
| q60 screenshot, 0.5 px (worst case) | 7.3 | 7.0 | 7.4 |

Segmental SNR is lower on sparse passages: 15.3 dB after the screenshot and 6.3 dB in the
worst case for the quietest excerpt.

What real music changed, relative to the synthetic pieces (grey, screenshot 0.15 px, mean,
each lever alone on the first format):

| Lever | Real music | Synthetic |
|---|---|---|
| Centred brightness | +0.13 dB | +0.86 dB |
| Whitening with 1 time block instead of 32 | −0.51 dB | −0.03 dB |
| Whitening with 8 time blocks | −0.22 dB | −0.10 dB |
| Grain amplitude 62 instead of 75 | −0.90 dB | −0.31 dB |
| Envelope range 36 dB instead of 54 dB | +0.02 dB | +0.23 dB |

Consequences: Droplets is 0.10 dB better than Record on real music after a screenshot, not
0.90 dB (that gain came mostly from one very tonal synthetic piece that clipped 1 % of its
modules; real music clips 0.02 %). The 32 whitening time blocks are needed. Reducing the
grain costs about three times more than the synthetic pieces suggested.

Envelope range on real music: 95 % of the 4 ms moments lie within 22 dB of the loudest
moment of their clip, and 1.4 % lie more than 36 dB below it. With the Droplets look:

| Envelope range | Screenshot SNR | Worst case | Tile brightness spread | Moments at the end of the scale |
|---|---|---|---|---|
| 54 dB (first format) | 17.01 dB | 7.35 dB | 11.4 grey levels | 0.9 % |
| 36 dB (adopted) | 16.96 dB | 7.32 dB | 14.6 grey levels | 1.3 % |
| 24 dB | 16.77 dB | 7.25 dB | 18.9 grey levels | 3.1 % (17.6 % in the sparsest excerpt) |

## Audio quality options (measured, none adopted)

The image carries 258 048 numbers, one per MDCT coefficient. That count is set by the
2 × 2 px module, the smallest that survives a phone screenshot. Today it is spent on
16.128 s of 16 kHz mono, so everything above 8 kHz and the stereo image are lost before
encoding. In the two excerpts examined, the band above 8 kHz holds −21 and −27 dB of the
energy, and the left-right difference is −9 and −12 dB relative to the sum.

`scripts/explore_quality.py` writes listening examples of the options that fit the same
image. SNR after the simulated screenshot is about 17 dB for all of the "real" options,
because each coefficient still gets one module.

| Option | Audio | Clip length | Cost |
|---|---|---|---|
| Today | 16 kHz mono | 16.1 s | — |
| Higher sample rate | 24 kHz mono (real band to 12 kHz) | 10.8 s | shorter clip |
| Higher sample rate | 32 kHz mono (real band to 16 kHz) | 8.1 s | shorter clip |
| Real stereo | 16 kHz stereo (sum and difference) | 8.1 s | shorter clip |
| Rebuilt highs | 32 kHz mono; 8–16 kHz is copied from 3–7 kHz and shaped by 512 loudness values | 16.1 s | 512 extra values; the highs are plausible, not exact (median level error 4–5 dB on a 500 Hz × 32 ms grid) |

`scripts/explore_fine_layer.py` measures the feasibility of a second tier: three more
numbers inside every module, as zero-mean patterns the present decoder does not see. That
is four times the capacity (for example 32 kHz stereo for the full 16 s), present only when
the image file itself is shared. Stand-in data, grey images, four real excerpts:

| Fine-layer amplitude | Today's decode after a screenshot | Fine layer: lossless PNG | JPEG q95 | JPEG q90 | JPEG q85 | Screenshot |
|---|---|---|---|---|---|---|
| none | 17.07 dB | — | — | — | — | — |
| 15 grey levels | 16.65 dB | 28.5 dB | 12.6 dB | 6.8 dB | 3.7 dB | −0.8 dB |
| 30 grey levels | 15.25 dB | 29.9 dB | 18.4 dB | 12.7 dB | 9.2 dB | 2.9 dB |

## Shape prototype (not adopted)

`scripts/explore_droplets_v2.py`: the clock is fixed (C at 12 o'clock) instead of starting at
the detected key note, and a drop is drawn only for a note that rises above the median of
the twelve. On the twelve real excerpts the number of drops goes from 5.2 to 3.6 on
average, excerpts of the same song share their main drops, and the audio is unchanged
within the usual layout variation (16.96 → 16.92 dB after a screenshot).

## Measured but not adopted

The numbers in this section are from the synthetic pieces only; see the table above for how
they change on real music.

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
