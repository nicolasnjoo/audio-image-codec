# audio-image-codec — a2i

**a2i** turns a ~16 s audio clip into a 1080 × 1080 image that is (1) pleasant to look at,
(2) decodable back to the audio, and (3) degrades gracefully: a worse copy of the image
gives noisier audio, never a decode failure.

The payload is analog. MDCT coefficients are written directly as luma amplitudes
(8 × 8-module orthonormal DCT tiles, 2 × 2 px modules); there is no quantiser, entropy
coder or error correction on the payload. Colour carries no data and is free for styling.
No ML, no third-party audio codec.

Status: Python prototype plus a browser version. The default look is **Droplets**, carrying
8.064 s of 32 kHz mono. A first, provisional self-contained image exists
(`a2i/container.py`): the picture carries its own header and can be found and aligned
inside a screenshot. The older functions with out-of-band side information still work and
are what most measurements below use.

## Layout

```
a2i/        mdct.py  codec.py  shapes.py  container.py  channel.py  looks.py  features.py
            testsignals.py
web/        a2i.js  worker.js  index.html   (the same codec in the browser, and the test page)
scripts/    check_baseline.py  levers.py  colour_test.py  render_options.py
            loss_budget.py  figures.py  real_music.py  container_test.py
            web_check.py  web_node.js
            explore_aesthetics.py  explore_shapes.py
            explore_quality.py  explore_fine_layer.py   (experimental, not the format)
tests/      test_mdct.py  test_roundtrip.py  test_determinism.py  test_colorize.py
            test_explore.py  test_shapes.py  test_rate.py  test_container.py  test_web.py
out/        generated images, WAVs, JSON (git-ignored)
```

Pipeline, one function per stage:

- Encoder: `prepare → analyze → whiten → pack → normalize → modulate → rasterize → colorize → frame`
- Decoder: `register → luma → demodulate → denormalize → unpack → unwhiten → synthesize`

## How to run

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q                  # 71 tests, ~50 s (three need Node.js)

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
| Audio | Mono, 258 048 samples per image, peak-normalised to 1/1.02. The sample rate is a format parameter: 16 kHz gives 16.128 s, 25.6 kHz gives 10.08 s, 32 kHz gives 8.064 s |
| Transform | MDCT, 64 bins, hop 64 samples, sine window, 4032 frames |
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
| Layout from the notes | Tiles are ordered along the contour lines of a landscape built from the notes (`a2i/shapes.py`) | 12 note weights, 4 bits each = 6 bytes |
| Sample rate | The image holds 258 048 samples at any rate; `RATE = 32000` is used for real music | one header field |
| Contour ripple | A light/dark wave of ±8 grey levels along the contours; the decoder knows it and subtracts it | none (constant) |
| Envelope range 36 dB | The dark-to-bright swing of a tile covers 36 dB of loudness instead of 54 dB, so beats show more | none (constant) |

Encoder policy, not a decoder constant: the centre luma follows the key hue (128–165), and
when it is raised, loud moments are drawn dark (`A_ENV` negative).

**Droplets:** one drop per note that stands out, placed on a fixed 12-hour clock (C at 12
o'clock, one hour per semitone). A note's weight is how far it rises above the median of
the twelve; a drop is drawn from 20 % upwards and its size follows the weight. The clock is
fixed, so the shape does not depend on the key detector. The drops merge outwards. Time
starts on the outermost contour and ends inside the drops. Colour follows the time path
continuously; hue = key hue plus the harmony (tonal centroid on the circle of fifths).

Not yet solved: the ordering uses floating-point sort keys. It is deterministic on one
machine (tested), but a bit-exact cross-platform format needs an integer definition.

## Current numbers

Measured on 2026-11-15 with Python 3.13.18, numpy 2.5.4, scipy 1.18.1, Pillow 13.0.0
(libjpeg-turbo). Audio SNR in dB on three synthetic test pieces (A, B, C; 16 kHz signals,
so run at 16 kHz), Droplets look, 60/75, decoded from the header-sized side information. The channel is simulated
(`a2i/channel.py`); no real phone screenshot has been measured yet. Real music is below.

| Condition | A | B | C | Mean | Record, mean |
|---|---|---|---|---|---|
| Float luma, no channel | 32.1 | 28.4 | 40.9 | 33.8 | 30.5 |
| Lossless PNG | 32.1 | 28.4 | 40.9 | 33.8 | 30.5 |
| JPEG q85 file | 23.6 | 24.1 | 24.6 | 24.1 | 22.6 |
| Screenshot (q75, 1080→1179→1080), perfect alignment | 19.0 | 19.2 | 19.8 | 19.3 | 18.4 |
| Screenshot, 0.15 px alignment error | 16.9 | 17.3 | 17.5 | 17.2 | 16.5 |
| Screenshot, 0.35 px alignment error | 11.2 | 11.7 | 11.6 | 11.5 | 11.2 |
| q60 screenshot, 0.5 px (worst case) | 7.1 | 7.7 | 7.5 | 7.4 | 7.3 |

"Record" is the previous default (first format); its numbers reproduce the project's
reference values within 0.05 dB. Clipped modules with Droplets: A 0.02 %, B 0.41 %,
C 0.09 % (Record: 0.02 %, 1.00 %, 0.02 %).

The MDCT stage alone reconstructs to > 270 dB except for the final 64 samples (4 ms),
which are not reconstructed because the last half-window frame is not carried.

## Real music

`scripts/real_music.py NAME=file.wav ... --at 0,60,120 --rate 32000` runs any PCM WAV
through the same pipeline (decode MP3 first, for example with ffmpeg). Measured on twelve
excerpts of four commercial songs (MP3, 44.1 or 48 kHz stereo), starting at 0, 60 and 120 s.
Droplets look. SNR is against the prepared signal (mono, at the image's rate), so the
deliberate loss of stereo and of everything above half the rate is not counted. No audio is
stored in this repository; `out/` is git-ignored.

| Condition | 32 kHz, 8.06 s: mean | minimum | 25.6 kHz, 10.08 s: mean | minimum |
|---|---|---|---|---|
| Lossless PNG | 37.5 | 27.9 | 36.1 | 24.1 |
| JPEG q85 file | 23.8 | 22.3 | 23.6 | 20.2 |
| Screenshot, perfect alignment | 19.1 | 18.2 | 19.1 | 17.4 |
| Screenshot, 0.15 px alignment error | 17.0 | 16.3 | 17.0 | 16.0 |
| Screenshot, 0.35 px alignment error | 11.4 | 11.0 | 11.4 | 11.2 |
| q60 screenshot, 0.5 px (worst case) | 7.3 | 6.9 | 7.4 | 7.2 |

The decode quality does not depend on the rate, because every coefficient still gets one
module; the rate only trades clip length for bandwidth. At 16 kHz (16.13 s) the same
excerpts gave 17.0 dB after the screenshot and 7.3 dB in the worst case.

How much of the music lies in the bands the rate decides (twelve 10 s excerpts): the band
above 8 kHz holds −24.7 dB of the energy on average (−29.4 to −19.1 dB), the band above
12.8 kHz holds −36.9 dB (−41.8 to −32.3 dB).

Segmental SNR is lower on sparse passages: about 15.3 dB after the screenshot and 6.2 dB
in the worst case for the quietest excerpt.

What real music changed, relative to the synthetic pieces (measured at 16 kHz, grey,
screenshot 0.15 px, mean, each lever alone on the first format):

| Lever | Real music | Synthetic |
|---|---|---|
| Centred brightness | +0.13 dB | +0.86 dB |
| Whitening with 1 time block instead of 32 | −0.51 dB | −0.03 dB |
| Whitening with 8 time blocks | −0.22 dB | −0.10 dB |
| Grain amplitude 62 instead of 75 | −0.90 dB | −0.31 dB |
| Envelope range 36 dB instead of 54 dB | +0.02 dB | +0.23 dB |

Consequences: Droplets is 0.1 dB better than Record on real music after a screenshot, not
0.9 dB (that gain came mostly from one very tonal synthetic piece that clipped 1 % of its
modules; real music clips 0.02 %). The 32 whitening time blocks are needed. Reducing the
grain costs about three times more than the synthetic pieces suggested.

Envelope range on real music (16 kHz): 95 % of the 4 ms moments lie within 22 dB of the
loudest moment of their clip, and 1.4 % lie more than 36 dB below it.

| Envelope range | Screenshot SNR | Worst case | Tile brightness spread | Moments at the end of the scale |
|---|---|---|---|---|
| 54 dB (first format) | 17.01 dB | 7.35 dB | 11.4 grey levels | 0.9 % |
| 36 dB (adopted) | 16.96 dB | 7.32 dB | 14.6 grey levels | 1.3 % |
| 24 dB | 16.77 dB | 7.25 dB | 18.9 grey levels | 3.1 % (17.6 % in the sparsest excerpt) |

Shape on real music: with the adopted note weights the number of drops went from 5.2 to 3.6
per image on average, excerpts of the same song share their main drops, and the audio is
unchanged within the usual layout variation (16.96 → 16.92 dB after a screenshot).

## Self-contained image and alignment (provisional, version 1)

`a2i/container.py` puts everything the decoder needs into the picture and finds the picture
in any upright image of it. It is a first version, built so that real screenshots can be
tested; it has not been agreed as the format.

| Part | How |
|---|---|
| Finders | Four QR-style squares (7 × 7 modules of 4 px) in the corners of the 28 px border |
| Ring | 4032 black/white cells of 4 × 4 px around the data square |
| Header | Version, sample rate, polarity, 12 note weights, CRC-8: 64 bits, 6 copies around the ring |
| Gains | The 512 whitening gains in 3 dB steps (5 bits each) plus CRC-16, 1.4 copies. The encoder whitens with the same quantised gains, so both sides use identical values |
| Level | Not carried: the decoder returns audio normalised to its peak, and quality is measured with a best-fit level |
| Alignment | Row and column scans for the finders, sub-pixel centres from their edges, offset and scale per axis, Lanczos resampling to 1080 × 1080 |

Twelve real excerpts at 32 kHz, decoded from the picture alone (`scripts/container_test.py`):

| Input | Mean SNR | Minimum |
|---|---|---|
| The image file (PNG) | 37.4 dB | 28.1 dB |
| The image file re-saved as JPEG q85 | 23.4 dB | 22.2 dB |
| Screenshot on a 1179 px wide phone, PNG | 28.2 dB | 24.7 dB |
| The same screenshot sent as JPEG q75 | 20.1 dB | 19.0 dB |
| The same screenshot sent as JPEG q60 | 17.0 dB | 16.3 dB |
| Screenshot where the viewer scaled bilinearly, PNG | 18.1 dB | 17.9 dB |

The ring was read without error in 72 of 72 decodes. On a 1080 px wide phone a screenshot of
the picture shown at full size is an exact copy (same result as the file). Alignment error
was 0.03–0.07 px RMS over the image (target 0.15 px), scale error at most 0.016 %. These are
simulated screenshots; no real phone screenshot has been measured yet.

Carrying the gains as analog values in reserved tiles was tried first and cost 2–3 dB as
soon as the channel added noise, because a small gain error becomes a noise floor.

## Browser version and test page

`web/a2i.js` is the same codec in JavaScript, with no dependencies; `web/worker.js` runs it
off the main thread and `web/index.html` is the test page (page content for the published
page, without an `<html>` wrapper). `scripts/web_check.py` runs the JavaScript under Node and
compares it with Python: on twelve real excerpts and the three synthetic pieces the two make
the same picture (luma equal to within one grey level) and each decodes the other's pictures
with the same SNR to two decimals. The browser's own sample-rate conversion differs from
scipy's, so a picture made in a browser from a 44.1 or 48 kHz file is not bit-identical to
one made in Python from the same file.

## Clip length against bandwidth (decided: square 1080, 32 kHz, 8.06 s)

The image holds 258 048 samples, so 32 kHz gives 8.06 s. The alternatives that were measured,
twelve real excerpts, SNR after the simulated screenshot:

| Option | Clip | Real band | Screenshot SNR | Changes |
|---|---|---|---|---|
| Square 1080, 32 kHz (chosen) | 8.06 s | 16 kHz | 17.0 dB | nothing |
| Square 1080, 25.6 kHz | 10.08 s | 12.8 kHz | 17.0 dB | nothing |
| Portrait 1080 × 1350, 32 kHz | 10.08 s | 16 kHz | same module size and scaling as today, so no change expected; not built | the image is no longer square |
| Square 1208, 32 kHz | 10.2 s | 16 kHz | 16.6 dB on a 1179 px wide phone, 15.8 dB on a 1080 px wide phone (measured by drawing today's image smaller) | the image is no longer 1080 px |

## Other audio quality options (measured, not adopted)

The image carries 258 048 numbers, one per MDCT coefficient. That count is set by the
2 × 2 px module, the smallest that survives a phone screenshot. The chosen direction is a
shorter clip at a higher sample rate (above). The alternatives below were measured against
16.128 s of 16 kHz mono, where everything above 8 kHz and the stereo image are lost before
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
