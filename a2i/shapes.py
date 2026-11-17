"""Layouts whose shape is generated from the music (adopted: droplets).

A layout is only an ordering of the 4096 tiles. Define a landscape h(x, y) over the image and
let time run along its contour lines: key = floor(h - angle) + angle, tiles sorted by key.
The round spiral is the landscape "distance from the centre". Every ordering is a
permutation, so the audio capacity does not change.

Droplets: one drop per note that stands out, placed on a fixed 12-hour clock (C at 12 o'clock,
one hour per semitone), size = how far the note stands out. The drops merge outwards. Time
starts on the outermost contour and ends inside the drops; all drops play the same moment.

The only music-dependent input is weights(): 12 integers, 4 bits each (6 bytes). The clock is
fixed, so the shape does not depend on the key detector. descriptor() is the older, key-
anchored description still used by the experimental shapes in scripts/explore_shapes.py.

NOTE: the landscape is computed in floating point. Its three discontinuous decisions are
made tolerant so that two implementations agree: the nearest drop (ties within 1e-9 go to the
first drop), the contour number (floor with a 1e-9 margin) and the sort (keys rounded to
1/65536 of a contour, ties by cell index). Symmetric drop layouts otherwise put whole lines of
tiles exactly on a tie, and Python and JavaScript then disagreed on 173 tiles of one clip. A guaranteed bit-exact format still needs an integer
definition of the landscape itself (roadmap step 10).
"""
from functools import lru_cache
import numpy as np
from .codec import NT, SIDE, T, BMOD, CANVAS, ORIGIN

TAU = 2*np.pi
RIPPLE = 8.0            # adopted: amplitude (grey levels) of the light/dark wave along the contours


def weights(ft):
    """12 integers 0..15 in fixed pitch order (C first): how far each note rises above the
    middle of the mix, (energy - median) / (strongest - median), 0 below the median."""
    cm = ft["chroma"].sum(1)
    med = np.median(cm)
    return np.round(15*np.clip((cm - med)/(cm.max() - med + 1e-30), 0, 1)).astype(int).tolist()

def descriptor(ft):
    """Experimental shapes only. 12 pitch-class weights (tonic first, sqrt of the energy
    share), brightness, tonality: integers 0..15 each."""
    cm = np.roll(ft["chroma"].sum(1), -ft["tonic"])
    return dict(w=np.round(15*np.sqrt(cm/cm.max())).astype(int).tolist(),
                bright=int(np.clip(round(15*(ft["brightness"] - 300)/2500), 0, 15)),
                tonal=int(np.clip(round(15*ft["tonality"]), 0, 15)))


# ---------------------------------------------------------------- geometry
def tile_xy():
    r, c = np.divmod(np.arange(NT), SIDE)
    return c - (SIDE - 1)/2, r - (SIDE - 1)/2               # tile units, origin at the centre

def pixel_xy():
    p = (np.arange(CANVAS) - ORIGIN + 0.5)/(T*BMOD) - SIDE/2
    return np.meshgrid(p, p)

def angle(x, y):
    return (np.arctan2(x, -y) % TAU)/TAU                    # turns, 0 at 12 o'clock, clockwise


# ---------------------------------------------------------------- landscapes: f(x, y) -> (h, angle)
def disc(w=None):
    return lambda x, y: (np.hypot(x, y), angle(x, y))

def droplets(w, ring=13.0, beta=0.45, most=6, floor=3):
    """w: 12 weights (0..15), see weights(). Drops for the `most` strongest notes with weight
    >= floor (3 of 15 = 0.2), on a circle of radius `ring` tiles; drop radius 2.5 .. 6.5 tiles;
    the drops are merged with a soft minimum (sharpness beta per tile). Angle = around the
    nearest drop."""
    w = np.asarray(w)
    ks = [int(k) for k in np.argsort(-w, kind="stable")[:most] if w[k] >= floor]
    if not ks:                                               # silence: no note stands out
        return disc()
    cx = np.array([ring*np.sin(TAU*k/12) for k in ks]); cy = np.array([-ring*np.cos(TAU*k/12) for k in ks])
    rad = np.array([1.5 + 5*w[k]/15 for k in ks])
    def raw(x, y):
        dist = np.stack([np.hypot(x - a, y - b) - r for a, b, r in zip(cx, cy, rad)])
        j = (dist <= dist.min(0) + TIE).argmax(0)            # nearest drop; exact ties go to the first
        return -np.log(np.exp(-beta*dist).sum(0))/beta, angle(x - cx[j], y - cy[j])
    h0 = raw(*tile_xy())[0].min()
    def f(x, y):
        h, th = raw(x, y)
        return h - h0, th
    return f

LANDSCAPES = {"disc": disc, "droplets": droplets}


# ---------------------------------------------------------------- landscape -> layout, ripple, time map
KEY_STEPS = 65536       # sort keys are compared as integers: 1/65536 of a contour
TIE = 1e-9              # decisions closer than this are settled by a fixed rule, not by rounding

def spiral_key(h, th):
    return np.floor(h - th + TIE) + th

def layout_from(f):
    """cells[t] = grid cell of tile t (a permutation); also the sorted keys."""
    key = np.round(spiral_key(*f(*tile_xy()))*KEY_STEPS)             # integer keys
    order = np.lexsort((np.arange(NT), key))                         # ties by cell index
    return order[::-1].copy(), key[order]/KEY_STEPS

def fields_from(f, ripple=0.0):
    """cells, under, tmap for a landscape f.
    under  ripple*cos(2 pi (h - angle)) on the canvas, or None: known to encoder and decoder
    tmap   tile-time of every canvas pixel, continuous: used by the encoder to draw colour"""
    cells, keys = layout_from(f)
    h, th = f(*pixel_xy())
    tmap = NT - 1 - np.interp(spiral_key(h, th), keys, np.arange(NT))
    return cells, (ripple*np.cos(TAU*(h - th)) if ripple else None), tmap

@lru_cache(maxsize=8)
def _build(name, w, ripple):
    return fields_from(LANDSCAPES[name](w), ripple)

def build(name="droplets", w=None, ripple=RIPPLE):
    """Everything derived from the header fields (shape name, 12 weights, ripple amplitude):
    cells and under are needed by encoder and decoder, tmap by the encoder only."""
    return _build(name, None if w is None else tuple(int(v) for v in w), float(ripple))
