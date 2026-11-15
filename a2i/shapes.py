"""Layouts whose shape is generated from the music (adopted: droplets).

A layout is only an ordering of the 4096 tiles. Define a landscape h(x, y) over the image and
let time run along its contour lines: key = floor(h - angle) + angle, tiles sorted by key.
The round spiral is the landscape "distance from the centre". Every ordering is a
permutation, so the audio capacity does not change.

Droplets: one drop per prominent pitch class, placed on a 12-hour clock (tonic at 12 o'clock,
one hour per semitone), size = how much that note is used. The drops merge outwards. Time
starts on the outermost contour and ends inside the drops; all drops play the same moment.

The only music-dependent input is the descriptor: 12 pitch-class weights, 4 bits each
(6 bytes). `brightness` and `tonality` are also returned for the experimental shapes in
scripts/shapes.py; droplets does not read them.

NOTE: the ordering uses float sort keys. It is deterministic on one machine; a bit-exact
cross-platform format needs an integer definition (roadmap step 10).
"""
from functools import lru_cache
import numpy as np
from .codec import NT, SIDE, T, BMOD, CANVAS, ORIGIN

TAU = 2*np.pi
RIPPLE = 8.0            # adopted: amplitude (grey levels) of the light/dark wave along the contours


def descriptor(ft):
    """12 pitch-class weights (tonic first), brightness, tonality: integers 0..15 each."""
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

def droplets(w, ring=13.0, beta=0.45, most=6, floor=8):
    """w: the 12 pitch-class weights (0..15). Drops for the `most` strongest notes with weight
    >= floor, on a circle of radius `ring` tiles; drop radius 1.5 .. 6.5 tiles; the drops are
    merged with a soft minimum (sharpness beta per tile). Angle = around the nearest drop."""
    w = np.asarray(w)
    ks = [int(k) for k in np.argsort(-w, kind="stable")[:most] if w[k] >= floor]
    cx = np.array([ring*np.sin(TAU*k/12) for k in ks]); cy = np.array([-ring*np.cos(TAU*k/12) for k in ks])
    rad = np.array([1.5 + 5*w[k]/15 for k in ks])
    def raw(x, y):
        dist = np.stack([np.hypot(x - a, y - b) - r for a, b, r in zip(cx, cy, rad)])
        j = dist.argmin(0)
        return -np.log(np.exp(-beta*dist).sum(0))/beta, angle(x - cx[j], y - cy[j])
    h0 = raw(*tile_xy())[0].min()
    def f(x, y):
        h, th = raw(x, y)
        return h - h0, th
    return f

LANDSCAPES = {"disc": disc, "droplets": droplets}


# ---------------------------------------------------------------- landscape -> layout, ripple, time map
def spiral_key(h, th):
    return np.floor(h - th) + th

def layout_from(f):
    """cells[t] = grid cell of tile t (a permutation); also the sorted keys."""
    key = spiral_key(*f(*tile_xy()))
    order = np.lexsort((np.arange(NT), key))
    return order[::-1].copy(), key[order]

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
