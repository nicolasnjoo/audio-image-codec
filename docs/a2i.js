/* a2i in the browser: a port of a2i/container.py (self-contained image, provisional version 1).
   One file for the page, its worker and Node tests. No dependencies.
     A2I.encode(mono Float32Array/Float64Array at 32 kHz) -> {rgba, info}
     A2I.decode(rgba, width, height)                      -> {audio, rate, info} or {error}
   The Python package is the reference; scripts/web_check.py compares the two. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.A2I = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------- constants (a2i/codec.py, container.py)
  const M = 64, F = 4032, NS = F * M;
  const CANVAS = 1080, INNER = 1024, G = 512, T = 8, SIDE = 64, NT = 4096, AC = 63;
  const NB = 16, NBLK = 32, ORIGIN = 28;
  const A_ENV = 60, A_RES = 75, DR = 6, RIPPLE = 8, KEY_STEPS = 65536;
  const VERSION = 1, RATES = [16000, 24000, 25600, 32000], RATE = 32000;
  const HBITS = 64, HCOPIES = 6, GBITS = 5, GSTEP = 0.5, THRESH = 128, MOD = 4, CELL = 4;
  const GLEN = NBLK * NB * GBITS + 16;
  const TAU = 2 * Math.PI, TIE = 1e-9;
  const FINDER = [[1,1,1,1,1,1,1],[1,0,0,0,0,0,1],[1,0,1,1,1,0,1],[1,0,1,1,1,0,1],[1,0,1,1,1,0,1],[1,0,0,0,0,0,1],[1,1,1,1,1,1,1]];
  const FC = [ORIGIN / 2, CANVAS - ORIGIN / 2];
  const KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  const NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
  const KCH = [[0.0, 1.402], [-0.344136, -0.714136], [1.772, 0.0]];      // (Cb, Cr) -> dR, dG, dB
  const WL = [0.299, 0.587, 0.114];

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  function rint(v) {                                         // round half to even, like numpy
    const r = Math.round(v);
    return (Math.abs(v % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
  }
  function median(a) {
    if (!a.length) return NaN;
    const s = Float64Array.from(a).sort();
    const n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }

  // zigzag order of the 63 AC slots
  const ZR = [], ZC = [];
  (function () {
    const z = [];
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) z.push([i, j]);
    z.sort((p, q) => (p[0] + p[1]) - (q[0] + q[1]) ||
      (((p[0] + p[1]) % 2 === 0 ? p[1] : -p[1]) - ((q[0] + q[1]) % 2 === 0 ? q[1] : -q[1])));
    for (let k = 1; k < 64; k++) { ZR.push(z[k][0]); ZC.push(z[k][1]); }
  })();

  // orthonormal 8-point DCT basis: D[u][i] = c(u) cos((2i+1) u pi / 16)
  const D8 = new Float64Array(64);
  for (let u = 0; u < 8; u++) for (let i = 0; i < 8; i++)
    D8[u * 8 + i] = Math.sqrt((u ? 2 : 1) / 8) * Math.cos((2 * i + 1) * u * Math.PI / 16);

  // ---------------------------------------------------------------- MDCT (a2i/mdct.py)
  const BASIS = new Float64Array(M * 2 * M), WIN = new Float64Array(2 * M);
  for (let n = 0; n < 2 * M; n++) {
    WIN[n] = Math.sin(Math.PI / (2 * M) * (n + 0.5));
    for (let k = 0; k < M; k++) BASIS[k * 2 * M + n] = Math.cos(Math.PI / M * (n + 0.5 + M / 2) * (k + 0.5));
  }
  function analyze(x) {
    const X = new Float64Array(F * M), fr = new Float64Array(2 * M);
    for (let i = 0; i < F; i++) {
      for (let n = 0; n < 2 * M; n++) { const j = i * M + n - M; fr[n] = (j >= 0 && j < NS) ? x[j] * WIN[n] : 0; }
      for (let k = 0; k < M; k++) {
        let s = 0; const b = k * 2 * M;
        for (let n = 0; n < 2 * M; n++) s += fr[n] * BASIS[b + n];
        X[i * M + k] = s;
      }
    }
    return X;
  }
  function synthesize(X) {
    const y = new Float64Array((F + 2) * M);
    for (let i = 0; i < F; i++) {
      for (let n = 0; n < 2 * M; n++) {
        let s = 0;
        for (let k = 0; k < M; k++) s += X[i * M + k] * BASIS[k * 2 * M + n];
        y[i * M + n] += s * (2 / M) * WIN[n];
      }
    }
    return y.slice(M, M + NS);
  }

  // ---------------------------------------------------------------- whitening gains
  function gainsOf(X) {
    const g = new Float64Array(NBLK * NB), per = F / NBLK, bw = M / NB;
    for (let b = 0; b < NBLK; b++) for (let c = 0; c < NB; c++) {
      let s = 0;
      for (let i = b * per; i < (b + 1) * per; i++) for (let k = c * bw; k < (c + 1) * bw; k++) s += X[i * M + k] * X[i * M + k];
      g[b * NB + c] = Math.sqrt(s / (per * bw)) + 1e-12;
    }
    return g;
  }
  function quantiseGains(g) {
    let mx = -Infinity;
    const lg = Float64Array.from(g, Math.log2);
    for (const v of lg) if (v > mx) mx = v;
    const q = new Int32Array(g.length), used = new Float64Array(g.length);
    for (let i = 0; i < g.length; i++) {
      q[i] = clamp(rint((mx - lg[i]) / GSTEP), 0, (1 << GBITS) - 1);
      used[i] = Math.pow(2, mx - GSTEP * q[i]);
    }
    return { q, used };
  }
  const gainAt = (g, frame, bin) => g[Math.floor(frame / (F / NBLK)) * NB + Math.floor(bin / (M / NB))];

  // ---------------------------------------------------------------- features (a2i/features.py)
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -TAU / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }
  function features(x, fs) {
    const hop = rint(256 * fs / 16000), N = 16 * hop;
    if (N & (N - 1)) throw new Error("features: the sample rate must give a power-of-two window");
    const nb = N / 2 + 1, nf = Math.floor(x.length / hop) + 1;
    const win = new Float64Array(N); let wsum = 0;
    for (let n = 0; n < N; n++) { win[n] = 0.5 - 0.5 * Math.cos(TAU * n / N); wsum += win[n]; }
    const f = new Float64Array(nb), pc = new Int32Array(nb);
    for (let b = 0; b < nb; b++) {
      f[b] = b * fs / N;
      pc[b] = f[b] > 0 ? ((rint(12 * Math.log2(f[b] / 261.6256)) % 12) + 12) % 12 : 0;
    }
    const chroma = new Float64Array(12 * nf), flux = new Float64Array(nf), centroid = new Float64Array(nf),
      flat = new Float64Array(nf), rms = new Float64Array(nf);
    const re = new Float64Array(N), im = new Float64Array(N), prev = new Float64Array(nb);
    for (let m = 0; m < nf; m++) {
      const s0 = m * hop - N / 2;
      for (let n = 0; n < N; n++) { const j = s0 + n; re[n] = (j >= 0 && j < x.length) ? x[j] * win[n] : 0; im[n] = 0; }
      fft(re, im);
      let e = 0, cs = 0, cw = 0, lsum = 0, cnt = 0, fl = 0;
      for (let b = 0; b < nb; b++) {
        const S = Math.hypot(re[b], im[b]) / wsum, fb = f[b];
        e += S * S;
        if (fb >= 55 && fb <= 4000) chroma[pc[b] * nf + m] += S * S;
        if (fb > 30 && fb < 7500) {
          const l = Math.log1p(S * 50);
          if (m > 0 && l > prev[b]) fl += l - prev[b];
          prev[b] = l;
        }
        if (fb > 60 && fb < 7500) { const v = S + 1e-9; cs += fb * v; cw += v; lsum += Math.log(v); cnt++; }
      }
      flux[m] = fl; rms[m] = Math.sqrt(e / nb);
      centroid[m] = cs / cw; flat[m] = Math.exp(lsum / cnt) / (cw / cnt);
    }
    const cm = new Float64Array(12); let tot = 0;
    for (let k = 0; k < 12; k++) { for (let m = 0; m < nf; m++) cm[k] += chroma[k * nf + m]; tot += cm[k]; }
    const cn = Array.from(cm, v => v / tot);
    const corr = (a, b) => {
      const ma = a.reduce((p, v) => p + v, 0) / 12, mb = b.reduce((p, v) => p + v, 0) / 12;
      let sab = 0, saa = 0, sbb = 0;
      for (let i = 0; i < 12; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
      return sab / Math.sqrt(saa * sbb);
    };
    let best = null;
    for (const [prof, mode] of [[KK_MAJ, "major"], [KK_MIN, "minor"]]) for (let k = 0; k < 12; k++) {
      const c = corr(cn, prof.map((_, i) => prof[((i - k) % 12 + 12) % 12]));
      if (best === null || c > best.c || (c === best.c && (k > best.k || (k === best.k && mode > best.mode)))) best = { c, k, mode };
    }
    if (!(tot > 0)) best = { c: 0, k: 0, mode: "major" };
    let rmax = 0; for (const v of rms) if (v > rmax) rmax = v;
    const af = [], ac = [];
    for (let m = 0; m < nf; m++) if (rms[m] > rmax * 0.02) { af.push(flat[m]); ac.push(centroid[m]); }
    return {
      hop, nf, fs, chroma, cm, rms, tonic: best.k, mode: best.mode, key: NAMES[best.k] + " " + best.mode,
      tonality: af.length ? 1 - median(af) : 0, brightness: ac.length ? median(ac) : 0
    };
  }
  function palette(ft) {
    const rel = (ft.tonic + (ft.mode === "minor" ? 3 : 0)) % 12;
    const h0 = ((7 * rel) % 12) / 12 + 0.04, sgn = ft.mode === "major" ? 1 : -1;
    const spread = 0.10 + 0.12 * clamp((ft.brightness - 400) / 2600, 0, 1);
    return { h0: h0 % 1, h1: (((h0 + sgn * spread) % 1) + 1) % 1, sgn, sat: clamp(0.45 + 0.55 * ft.tonality, 0.45, 1), rel };
  }
  function weights(ft) {                                     // a2i/shapes.py weights()
    const med = median(ft.cm); let mx = -Infinity;
    for (const v of ft.cm) if (v > mx) mx = v;
    return Array.from(ft.cm, v => rint(15 * clamp((v - med) / (mx - med + 1e-30), 0, 1)));
  }

  // ---------------------------------------------------------------- layout from the notes (a2i/shapes.py)
  const angle = (x, y) => { let a = Math.atan2(x, -y); if (a < 0) a += TAU; return a / TAU; };
  const _shapeCache = new Map();
  function buildShape(w, ripple, wantPixels) {
    const id = w.join(",") + "|" + ripple + "|" + (wantPixels ? 1 : 0);
    if (_shapeCache.has(id)) return _shapeCache.get(id);
    const ks = [...w.keys()].sort((a, b) => (w[b] - w[a]) || (a - b)).slice(0, 6).filter(k => w[k] >= 3);
    const ring = 13, beta = 0.45;
    const cx = ks.map(k => ring * Math.sin(TAU * k / 12)), cy = ks.map(k => -ring * Math.cos(TAU * k / 12));
    const rad = ks.map(k => 1.5 + 5 * w[k] / 15);
    const out = [0, 0], dd = new Float64Array(Math.max(1, ks.length));
    const raw = ks.length ? (x, y) => {
      let s = 0, dmin = Infinity, j = 0;
      for (let k = 0; k < ks.length; k++) {
        const d = Math.hypot(x - cx[k], y - cy[k]) - rad[k];
        dd[k] = d; if (d < dmin) dmin = d;
        s += Math.exp(-beta * d);
      }
      while (dd[j] > dmin + TIE) j++;                        // nearest drop; exact ties go to the first
      out[0] = -Math.log(s) / beta; out[1] = angle(x - cx[j], y - cy[j]);
    } : (x, y) => { out[0] = Math.hypot(x, y); out[1] = angle(x, y); };
    const h = new Float64Array(NT), th = new Float64Array(NT);
    let h0 = Infinity;
    for (let c = 0; c < NT; c++) {
      raw((c % SIDE) - (SIDE - 1) / 2, Math.floor(c / SIDE) - (SIDE - 1) / 2);
      h[c] = out[0]; th[c] = out[1];
      if (out[0] < h0) h0 = out[0];
    }
    if (!ks.length) h0 = 0;
    const key = new Float64Array(NT);
    for (let c = 0; c < NT; c++) key[c] = rint((Math.floor(h[c] - h0 - th[c] + TIE) + th[c]) * KEY_STEPS);
    const order = [...key.keys()].sort((a, b) => (key[a] - key[b]) || (a - b));
    const cells = new Int32Array(NT), keys = new Float64Array(NT);
    for (let i = 0; i < NT; i++) { cells[NT - 1 - i] = order[i]; keys[i] = key[order[i]] / KEY_STEPS; }
    const res = { cells, under: null, tmap: null };
    if (wantPixels) {
      const under = new Float32Array(CANVAS * CANVAS), tmap = new Float32Array(CANVAS * CANVAS);
      for (let py = 0; py < CANVAS; py++) {
        const y = (py - ORIGIN + 0.5) / 16 - SIDE / 2;
        for (let px = 0; px < CANVAS; px++) {
          raw((px - ORIGIN + 0.5) / 16 - SIDE / 2, y);
          const hh = out[0] - h0, t = out[1], sk = Math.floor(hh - t + TIE) + t;
          under[py * CANVAS + px] = ripple * Math.cos(TAU * (hh - t));
          let lo = 0, hi = NT - 1, r;                        // np.interp(sk, keys, 0..NT-1)
          if (sk <= keys[0]) r = 0;
          else if (sk >= keys[NT - 1]) r = NT - 1;
          else {
            while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (keys[mid] <= sk) lo = mid; else hi = mid; }
            r = keys[hi] > keys[lo] ? lo + (sk - keys[lo]) / (keys[hi] - keys[lo]) : lo;
          }
          tmap[py * CANVAS + px] = NT - 1 - r;
        }
      }
      res.under = under; res.tmap = tmap;
    }
    if (_shapeCache.size > 6) _shapeCache.clear();
    _shapeCache.set(id, res);
    return res;
  }

  // ---------------------------------------------------------------- colour helpers (a2i/looks.py)
  function oklchCbCr(L, C, hTurn) {
    const h = TAU * (hTurn + 29 / 360), a = C * Math.cos(h), b = C * Math.sin(h);
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
      s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
    const lin = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
    const g = lin.map(v => 255 * Math.sign(v) * (Math.abs(v) <= 0.0031308 ? 12.92 * Math.abs(v) : 1.055 * Math.pow(Math.abs(v), 1 / 2.4) - 0.055));
    return [-0.168736 * g[0] - 0.331264 * g[1] + 0.5 * g[2], 0.5 * g[0] - 0.418688 * g[1] - 0.081312 * g[2]];
  }
  function brightnessFor(pal) {
    const [cb, cr] = oklchCbCr(0.62, 0.15, pal.h0), n = Math.hypot(cb, cr);
    const d = KCH.map(k => (k[0] * cb + k[1] * cr) / n);
    let best = -Infinity, by = 128;
    for (let y = 40; y < 216; y++) {
      let room = Infinity;
      for (const v of d) if (Math.abs(v) > 1e-9) room = Math.min(room, v > 0 ? (255 - 16 - y) / v : (y - 16) / -v);
      if (room > best) { best = room; by = y; }
    }
    const centre = clamp(by, 128, 165);
    return { centre, sign: centre > 128 ? -1 : 1 };
  }
  function gaussKernel(sigma) {
    const r = Math.floor(4 * sigma + 0.5), k = new Float64Array(2 * r + 1); let s = 0;
    for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-0.5 * i * i / (sigma * sigma)); s += k[i + r]; }
    for (let i = 0; i < k.length; i++) k[i] /= s;
    return { k, r };
  }
  function gauss1d(src, n, stride, offset, sigma, reflect, dst) {  // one line of length n
    const { k, r } = gaussKernel(sigma);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let t = -r; t <= r; t++) {
        let j = i + t;
        if (reflect) { while (j < 0 || j >= n) j = j < 0 ? -j - 1 : 2 * n - 1 - j; }
        else j = j < 0 ? 0 : j >= n ? n - 1 : j;
        s += k[t + r] * src[offset + j * stride];
      }
      dst[i] = s;
    }
  }
  function blur2d(a, n, sigma) {                             // gaussian_filter(a, sigma), mode reflect
    const line = new Float64Array(n);
    for (let y = 0; y < n; y++) { gauss1d(a, n, 1, y * n, sigma, true, line); for (let x = 0; x < n; x++) a[y * n + x] = line[x]; }
    for (let x = 0; x < n; x++) { gauss1d(a, n, n, x, sigma, true, line); for (let y = 0; y < n; y++) a[y * n + x] = line[y]; }
  }
  function window6(a, n, fn) {                               // size-6 filter, offsets -3..+2, mode reflect, in place
    const line = new Float64Array(n), tmp = new Float64Array(n);
    const run = (stride, off) => {
      for (let i = 0; i < n; i++) tmp[i] = a[off + i * stride];
      for (let i = 0; i < n; i++) {
        let acc = fn === "min" ? Infinity : 0;
        for (let t = -3; t <= 2; t++) {
          let j = i + t; if (j < 0) j = -j - 1; else if (j >= n) j = 2 * n - 1 - j;
          acc = fn === "min" ? Math.min(acc, tmp[j]) : acc + tmp[j];
        }
        line[i] = fn === "min" ? acc : acc / 6;
      }
      for (let i = 0; i < n; i++) a[off + i * stride] = line[i];
    };
    for (let y = 0; y < n; y++) run(1, y * n);
    for (let x = 0; x < n; x++) run(n, x);
  }

  // ---------------------------------------------------------------- envelope field (a2i/codec.py)
  let _field = null;
  function fieldMatrices() {
    if (_field) return _field;
    const t0 = new Int32Array(G), b = new Float64Array(G), K = new Float64Array(SIDE * SIDE);
    for (let m = 0; m < G; m++) {
      const p = (m + 0.5) / T - 0.5;
      t0[m] = clamp(Math.floor(p), 0, SIDE - 2); b[m] = clamp(p - t0[m], 0, 1);
      const row = Math.floor(m / T);
      K[row * SIDE + t0[m]] += (1 - b[m]) / T; K[row * SIDE + t0[m] + 1] += b[m] / T;
    }
    const n = SIDE, A = Float64Array.from(K), Ki = new Float64Array(n * n);   // Gauss-Jordan inverse
    for (let i = 0; i < n; i++) Ki[i * n + i] = 1;
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[p * n + c])) p = r;
      if (p !== c) for (let j = 0; j < n; j++) {
        let t = A[c * n + j]; A[c * n + j] = A[p * n + j]; A[p * n + j] = t;
        t = Ki[c * n + j]; Ki[c * n + j] = Ki[p * n + j]; Ki[p * n + j] = t;
      }
      const d = A[c * n + c];
      for (let j = 0; j < n; j++) { A[c * n + j] /= d; Ki[c * n + j] /= d; }
      for (let r = 0; r < n; r++) if (r !== c) {
        const fct = A[r * n + c];
        if (fct !== 0) for (let j = 0; j < n; j++) { A[r * n + j] -= fct * A[c * n + j]; Ki[r * n + j] -= fct * Ki[c * n + j]; }
      }
    }
    return (_field = { t0, b, Ki });
  }
  function envelopeField(E) {                                // (64x64 tile means) -> (512x512), same tile means
    const { t0, b, Ki } = fieldMatrices(), n = SIDE;
    const A = new Float64Array(n * n), Gm = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < n; k++) s += Ki[i * n + k] * E[k * n + j]; A[i * n + j] = s; }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < n; k++) s += A[i * n + k] * Ki[j * n + k]; Gm[i * n + j] = s; }
    const out = new Float64Array(G * G);
    for (let y = 0; y < G; y++) {
      const r0 = t0[y], by = b[y];
      for (let x = 0; x < G; x++) {
        const c0 = t0[x], bx = b[x];
        out[y * G + x] = (1 - by) * ((1 - bx) * Gm[r0 * n + c0] + bx * Gm[r0 * n + c0 + 1]) +
          by * ((1 - bx) * Gm[(r0 + 1) * n + c0] + bx * Gm[(r0 + 1) * n + c0 + 1]);
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- ring bits (a2i/container.py)
  function crc(bits, width, poly) {
    let c = 0; const top = 1 << (width - 1), mask = (1 << width) - 1;
    for (const b of bits) { c ^= b << (width - 1); c = (c & top) ? ((c << 1) ^ poly) & mask : (c << 1) & mask; }
    const out = [];
    for (let k = width - 1; k >= 0; k--) out.push((c >> k) & 1);
    return out;
  }
  const field = (v, n) => { const o = []; for (let k = n - 1; k >= 0; k--) o.push((v >> k) & 1); return o; };
  const value = bits => bits.reduce((p, b) => (p << 1) | b, 0);
  function headerBits(rate, sign, w) {
    let body = field(VERSION, 4).concat(field(RATES.indexOf(rate), 2), [sign < 0 ? 1 : 0, 0]);
    for (const v of w) body = body.concat(field(v, 4));
    return body.concat(crc(body, 8, 0x07));
  }
  function parseHeader(bits) {
    const c = crc(bits.slice(0, 56), 8, 0x07);
    for (let i = 0; i < 8; i++) if (c[i] !== bits[56 + i]) return null;
    if (value(bits.slice(0, 4)) !== VERSION) return null;
    const w = []; for (let k = 0; k < 12; k++) w.push(value(bits.slice(8 + 4 * k, 12 + 4 * k)));
    return { rate: RATES[value(bits.slice(4, 6))], sign: bits[6] ? -1 : 1, w };
  }
  function gainBits(q) {
    let body = [];
    for (const v of q) body = body.concat(field(v, GBITS));
    return body.concat(crc(body, 16, 0x1021));
  }
  let _ring = null;
  function ring() {
    if (_ring) return _ring;
    const cells = [], span = [];
    for (let v = ORIGIN + 8; v < CANVAS - ORIGIN - 8; v += CELL) span.push(v);
    for (let k = 0; k < 4; k++) for (const x of span) cells.push([4 + CELL * k, x]);
    for (let k = 0; k < 4; k++) for (const y of span) cells.push([y, CANVAS - 8 - CELL * k]);
    for (let k = 0; k < 4; k++) for (const x of span) cells.push([CANVAS - 8 - CELL * k, x]);
    for (let k = 0; k < 4; k++) for (const y of span) cells.push([y, 4 + CELL * k]);
    const n = cells.length, mask = new Uint8Array(n);
    let s = 1;
    for (let i = 0; i < n; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; mask[i] = (s >>> 16) & 1; }
    const nh = HBITS * HCOPIES, hpos = new Map();
    for (let k = 0; k < nh; k++) hpos.set(Math.floor(k * n / nh), k % HBITS);
    const kind = new Uint8Array(n), bit = new Int32Array(n);
    for (let i = 0, g = 0; i < n; i++) {
      if (hpos.has(i)) { kind[i] = 0; bit[i] = hpos.get(i); } else { kind[i] = 1; bit[i] = g % GLEN; g++; }
    }
    return (_ring = { cells, mask, kind, bit });
  }
  function drawFrame(rgba, hbits, gbits) {
    const put = (y, x, v) => { const o = 4 * (y * CANVAS + x); rgba[o] = rgba[o + 1] = rgba[o + 2] = v; rgba[o + 3] = 255; };
    for (let y = 0; y < CANVAS; y++) for (let x = 0; x < CANVAS; x++)
      if (y < 20 || y >= CANVAS - 20 || x < 20 || x >= CANVAS - 20) put(y, x, 255);
    const { cells, mask, kind, bit } = ring();
    for (let i = 0; i < cells.length; i++) {
      if ((kind[i] === 0 ? hbits[bit[i]] : gbits[bit[i]]) ^ mask[i])
        for (let dy = 0; dy < CELL; dy++) for (let dx = 0; dx < CELL; dx++) put(cells[i][0] + dy, cells[i][1] + dx, 0);
    }
    const n = 7 * MOD;
    for (const y0 of [0, CANVAS - n]) for (const x0 of [0, CANVAS - n])
      for (let dy = 0; dy < n; dy++) for (let dx = 0; dx < n; dx++)
        put(y0 + dy, x0 + dx, FINDER[Math.floor(dy / MOD)][Math.floor(dx / MOD)] ? 0 : 255);
  }
  function readRing(L) {                                     // L: canonical luma, CANVAS x CANVAS
    const { cells, mask, kind, bit } = ring();
    const sh = new Float64Array(HBITS), sg = new Float64Array(GLEN);
    for (let i = 0; i < cells.length; i++) {
      const [y, x] = cells[i];
      let v = (L[(y + 1) * CANVAS + x + 1] + L[(y + 1) * CANVAS + x + 2] + L[(y + 2) * CANVAS + x + 1] + L[(y + 2) * CANVAS + x + 2]) / 4 - THRESH;
      if (mask[i]) v = -v;
      if (kind[i] === 0) sh[bit[i]] += v; else sg[bit[i]] += v;
    }
    return { hb: Array.from(sh, v => (v < 0 ? 1 : 0)), gb: Array.from(sg, v => (v < 0 ? 1 : 0)) };
  }

  // ---------------------------------------------------------------- encoder
  function prepare(mono, start) {                            // NS samples from `start`, peak-normalised
    const x = new Float64Array(NS); let pk = 0;
    for (let i = 0; i < NS; i++) { const j = start + i; x[i] = j < mono.length ? mono[j] : 0; if (Math.abs(x[i]) > pk) pk = Math.abs(x[i]); }
    if (pk > 0) for (let i = 0; i < NS; i++) x[i] /= pk * 1.02;
    return x;
  }
  function encode(mono, opts) {
    opts = opts || {};
    const rate = opts.rate || RATE, progress = opts.progress || (() => {});
    const x = prepare(mono, opts.start || 0);
    progress("transform");
    const X = analyze(x);
    const { q, used } = quantiseGains(gainsOf(X));
    // whiten, pack 63 per tile, normalise each tile to its peak
    const lp = new Float64Array(NT), R = new Float64Array(NT * AC);
    let ref = -Infinity;
    for (let t = 0; t < NT; t++) {
      let pk = 0;
      for (let s = 0; s < AC; s++) {
        const fl = t * AC + s, c = X[fl] / gainAt(used, Math.floor(fl / M), fl % M);
        R[fl] = c; if (Math.abs(c) > pk) pk = Math.abs(c);
      }
      lp[t] = Math.log2(pk + 1e-12); if (lp[t] > ref) ref = lp[t];
    }
    const env = new Float64Array(NT);
    for (let t = 0; t < NT; t++) {
      const l = clamp(lp[t], ref - DR, ref), sc = Math.pow(2, l);
      env[t] = 2 * (l - (ref - DR)) / DR - 1;
      for (let s = 0; s < AC; s++) R[t * AC + s] = clamp(R[t * AC + s] / sc, -1, 1);
    }
    progress("listening");
    const ft = features(x, rate), pal = palette(ft), w = weights(ft);
    progress("shape");
    const shape = buildShape(w, RIPPLE, true), cells = shape.cells;
    const { centre, sign } = brightnessFor(pal), aEnv = sign * A_ENV;
    const shift = rint(centre - 128 - aEnv * median(env));
    progress("drawing");
    // modulate: texture per tile (inverse DCT of the 63 AC slots) + smooth envelope field
    const P = new Float64Array(G * G), V = new Float64Array(64), tmp = new Float64Array(64), E = new Float64Array(NT);
    for (let t = 0; t < NT; t++) {
      V.fill(0);
      for (let s = 0; s < AC; s++) V[ZR[s] * 8 + ZC[s]] = A_RES * R[t * AC + s];
      for (let u = 0; u < 8; u++) for (let j = 0; j < 8; j++) { let a = 0; for (let v = 0; v < 8; v++) a += V[u * 8 + v] * D8[v * 8 + j]; tmp[u * 8 + j] = a; }
      const cell = cells[t], r0 = Math.floor(cell / SIDE) * 8, c0 = (cell % SIDE) * 8;
      for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { let a = 0; for (let u = 0; u < 8; u++) a += D8[u * 8 + i] * tmp[u * 8 + j]; P[(r0 + i) * G + c0 + j] = a; }
      E[cell] = aEnv * env[t] + shift;
    }
    const fld = envelopeField(E);
    const Y = new Float64Array(CANVAS * CANVAS).fill(128);
    let clipped = 0;
    for (let my = 0; my < G; my++) for (let mx = 0; mx < G; mx++) {
      const base = 128 + P[my * G + mx] + fld[my * G + mx];
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const o = (ORIGIN + 2 * my + dy) * CANVAS + ORIGIN + 2 * mx + dx, v = base + shape.under[o];
        if (v < 0 || v > 255) clipped++;
        Y[o] = clamp(v, 0, 255);
      }
    }
    progress("colour");
    // colour: per tile along time, then drawn continuously along the time path
    const nf = ft.nf, lmaxv = (() => { let m = -Infinity; for (const v of ft.rms) { const l = 20 * Math.log10(v + 1e-9); if (l > m) m = l; } return m; })();
    const chs = new Float64Array(12 * nf), line = new Float64Array(nf);
    for (let k = 0; k < 12; k++) { gauss1d(ft.chroma, nf, 1, k * nf, 10, false, line); chs.set(line, k * nf); }
    const turn = new Float64Array(nf), mag = new Float64Array(nf);
    for (let m = 0; m < nf; m++) {
      let zr = 0, zi = 0, tot = 0;
      for (let k = 0; k < 12; k++) {
        const a = TAU * ((((7 * (k - pal.rel)) % 12) + 12) % 12) / 12, c = chs[k * nf + m];
        zr += c * Math.cos(a); zi += c * Math.sin(a); tot += c;
      }
      zr /= tot + 1e-12; zi /= tot + 1e-12;
      turn[m] = (Math.atan2(zi, zr) - Math.PI / 4) / TAU; mag[m] = Math.hypot(zr, zi);
    }
    const jt = new Int32Array(NT), loudRaw = new Float64Array(NT), loud = new Float64Array(NT);
    for (let t = 0; t < NT; t++) {
      jt[t] = Math.min(Math.floor(63 * t / ft.hop), nf - 1);
      loudRaw[t] = clamp((20 * Math.log10(ft.rms[jt[t]] + 1e-9) - (lmaxv - 42)) / 42, 0, 1);
    }
    gauss1d(loudRaw, NT, 1, 0, 6, false, loud);
    const cbT = new Float64Array(NT), crT = new Float64Array(NT);
    for (let t = 0; t < NT; t++) {
      const hue = pal.h0 + 0.45 * turn[jt[t]] + pal.sgn * 0.06 * (loud[t] - 0.5);
      const v = (0.45 + 0.55 * clamp(mag[jt[t]] / 0.64, 0, 1)) * (0.4 + 0.6 * loud[t]);
      const c = oklchCbCr(0.62, 0.03 + 0.17 * v, hue);
      cbT[t] = c[0]; crT[t] = c[1];
    }
    const cb = new Float64Array(CANVAS * CANVAS), cr = new Float64Array(CANVAS * CANVAS);
    for (let i = 0; i < cb.length; i++) {
      const tm = shape.tmap[i], i0 = Math.min(Math.floor(tm), NT - 2), fr = tm - i0;
      cb[i] = cbT[i0] + fr * (cbT[i0 + 1] - cbT[i0]); cr[i] = crT[i0] + fr * (crT[i0 + 1] - crT[i0]);
    }
    blur2d(cb, CANVAS, 5); blur2d(cr, CANVAS, 5);
    // colorize: chroma scaled to stay inside the RGB cube here and in the 6 px neighbourhood, luma-exact rounding
    const sc = new Float64Array(CANVAS * CANVAS);
    for (let i = 0; i < sc.length; i++) {
      let s = Infinity;
      for (let c = 0; c < 3; c++) {
        const d = KCH[c][0] * cb[i] + KCH[c][1] * cr[i];
        if (d > 1e-9) s = Math.min(s, Math.max(0, 255 - 16 - Y[i]) / d);
        else if (d < -1e-9) s = Math.min(s, Math.max(0, Y[i] - 16) / -d);
      }
      sc[i] = clamp(s, 0, 1);
    }
    const nbh = Float64Array.from(sc);
    window6(nbh, CANVAS, "min"); window6(nbh, CANVAS, "mean");
    const rgba = new Uint8ClampedArray(CANVAS * CANVAS * 4), f3 = [0, 0, 0], lo = [0, 0, 0];
    for (let i = 0; i < sc.length; i++) {
      const s = Math.min(sc[i], nbh[i]), y = Y[i];
      for (let c = 0; c < 3; c++) { f3[c] = y + s * (KCH[c][0] * cb[i] + KCH[c][1] * cr[i]); lo[c] = Math.floor(f3[c]); }
      let be = Infinity, br = 0, bg = 0, bb = 0;
      for (let m = 0; m < 8; m++) {
        const r = clamp(lo[0] + ((m >> 2) & 1), 0, 255), g = clamp(lo[1] + ((m >> 1) & 1), 0, 255), b = clamp(lo[2] + (m & 1), 0, 255);
        const err = Math.abs(r * WL[0] + g * WL[1] + b * WL[2] - y) + 1e-3 * ((r - f3[0]) ** 2 + (g - f3[1]) ** 2 + (b - f3[2]) ** 2);
        if (err < be) { be = err; br = r; bg = g; bb = b; }
      }
      rgba[4 * i] = br; rgba[4 * i + 1] = bg; rgba[4 * i + 2] = bb; rgba[4 * i + 3] = 255;
    }
    drawFrame(rgba, headerBits(rate, sign, w), gainBits(q));
    const drops = [...w.keys()].sort((a, b) => (w[b] - w[a]) || (a - b)).slice(0, 6).filter(k => w[k] >= 3).map(k => NAMES[k]);
    return { rgba, x, info: { rate, seconds: NS / rate, key: ft.key, weights: w, drops, loudIsDark: sign < 0, clipped: 100 * clipped / (INNER * INNER) } };
  }

  // ---------------------------------------------------------------- register: find the image in a picture
  function findFinders(L, W, H) {
    const B = new Uint8Array(W * H);
    for (let i = 0; i < B.length; i++) B[i] = L[i] < THRESH ? 1 : 0;
    const mLo = MOD * 0.7, mHi = MOD * 2.2, cands = [];
    const starts = new Int32Array(W + 2);
    const colCheck = (y, x, m) => {
      if (!B[y * W + x]) return null;
      let y0 = y; while (y0 > 0 && B[(y0 - 1) * W + x]) y0--;
      let y1 = y; while (y1 < H - 1 && B[(y1 + 1) * W + x]) y1++;
      const b2 = y1 - y0 + 1;
      let a = y0 - 1, wu = 0; while (a >= 0 && !B[a * W + x]) { wu++; a--; }
      let c = y1 + 1, wd = 0; while (c < H && !B[c * W + x]) { wd++; c++; }
      if (a < 0 || c >= H) return null;
      const mv = (wu + b2 + wd) / 5;
      if (Math.abs(wu - mv) <= 0.5 * mv + 0.5 && Math.abs(wd - mv) <= 0.5 * mv + 0.5 && Math.abs(b2 - 3 * mv) <= mv + 0.5 &&
        Math.abs(mv - m) <= 0.35 * m + 0.5) return [y0 + b2 / 2, mv];
      return null;
    };
    for (let y = 0; y < H; y++) {
      let n = 0; starts[n++] = 0;
      for (let x = 1; x < W; x++) if (B[y * W + x] !== B[y * W + x - 1]) starts[n++] = x;
      starts[n] = W;
      for (let i = 0; i + 4 < n; i++) {
        if (!B[y * W + starts[i]]) continue;
        const b1 = starts[i + 1] - starts[i], w1 = starts[i + 2] - starts[i + 1], b2 = starts[i + 3] - starts[i + 2],
          w2 = starts[i + 4] - starts[i + 3], b3 = starts[i + 5] - starts[i + 4], m = (w1 + b2 + w2) / 5;
        if (m >= mLo && m <= mHi && Math.abs(w1 - m) <= 0.5 * m + 0.5 && Math.abs(w2 - m) <= 0.5 * m + 0.5 &&
          Math.abs(b2 - 3 * m) <= m + 0.5 && b1 >= 0.5 * m && b3 >= 0.5 * m) {
          const xc = starts[i + 2] + b2 / 2, v = colCheck(y, Math.floor(xc), m);
          if (v) cands.push([xc, v[0], (m + v[1]) / 2]);
        }
      }
    }
    const clusters = [];
    for (const [x, y, m] of cands) {
      let hit = false;
      for (const c of clusters) if (Math.abs(c[0] / c[3] - x) <= 1.5 * m && Math.abs(c[1] / c[3] - y) <= 1.5 * m) { c[0] += x; c[1] += y; c[2] += m; c[3]++; hit = true; break; }
      if (!hit) clusters.push([x, y, m, 1]);
    }
    const P = clusters.filter(c => c[3] >= 3).map(c => [c[0] / c[3], c[1] / c[3], c[2] / c[3], c[3]]);
    const span = (FC[1] - FC[0]) / MOD;
    let best = null;
    for (const a of P) for (const b of P) {
      const side = b[0] - a[0];
      if (side <= 0 || Math.abs(b[1] - a[1]) > 0.02 * side + 3) continue;
      const m = side / span;
      if (Math.abs(a[2] - m) > 0.3 * m || Math.abs(b[2] - m) > 0.3 * m) continue;
      const tol = 0.02 * side + 3;
      const pick = (px, py) => { let o = null; for (const p of P) if (Math.abs(p[0] - px) <= tol && Math.abs(p[1] - py) <= tol && (!o || p[3] > o[3])) o = p; return o; };
      const c = pick(a[0], a[1] + side), d = pick(b[0], b[1] + side);
      if (c && d) { const score = a[3] + b[3] + c[3] + d[3]; if (!best || score > best[0]) best = [score, [a, b, c, d]]; }
    }
    if (!best) return null;
    const refine = (x0, y0, m) => {
      const centre = (get, n, c0) => {
        const c = Math.round(c0);
        if (!(c > 0 && c < n - 1) || get(c) >= THRESH) return null;
        let lo = c; while (lo > 0 && get(lo - 1) < THRESH) lo--;
        let hi = c; while (hi < n - 1 && get(hi + 1) < THRESH) hi++;
        if (lo === 0 || hi === n - 1) return null;
        const left = (lo - 1) + (get(lo - 1) - THRESH) / (get(lo - 1) - get(lo)), right = hi + (get(hi) - THRESH) / (get(hi) - get(hi + 1));
        return (left + right) / 2 + 0.5;
      };
      const r = Math.max(1, Math.round(m)), xs = [], ys = [], yr = Math.round(y0), xr = Math.round(x0);
      for (let y = yr - r; y <= yr + r; y++) if (y >= 0 && y < H) { const v = centre(i => L[y * W + i], W, x0); if (v !== null) xs.push(v); }
      for (let x = xr - r; x <= xr + r; x++) if (x >= 0 && x < W) { const v = centre(i => L[i * W + x], H, y0); if (v !== null) ys.push(v); }
      return [xs.length ? median(xs) : x0 + 0.5, ys.length ? median(ys) : y0 + 0.5];
    };
    return best[1].map(p => refine(p[0] - 0.5, p[1] - 0.5, p[2]));
  }
  function lanczosTaps(nOut, nIn, a, b) {                    // output i reads the source around a*(i+0.5)+b
    const s = Math.max(1, a), lobes = 3, taps = [];
    const sinc = t => (t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t));
    for (let i = 0; i < nOut; i++) {
      const xs = a * (i + 0.5) + b, j0 = Math.floor(xs - lobes * s), j1 = Math.ceil(xs + lobes * s);
      const idx = [], wgt = []; let sum = 0;
      for (let j = j0; j <= j1; j++) {
        const t = (j + 0.5 - xs) / s, w = Math.abs(t) < lobes ? sinc(t) * sinc(t / lobes) : 0;
        if (w !== 0) { idx.push(clamp(j, 0, nIn - 1)); wgt.push(w); sum += w; }
      }
      taps.push([Int32Array.from(idx), Float64Array.from(wgt, w => w / sum)]);
    }
    return taps;
  }
  function register(L, W, H) {
    const f = findFinders(L, W, H);
    if (!f) return null;
    const d = FC[1] - FC[0];
    const ax = ((f[1][0] - f[0][0]) + (f[3][0] - f[2][0])) / (2 * d), ay = ((f[2][1] - f[0][1]) + (f[3][1] - f[1][1])) / (2 * d);
    const bx = (f[0][0] - ax * FC[0] + f[1][0] - ax * FC[1] + f[2][0] - ax * FC[0] + f[3][0] - ax * FC[1]) / 4;
    const by = (f[0][1] - ay * FC[0] + f[1][1] - ay * FC[0] + f[2][1] - ay * FC[1] + f[3][1] - ay * FC[1]) / 4;
    const tx = lanczosTaps(CANVAS, W, ax, bx), ty = lanczosTaps(CANVAS, H, ay, by);
    let r0 = H, r1 = 0;
    for (const [idx] of ty) for (const j of idx) { if (j < r0) r0 = j; if (j > r1) r1 = j; }
    const mid = new Float64Array((r1 - r0 + 1) * CANVAS);      // horizontal pass on the rows that are needed
    for (let y = r0; y <= r1; y++) for (let i = 0; i < CANVAS; i++) {
      const [idx, wgt] = tx[i]; let s = 0;
      for (let k = 0; k < idx.length; k++) s += wgt[k] * L[y * W + idx[k]];
      mid[(y - r0) * CANVAS + i] = s;
    }
    const out = new Float64Array(CANVAS * CANVAS);
    for (let i = 0; i < CANVAS; i++) {
      const [idx, wgt] = ty[i];
      for (let x = 0; x < CANVAS; x++) { let s = 0; for (let k = 0; k < idx.length; k++) s += wgt[k] * mid[(idx[k] - r0) * CANVAS + x]; out[i * CANVAS + x] = s; }
    }
    return { L: out, fit: { ax, bx, ay, by } };
  }

  // ---------------------------------------------------------------- decoder
  function decodeCanonical(Lc) {
    const { hb, gb } = readRing(Lc), h = parseHeader(hb);
    if (!h) return null;
    const n = NBLK * NB, gains = new Float64Array(n);
    for (let i = 0; i < n; i++) gains[i] = Math.pow(2, -GSTEP * value(gb.slice(i * GBITS, (i + 1) * GBITS)));
    const c16 = crc(gb.slice(0, n * GBITS), 16, 0x1021);
    let gainsOk = true; for (let i = 0; i < 16; i++) if (c16[i] !== gb[n * GBITS + i]) gainsOk = false;
    const shape = buildShape(h.w, RIPPLE, true), cells = shape.cells;
    const P = new Float64Array(G * G), E = new Float64Array(NT);
    for (let my = 0; my < G; my++) for (let mx = 0; mx < G; mx++) {
      let s = 0;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) { const o = (ORIGIN + 2 * my + dy) * CANVAS + ORIGIN + 2 * mx + dx; s += Lc[o] - shape.under[o]; }
      const v = s / 4 - 128; P[my * G + mx] = v; E[Math.floor(my / 8) * SIDE + Math.floor(mx / 8)] += v / 64;
    }
    const fld = envelopeField(E);
    const e = new Float64Array(NT);
    for (let t = 0; t < NT; t++) e[t] = E[cells[t]];
    const med = median(e), W = new Float64Array(F * M), V = new Float64Array(64), tmp = new Float64Array(64);
    for (let t = 0; t < NT; t++) {
      const cell = cells[t], r0 = Math.floor(cell / SIDE) * 8, c0 = (cell % SIDE) * 8;
      for (let i = 0; i < 8; i++) for (let v = 0; v < 8; v++) { let a = 0; for (let j = 0; j < 8; j++) { const o = (r0 + i) * G + c0 + j; a += (P[o] - fld[o]) * D8[v * 8 + j]; } tmp[i * 8 + v] = a; }
      for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) { let a = 0; for (let i = 0; i < 8; i++) a += D8[u * 8 + i] * tmp[i * 8 + v]; V[u * 8 + v] = a; }
      const sc = Math.pow(2, (e[t] - med) / (h.sign * A_ENV) * DR / 2);
      for (let s = 0; s < AC; s++) {
        const fl = t * AC + s;
        W[fl] = clamp(V[ZR[s] * 8 + ZC[s]] / A_RES, -1, 1) * sc * gainAt(gains, Math.floor(fl / M), fl % M);
      }
    }
    const y = synthesize(W);
    let pk = 0; for (const v of y) if (Math.abs(v) > pk) pk = Math.abs(v);
    const audio = new Float32Array(NS);
    if (pk > 0) for (let i = 0; i < NS; i++) audio[i] = y[i] / (pk * 1.02);
    return { audio, header: h, gainsOk };
  }
  function lumaOf(rgba, n) {
    const L = new Float64Array(n);
    for (let i = 0; i < n; i++) L[i] = rgba[4 * i] * WL[0] + rgba[4 * i + 1] * WL[1] + rgba[4 * i + 2] * WL[2];
    return L;
  }
  function decode(rgba, width, height, opts) {
    const progress = (opts && opts.progress) || (() => {});
    progress("looking for the image");
    const reg = register(lumaOf(rgba, width * height), width, height);
    if (!reg) return { error: "not_found" };
    progress("reading");
    const d = decodeCanonical(reg.L);
    if (!d) return { error: "header" };
    const w = d.header.w;
    const drops = [...w.keys()].sort((a, b) => (w[b] - w[a]) || (a - b)).slice(0, 6).filter(k => w[k] >= 3).map(k => NAMES[k]);
    return { audio: d.audio, rate: d.header.rate, info: { rate: d.header.rate, seconds: NS / d.header.rate, weights: w, drops, loudIsDark: d.header.sign < 0,
      gainsOk: d.gainsOk, scale: (reg.fit.ax + reg.fit.ay) / 2, fit: reg.fit } };
  }

  // 16-bit mono WAV, for playback and saving
  function wavBytes(samples, rate) {
    const n = samples.length, buf = new ArrayBuffer(44 + 2 * n), v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + 2 * n, true); str(8, "WAVEfmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, 2 * rate, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, 2 * n, true);
    for (let i = 0; i < n; i++) v.setInt16(44 + 2 * i, Math.round(clamp(samples[i], -1, 1) * 32767), true);
    return buf;
  }

  return { encode, decode, register, decodeCanonical, lumaOf, wavBytes, features, palette, weights, buildShape,
    headerBits, parseHeader, gainBits, analyze, synthesize, NS, RATE, CANVAS, NT, VERSION };
});
