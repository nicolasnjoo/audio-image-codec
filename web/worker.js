/* Runs the codec off the main thread. Messages:
     {id, op: "encode", mono: Float32Array (32 kHz), start}   -> {id, ok, rgba, x, info}
     {id, op: "decode", rgba, width, height}                  -> {id, ok, audio, wav, info} | {id, ok: false, error}
   and {id, progress} while working. */
importScripts("a2i.js");
self.onmessage = (e) => {
  const m = e.data, progress = (p) => self.postMessage({ id: m.id, progress: p });
  try {
    if (m.op === "encode") {
      const r = A2I.encode(m.mono, { start: m.start, progress });
      const x = Float32Array.from(r.x);
      self.postMessage({ id: m.id, ok: true, rgba: r.rgba, x, info: r.info }, [r.rgba.buffer, x.buffer]);
    } else if (m.op === "decode") {
      const r = A2I.decode(m.rgba, m.width, m.height, { progress });
      if (r.error) { self.postMessage({ id: m.id, ok: false, error: r.error }); return; }
      const wav = A2I.wavBytes(r.audio, r.rate);
      self.postMessage({ id: m.id, ok: true, audio: r.audio, wav, info: r.info }, [r.audio.buffer, wav]);
    }
  } catch (err) {
    self.postMessage({ id: m.id, ok: false, error: "crash", detail: String(err && err.message || err) });
  }
};
