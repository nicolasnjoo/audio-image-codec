// Runs web/a2i.js under Node for scripts/web_check.py.
//   node web_node.js encode in.f32 out.rgba out.json
//   node web_node.js decode in.rgba W H out.f32 out.json
const fs = require("fs"), path = require("path");
const A2I = require(path.join(__dirname, "..", "web", "a2i.js"));
const [, , cmd, ...a] = process.argv;
const t0 = Date.now();
if (cmd === "encode") {
  const buf = fs.readFileSync(a[0]);
  const x = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const r = A2I.encode(x);
  fs.writeFileSync(a[1], Buffer.from(r.rgba.buffer));
  fs.writeFileSync(a[2], JSON.stringify(Object.assign({ ms: Date.now() - t0 }, r.info)));
} else if (cmd === "decode") {
  const buf = fs.readFileSync(a[0]);
  const r = A2I.decode(new Uint8Array(buf.buffer, buf.byteOffset, buf.length), +a[1], +a[2]);
  if (r.error) { fs.writeFileSync(a[4], JSON.stringify({ error: r.error, ms: Date.now() - t0 })); process.exit(0); }
  fs.writeFileSync(a[3], Buffer.from(r.audio.buffer));
  fs.writeFileSync(a[4], JSON.stringify(Object.assign({ ms: Date.now() - t0 }, r.info)));
} else { console.error("usage: encode|decode"); process.exit(2); }
