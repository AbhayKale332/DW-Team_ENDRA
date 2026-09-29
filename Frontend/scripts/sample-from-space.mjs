// Builds a bundled sample scene from a GeoTIFF by running it through the hosted DepthWizard model
// (the private Hugging Face Space). Nothing is predicted locally: the TIFF is only decoded to the same
// 8-bit PNG the app uploads (lib/geotiff.ts decodeTiff + lib/input.ts), the Space does the inference,
// and its result bundle is saved in the format lib/samples.ts loads.
//
// The Space drops the affine from meta.json (it may resample), so the GeoTIFF's georeference is
// written back into meta.scene — exactly what the app does in the browser for an uploaded GeoTIFF.
//
// Usage: node scripts/sample-from-space.mjs <input.tif> <output dir> [--tta]
// Needs HF_TOKEN (and optionally VITE_SPACE_ID / HF_SPACE_URL) in .env, like `npm run serve`.
import { deflateSync } from 'node:zlib';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fromArrayBuffer } from 'geotiff';
import { unzipSync } from 'fflate';
import { spaceUrlFromId } from '../server/space.mjs';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const envFile = join(root, '.env');
if (existsSync(envFile)) {
  for (const line of readFileSync(envFile, 'utf-8').split(/\r?\n/)) {
    const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/i.exec(line);
    if (m && !line.trim().startsWith('#') && process.env[m[1]] === undefined) process.env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
  }
}

const [input, outDir] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const tta = process.argv.includes('--tta');
if (!input || !outDir) {
  console.error('Usage: node scripts/sample-from-space.mjs <input.tif> <output dir> [--tta]');
  process.exit(1);
}
const TOKEN = process.env.HF_TOKEN;
if (!TOKEN) throw new Error('HF_TOKEN is not set (.env) — the private Space will reject requests.');
const SPACE = (process.env.HF_SPACE_URL || spaceUrlFromId(process.env.VITE_SPACE_ID || 'akashch1512/SingleViewHeigthEstimation')).replace(/\/$/, '');
const auth = { Authorization: `Bearer ${TOKEN}` };

// ------------------------------------------------------------------ decode (mirrors lib/geotiff.ts)
function stretchBand(src, n, is8bit) {
  const out = new Uint8ClampedArray(n);
  if (is8bit) {
    for (let i = 0; i < n; i++) out[i] = src[i];
    return out;
  }
  const stride = Math.max(1, Math.floor(n / 100_000));
  const sample = [];
  for (let i = 0; i < n; i += stride) if (Number.isFinite(src[i])) sample.push(src[i]);
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.02)] ?? 0;
  const hi = sample[Math.floor(sample.length * 0.98)] ?? 1;
  const span = hi - lo || 1;
  for (let i = 0; i < n; i++) out[i] = ((src[i] - lo) / span) * 255;
  return out;
}

function readGeoref(image) {
  const fd = image.getFileDirectory();
  const get = (k) => (typeof fd.getValue === 'function' ? fd.getValue(k) : fd[k]);
  const mt = get('ModelTransformation');
  let transform;
  if (mt && mt.length >= 16) transform = [mt[0], mt[1], mt[3], mt[4], mt[5], mt[7]];
  else {
    const [ox, oy] = image.getOrigin();
    const [rx, ry] = image.getResolution();
    transform = [rx, 0, ox, 0, ry, oy];
  }
  const k = image.getGeoKeys() ?? {};
  const projected = Number(k.ProjectedCSTypeGeoKey) || null;
  const geographic = Number(k.GeographicTypeGeoKey) || null;
  const modelType = Number(k.GTModelTypeGeoKey) || null;
  let epsg = projected && projected !== 32767 ? projected : null;
  if (!epsg && modelType === 2) epsg = geographic && geographic !== 32767 ? geographic : 4326;
  if (!epsg && geographic && geographic !== 32767 && modelType !== 1) epsg = geographic;
  return { epsg, transform };
}

// ------------------------------------------------------------------ PNG (RGB, 8-bit)
const CRC = new Int32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c;
});
const crc32 = (buf) => {
  let c = -1;
  for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
};
function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  data.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}
function encodePngRgb(rgb, w, h) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    Buffer.from(rgb.buffer, rgb.byteOffset + y * w * 3, w * 3).copy(raw, y * (w * 3 + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type RGB
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
}

// ------------------------------------------------------------------ Space client (mirrors GradioSpaceProvider)
async function http(path, init = {}) {
  const url = path.startsWith('http') ? path : `${SPACE}${path}`;
  let res;
  // Connection hiccups to *.hf.space are common. These all fail before the request reaches the server,
  // so repeating them cannot run the model twice.
  const NEVER_SENT = ['UND_ERR_CONNECT_TIMEOUT', 'ENOTFOUND', 'EAI_AGAIN', 'ECONNREFUSED', 'ETIMEDOUT'];
  for (let attempt = 1; ; attempt++) {
    try {
      res = await fetch(url, { ...init, headers: { ...auth, ...(init.headers ?? {}) } });
      break;
    } catch (err) {
      const code = err?.cause?.code ?? err?.code;
      if (attempt >= 6 || !NEVER_SENT.includes(code)) throw new Error(`${new URL(url).host}: ${code ?? err.message}`);
      console.log(`retry   ${new URL(url).host}: ${code} (attempt ${attempt}), retrying…`);
      await new Promise((r) => setTimeout(r, 3000 * attempt));
    }
  }
  if (!res.ok) throw new Error(`${init.method ?? 'GET'} ${new URL(url).pathname}: HTTP ${res.status} ${await res.text().catch(() => '')}`.slice(0, 400));
  return res;
}

/** GET a result file whole; any failure (including a connection dropped mid-body) is retried. */
async function download(url) {
  for (let attempt = 1; ; attempt++) {
    try {
      return new Uint8Array(await (await http(url)).arrayBuffer());
    } catch (err) {
      if (attempt >= 6) throw err;
      console.log(`retry   download interrupted (${err?.cause?.code ?? err.message}), attempt ${attempt}…`);
      await new Promise((r) => setTimeout(r, 3000 * attempt));
    }
  }
}

async function* readSse(res) {
  const dec = new TextDecoder();
  let buf = '';
  for await (const part of res.body) {
    buf += dec.decode(part, { stream: true });
    let idx;
    while ((idx = buf.search(/\r?\n\r?\n/)) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx).replace(/^\r?\n\r?\n/, '');
      let event = 'message';
      const data = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
      }
      yield { event, data: data.join('\n') };
    }
  }
}

const fileUrl = (f) => (f.url ? f.url : `${SPACE}/gradio_api/file=${f.path}`);
const fileName = (f) => basename(String(f.orig_name || f.path || f.url || '')).replace(/^.*file=/, '');

// ------------------------------------------------------------------ run
const stem = basename(input).replace(/\.[^.]+$/, '');
const tiff = await fromArrayBuffer(readFileSync(input).buffer.slice(0));
const image = await tiff.getImage();
const W = image.getWidth();
const H = image.getHeight();
const bands = image.getSamplesPerPixel();
const bits = image.getBitsPerSample(0);
const georef = readGeoref(image);
const gsd = Math.hypot(georef.transform[0], georef.transform[3]);
if (!georef.epsg || !(gsd > 0 && gsd < 1000)) throw new Error('Expected a projected, georeferenced GeoTIFF (metres).');
console.log(`input   ${basename(input)}: ${W}×${H}, ${bands} bands × ${bits} bit, EPSG:${georef.epsg}, ${gsd.toFixed(3)} m/px`);

const rasters = await image.readRasters({ samples: bands >= 3 ? [0, 1, 2] : [0] });
const n = W * H;
const chans = rasters.map((b) => stretchBand(b, n, bits === 8));
const rgb = new Uint8Array(n * 3);
for (let i = 0; i < n; i++) {
  rgb[i * 3] = chans[0][i];
  rgb[i * 3 + 1] = chans.length > 1 ? chans[1][i] : chans[0][i];
  rgb[i * 3 + 2] = chans.length > 2 ? chans[2][i] : chans[0][i];
}
const png = encodePngRgb(rgb, W, H);

console.log(`space   ${SPACE} — uploading ${(png.length / 1e6).toFixed(1)} MB PNG…`);
const form = new FormData();
form.append('files', new Blob([png], { type: 'image/png' }), `${stem}.png`);
const uploaded = await (await http('/gradio_api/upload', { method: 'POST', body: form })).json();
if (!Array.isArray(uploaded) || !uploaded[0]) throw new Error('The Space did not accept the upload.');
const fileData = { path: uploaded[0], orig_name: `${stem}.png`, size: png.length, mime_type: 'image/png', meta: { _type: 'gradio.FileData' } };

const call = await (
  await http('/gradio_api/call/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: [fileData, gsd, tta] }) })
).json();
if (!call.event_id) throw new Error('The Space returned no event id.');
console.log(`space   queued (event ${call.event_id}), waiting for the model…`);

const t0 = Date.now();
let data = null;
for await (const ev of readSse(await http(`/gradio_api/call/predict/${call.event_id}`, { headers: { Accept: 'text/event-stream' } }))) {
  if (ev.event === 'complete') {
    data = JSON.parse(ev.data);
    break;
  }
  if (ev.event === 'error') throw new Error(`The Space reported an error: ${ev.data}`);
}
if (!data) throw new Error('The Space closed the stream without a result.');
console.log(`space   done in ${((Date.now() - t0) / 1000).toFixed(0)} s`);

const status = String(data[4] ?? '');
const files = (Array.isArray(data[3]) ? data[3] : data[3] ? [data[3]] : []).filter((f) => f && (f.url || f.path));
const zip = files.find((f) => fileName(f).endsWith('.zip'));
if (!zip) throw new Error(`No result bundle returned. Status: ${status.slice(0, 300)}`);
console.log(`space   downloading ${fileName(zip)}…`);
const bundle = unzipSync(await download(fileUrl(zip)));

const pick = (name) => {
  const key = Object.keys(bundle).find((k) => basename(k) === name);
  return key ? bundle[key] : null;
};
const need = ['ndsm_m.npy', 'meta.json', 'rgb.png'];
for (const f of need) if (!pick(f)) throw new Error(`${f} missing from the result bundle (${Object.keys(bundle).join(', ')})`);

const meta = JSON.parse(Buffer.from(pick('meta.json')).toString('utf-8'));
const [outH, outW] = meta.size_px;
const [a, b, c, d, e, f] = georef.transform;
const sx = W / outW;
const sy = H / outH;
meta.scene = {
  ...meta.scene,
  // describe the GeoTIFF, not the PNG the Space saw (its temp path, "not georeferenced")
  path: basename(input),
  georeferenced: true,
  crs: `EPSG:${georef.epsg}`,
  // lib/georef.ts rescaleTransform: the model grid is the input resampled uniformly (read_scene max_side)
  transform: [a * sx, b * sy, c, d * sx, e * sy, f],
  crs_epsg: georef.epsg,
  gsd_source: 'geotiff',
  transform_note: `restored from ${basename(input)} (EPSG:${georef.epsg}), rescaled ${W}×${H} → ${outW}×${outH}`,
};
meta.note = `Predicted by the DepthWizard model (Hugging Face Space) from ${basename(input)} · ${W}×${H} px at ${gsd.toFixed(2)} m/px, EPSG:${georef.epsg}.`;

mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'meta.json'), JSON.stringify(meta, null, 2));
for (const name of ['ndsm_m.npy', 'rgb.png', 'seg.png', 'objects.json']) {
  const buf = pick(name);
  if (buf) writeFileSync(join(outDir, name), buf);
  console.log(`wrote   ${join(outDir, name)}${buf ? ` (${(buf.length / 1024).toFixed(0)} KB)` : ' — not in bundle, skipped'}`);
}
console.log(`heights ${meta.height_min_m?.toFixed?.(2)}–${meta.height_max_m?.toFixed?.(2)} m · grid ${outW}×${outH} · objects ${JSON.stringify(meta.objects?.counts ?? meta.objects)}`);
