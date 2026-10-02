// Write the two quick-look images a sample shows while its .dwproj downloads: the input image (input.png / .jpg,
// copied from the project) and the height map (height.png, coloured like the 3D view's default: Terrain over 2–98 %).
// A PNG preview over 10 MB is box-shrunk to at most 1024 px on its longer side.
//
//   node scripts/sample-previews.mjs [public/samples/<folder> ...]     (no folder: every sample)
//
// server/samples.mjs lists them with the sample; without them the sample opens straight into 3D as before.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { strFromU8, unzipSync, unzlibSync, zlibSync } from 'fflate';

// src/theme/colormaps.ts 'terrain' (the default colormap in src/store/view.ts)
const TERRAIN = [
  [22, 110, 58], [46, 160, 67], [120, 198, 61], [214, 222, 64], [250, 190, 50],
  [244, 122, 36], [220, 50, 32], [160, 20, 30],
];

function colorAt(t) {
  const s = Math.min(1, Math.max(0, t)) * (TERRAIN.length - 1);
  const k = Math.min(TERRAIN.length - 2, Math.floor(s));
  const f = s - k;
  return TERRAIN[k].map((a, i) => Math.round(a + (TERRAIN[k + 1][i] - a) * f));
}

function parseNpyF32(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const major = u8[6];
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const start = 8 + (major === 1 ? 2 : 4);
  const header = strFromU8(u8.subarray(start, start + hlen));
  if (!/'descr':\s*'<f4'/.test(header) || /'fortran_order':\s*True/.test(header)) throw new Error(`unsupported .npy header ${header}`);
  const [h, w] = /'shape':\s*\((\d+),\s*(\d+)/.exec(header).slice(1).map(Number);
  const off = start + hlen;
  return { data: new Float32Array(u8.buffer.slice(u8.byteOffset + off, u8.byteOffset + off + w * h * 4)), width: w, height: h };
}

const CRC = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  let c = 0xffffffff;
  for (let i = 4; i < 8 + data.length; i++) c = CRC[(c ^ out[i]) & 255] ^ (c >>> 8);
  dv.setUint32(8 + data.length, (c ^ 0xffffffff) >>> 0);
  return out;
}

/** RGBA 8-bit PNG. */
function encodeRgbaPng(rgba, width, height) {
  const stride = width * 4 + 1;
  const raw = new Uint8Array(stride * height);
  for (let r = 0; r < height; r++) raw.set(rgba.subarray(r * width * 4, (r + 1) * width * 4), r * stride + 1);
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr.set([8, 6, 0, 0, 0], 8);
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlibSync(raw, { level: 9 })), chunk('IEND', new Uint8Array(0))];
  return Buffer.concat(parts);
}

/** 8-bit, non-interlaced PNG → RGBA. */
function decodePng(png) {
  const dv = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const width = dv.getUint32(16);
  const height = dv.getUint32(20);
  const [depth, type, , , interlace] = png.subarray(24, 29);
  const bpp = { 0: 1, 2: 3, 4: 2, 6: 4 }[type];
  if (depth !== 8 || !bpp || interlace) throw new Error(`unsupported PNG (depth ${depth}, type ${type}, interlace ${interlace})`);
  const idat = [];
  for (let p = 8; p < png.length; ) {
    const len = dv.getUint32(p);
    if (strFromU8(png.subarray(p + 4, p + 8)) === 'IDAT') idat.push(png.subarray(p + 8, p + 8 + len));
    p += 12 + len;
  }
  const raw = unzlibSync(Buffer.concat(idat));
  const stride = width * bpp;
  const px = new Uint8Array(stride * height);
  const zero = new Uint8Array(stride);
  for (let r = 0; r < height; r++) {
    const f = raw[r * (stride + 1)];
    const src = raw.subarray(r * (stride + 1) + 1, (r + 1) * (stride + 1));
    const cur = px.subarray(r * stride, (r + 1) * stride);
    const up = r ? px.subarray((r - 1) * stride, r * stride) : zero;
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = up[i];
      const c = i >= bpp ? up[i - bpp] : 0;
      let pred = 0;
      if (f === 1) pred = a;
      else if (f === 2) pred = b;
      else if (f === 3) pred = (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      cur[i] = (src[i] + pred) & 255;
    }
  }
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0, s = 0; i < rgba.length; i += 4, s += bpp) {
    if (bpp <= 2) rgba.set([px[s], px[s], px[s], bpp === 2 ? px[s + 1] : 255], i);
    else rgba.set([px[s], px[s + 1], px[s + 2], bpp === 4 ? px[s + 3] : 255], i);
  }
  return { rgba, width, height };
}

const MAX_PREVIEW_BYTES = 10 * 1024 * 1024;
const SHRUNK_SIDE = 1024;

/** Box-average by a whole factor so the longer side is at most SHRUNK_SIDE; transparent pixels don't bleed colour. */
function shrink({ rgba, width, height }) {
  const k = Math.ceil(Math.max(width, height) / SHRUNK_SIDE);
  const w = Math.ceil(width / k);
  const h = Math.ceil(height / k);
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let yy = y * k; yy < Math.min(height, (y + 1) * k); yy++)
        for (let xx = x * k; xx < Math.min(width, (x + 1) * k); xx++) {
          const i = (yy * width + xx) * 4;
          const al = rgba[i + 3];
          r += rgba[i] * al; g += rgba[i + 1] * al; b += rgba[i + 2] * al; a += al; n++;
        }
      if (a) out.set([Math.round(r / a), Math.round(g / a), Math.round(b / a), Math.round(a / n)], (y * w + x) * 4);
    }
  return { rgba: out, width: w, height: h };
}

/** A PNG preview over MAX_PREVIEW_BYTES is shrunk: it is only a quick look while the project downloads. */
function capPng(png, decoded = () => decodePng(png)) {
  if (png.length <= MAX_PREVIEW_BYTES) return png;
  const s = shrink(decoded());
  return encodeRgbaPng(s.rgba, s.width, s.height);
}

function heightPng({ data, width, height }) {
  const finite = data.filter(Number.isFinite).sort();
  const q = (p) => finite[Math.min(finite.length - 1, Math.floor(p * finite.length))] ?? 0;
  const lo = q(0.02);
  // same floor as src/features/viewport/terrainState.ts: a flat scene is not stretched into noise
  const hi = Math.max(q(0.98), lo + 0.5);
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) continue; // transparent: no height
    rgba.set(colorAt((data[i] - lo) / (hi - lo)), i * 4);
    rgba[i * 4 + 3] = 255;
  }
  return capPng(encodeRgbaPng(rgba, width, height), () => ({ rgba, width, height }));
}

/** Write input.<ext> and height.png next to the folder's .dwproj. */
export function writePreviews(dir) {
  const proj = readdirSync(dir).find((f) => /\.dwproj$/i.test(f));
  if (!proj) throw new Error(`${dir} has no .dwproj`);
  const files = unzipSync(new Uint8Array(readFileSync(join(dir, proj))), {
    filter: (f) => !f.name.startsWith('source/') && !f.name.startsWith('outputs/'),
  });
  const manifest = JSON.parse(strFromU8(files['manifest.json']));
  const image = `input${extname(manifest.imageName).toLowerCase()}`;
  const bytes = files[manifest.imageName];
  writeFileSync(join(dir, image), image === 'input.png' ? capPng(bytes) : bytes);
  writeFileSync(join(dir, 'height.png'), heightPng(parseNpyF32(files['ndsm_m.npy'])));
  console.log(`${dir}  (${image}, height.png)`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const root = resolve('public/samples');
  const dirs = process.argv.length > 2 ? process.argv.slice(2).map((d) => resolve(d)) : readdirSync(root, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => join(root, e.name));
  for (const d of dirs) writePreviews(d);
}
