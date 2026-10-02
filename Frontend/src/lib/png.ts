import { unzlibSync, zlibSync } from 'fflate';

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(parts: Uint8Array[]) {
  let c = 0xffffffff;
  for (const p of parts) for (let i = 0; i < p.length; i++) c = CRC_TABLE[(c ^ p[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const t = new TextEncoder().encode(type);
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  out.set(t, 4);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32([t, data]));
  return out;
}

/** Encode a single-channel 16-bit greyscale PNG, optionally with tEXt metadata. */
export function encodeGray16Png(values: Uint16Array, width: number, height: number, text: Record<string, string> = {}): Blob {
  const stride = width * 2 + 1;
  const raw = new Uint8Array(stride * height);
  for (let r = 0; r < height; r++) {
    raw[r * stride] = 0;
    for (let c = 0; c < width; c++) {
      const v = values[r * width + c];
      raw[r * stride + 1 + c * 2] = v >> 8;
      raw[r * stride + 2 + c * 2] = v & 255;
    }
  }
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = 16; // bit depth
  ihdr[9] = 0; // greyscale
  const parts: Uint8Array[] = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr)];
  for (const [k, v] of Object.entries(text)) parts.push(chunk('tEXt', new TextEncoder().encode(`${k}\0${v}`)));
  parts.push(chunk('IDAT', zlibSync(raw, { level: 6 })), chunk('IEND', new Uint8Array(0)));
  return new Blob(parts as BlobPart[], { type: 'image/png' });
}

/** Encode a single-channel 8-bit greyscale PNG (e.g. a class-id map). */
export function encodeGray8Png(values: Uint8Array, width: number, height: number): Blob {
  const stride = width + 1;
  const raw = new Uint8Array(stride * height);
  for (let r = 0; r < height; r++) raw.set(values.subarray(r * width, (r + 1) * width), r * stride + 1);
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 0; // greyscale
  const parts: Uint8Array[] = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlibSync(raw, { level: 6 })), chunk('IEND', new Uint8Array(0))];
  return new Blob(parts as BlobPart[], { type: 'image/png' });
}

function paeth(a: number, b: number, c: number) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** Decode an 8-bit greyscale, non-interlaced PNG to its exact byte values.
 *  Label maps must not go through a canvas: colour management and premultiplication may alter values. */
export function decodeGray8Png(buffer: ArrayBuffer): { data: Uint8Array; width: number; height: number } {
  const { data, width, height } = decodePng8(buffer, [0]);
  return { data, width, height };
}

/** PNG colour type -> channels per pixel, for the 8-bit types decodePng8 reads. */
const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 6: 4 };
const COLOUR_NAMES: Record<number, string> = { 0: 'greyscale', 2: 'RGB', 6: 'RGBA' };

/** Decode an 8-bit, non-interlaced PNG of one of the `colours` types to its exact bytes, channels interleaved.
 *  Data images (label maps, Terrarium elevation tiles) must not go through a canvas: colour management and
 *  premultiplication may alter values, and anti-fingerprinting browsers (Brave, Firefox strict / private) flip the
 *  low bits of canvas readback, which in a Terrarium tile's red channel is a 256 m error. */
export function decodePng8(buffer: ArrayBuffer, colours: number[] = [0, 2, 6]): { data: Uint8Array; width: number; height: number; channels: number } {
  const bytes = new Uint8Array(buffer);
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length < 8 || sig.some((b, i) => bytes[i] !== b)) throw new Error('Not a PNG file');
  const dv = new DataView(buffer);
  let width = 0;
  let height = 0;
  let channels = 0;
  const idat: Uint8Array[] = [];
  for (let off = 8; off + 8 <= bytes.length; ) {
    const len = dv.getUint32(off);
    const type = String.fromCharCode(...bytes.subarray(off + 4, off + 8));
    const body = bytes.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(off + 8);
      height = dv.getUint32(off + 12);
      const [depth, colour, , , interlace] = body.subarray(8, 13);
      if (depth !== 8 || !colours.includes(colour) || interlace !== 0) {
        throw new Error(`Unsupported PNG (depth ${depth}, colour type ${colour}, interlace ${interlace}); expected 8-bit ${colours.map((c) => COLOUR_NAMES[c]).join(' or ')}`);
      }
      channels = CHANNELS[colour];
    } else if (type === 'IDAT') idat.push(body);
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  if (!width || !height || !idat.length) throw new Error('PNG has no image data');
  const joined = new Uint8Array(idat.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of idat) {
    joined.set(c, at);
    at += c.length;
  }
  const raw = unzlibSync(joined);
  // filters work on bytes; "left" is the same channel of the previous pixel
  const stride = width * channels;
  if (raw.length < (stride + 1) * height) throw new Error('PNG image data is truncated');
  const out = new Uint8Array(stride * height);
  for (let r = 0; r < height; r++) {
    const filter = raw[r * (stride + 1)];
    const src = raw.subarray(r * (stride + 1) + 1, (r + 1) * (stride + 1));
    const row = out.subarray(r * stride, (r + 1) * stride);
    const up = r > 0 ? out.subarray((r - 1) * stride, r * stride) : null;
    for (let c = 0; c < stride; c++) {
      const a = c >= channels ? row[c - channels] : 0;
      const b = up ? up[c] : 0;
      const d = up && c >= channels ? up[c - channels] : 0;
      let v = src[c];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) v += paeth(a, b, d);
      else if (filter !== 0) throw new Error(`Bad PNG row filter ${filter}`);
      row[c] = v & 255;
    }
  }
  return { data: out, width, height, channels };
}
