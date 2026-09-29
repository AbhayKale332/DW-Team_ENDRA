/** Height encodings used by the backend (SingleViewHeigthEstimation app.py / viz/mesh.py). */

/** RG16: q16 = R*256 + G, height = lo + q16/65535 * (hi - lo). */
export function decodeRG16(rgba: Uint8ClampedArray | Uint8Array, lo: number, hi: number): Float32Array {
  const n = rgba.length / 4;
  const out = new Float32Array(n);
  const span = hi - lo;
  for (let i = 0; i < n; i++) {
    const q = rgba[i * 4] * 256 + rgba[i * 4 + 1];
    out[i] = lo + (q / 65535) * span;
  }
  return out;
}

export function encodeRG16(h: Float32Array, lo: number, hi: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(h.length * 4);
  const span = Math.max(hi - lo, 1e-3);
  for (let i = 0; i < h.length; i++) {
    const norm = Math.min(1, Math.max(0, (h[i] - lo) / span));
    const q = Math.floor(norm * 65535 + 0.5);
    out[i * 4] = q >> 8;
    out[i * 4 + 1] = q & 255;
    out[i * 4 + 2] = 0;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Quantise to uint16 for a single-channel 16-bit PNG (ndsm16.png contract). */
export function quantise16(h: Float32Array, lo: number, hi: number): Uint16Array {
  const out = new Uint16Array(h.length);
  const span = Math.max(hi - lo, 1e-3);
  for (let i = 0; i < h.length; i++) {
    const v = Number.isFinite(h[i]) ? h[i] : lo;
    out[i] = Math.floor(Math.min(1, Math.max(0, (v - lo) / span)) * 65535 + 0.5);
  }
  return out;
}
