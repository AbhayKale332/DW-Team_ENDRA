import type { HeightGrid, HeightStats } from '@/domain/types';

export const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

/** Robust statistics over finite values. Percentiles use a 4096-bin histogram (exact enough for display). */
export function computeStats(a: Float32Array): HeightStats {
  let lo = Infinity;
  let hi = -Infinity;
  let sum = 0;
  let n = 0;
  let below = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
    n++;
    if (v < 1) below++;
  }
  if (!n) return { min: 0, max: 1, mean: 0, median: 0, p2: 0, p98: 1, fracBelow1m: 0, valid: 0 };
  const B = 4096;
  const hist = new Uint32Array(B);
  const span = hi - lo || 1;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    hist[Math.min(B - 1, Math.floor(((v - lo) / span) * B))]++;
  }
  const q = (p: number) => {
    const target = p * n;
    let acc = 0;
    for (let b = 0; b < B; b++) {
      acc += hist[b];
      if (acc >= target) return lo + ((b + 0.5) / B) * span;
    }
    return hi;
  };
  return { min: lo, max: hi, mean: sum / n, median: q(0.5), p2: q(0.02), p98: q(0.98), fracBelow1m: below / n, valid: n };
}

/** Histogram over [lo, hi] with `bins` buckets. */
export function histogram(a: Float32Array, lo: number, hi: number, bins: number): Uint32Array {
  const out = new Uint32Array(bins);
  const span = hi - lo || 1;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    const b = Math.floor(((v - lo) / span) * bins);
    out[b < 0 ? 0 : b >= bins ? bins - 1 : b]++;
  }
  return out;
}

/** Bilinear height sample at fractional pixel coordinates (col, row are pixel centres). */
export function sampleBilinear(g: HeightGrid, col: number, row: number): number {
  const { data, width: W, height: H } = g;
  const x = clamp(col, 0, W - 1);
  const y = clamp(row, 0, H - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(W - 1, x0 + 1);
  const y1 = Math.min(H - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const v00 = data[y0 * W + x0];
  const v10 = data[y0 * W + x1];
  const v01 = data[y1 * W + x0];
  const v11 = data[y1 * W + x1];
  const top = v00 + (v10 - v00) * fx;
  const bot = v01 + (v11 - v01) * fx;
  return top + (bot - top) * fy;
}

/** Slope (degrees) and aspect (degrees clockwise from north, downslope direction) at a pixel,
 *  using central differences (Horn-like). Rows grow southwards. */
export function slopeAspectAt(g: HeightGrid, col: number, row: number, gsd: number) {
  const { data, width: W, height: H } = g;
  const c = clamp(Math.round(col), 0, W - 1);
  const r = clamp(Math.round(row), 0, H - 1);
  const c0 = Math.max(0, c - 1);
  const c1 = Math.min(W - 1, c + 1);
  const r0 = Math.max(0, r - 1);
  const r1 = Math.min(H - 1, r + 1);
  const dzdx = (data[r * W + c1] - data[r * W + c0]) / (gsd * (c1 - c0 || 1)); // east
  const dzdy = (data[r1 * W + c] - data[r0 * W + c]) / (gsd * (r1 - r0 || 1)); // south
  const slope = (Math.atan(Math.hypot(dzdx, dzdy)) * 180) / Math.PI;
  // Downslope = -gradient. dz/dnorth = -dzdy, so the downslope north component is +dzdy.
  const east = -dzdx;
  const north = dzdy;
  let aspect = (Math.atan2(east, north) * 180) / Math.PI;
  if (aspect < 0) aspect += 360;
  return { slope, aspect: Math.hypot(dzdx, dzdy) < 1e-6 ? NaN : aspect };
}

/** Horn hillshade into an 8-bit buffer (azimuth clockwise from north). Ported from the v5 viewer. */
export function hillshade(h: Float32Array, W: number, H: number, gsd: number, azDeg: number, elDeg: number) {
  const out = new Uint8ClampedArray(W * H);
  const az = ((360 - azDeg + 90) * Math.PI) / 180;
  const el = (elDeg * Math.PI) / 180;
  const sinEl = Math.sin(el);
  const cosEl = Math.cos(el);
  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - 1) * W;
    const y1 = Math.min(H - 1, y + 1) * W;
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - 1);
      const x1 = Math.min(W - 1, x + 1);
      const dx = (h[y * W + x1] - h[y * W + x0]) / (gsd * (x1 - x0 || 1));
      const dy = (h[y1 + x] - h[y0 + x]) / (gsd * 2 || 1);
      const slope = Math.atan(Math.hypot(dx, dy));
      const aspect = Math.atan2(-dx, dy);
      out[y * W + x] = 255 * clamp(sinEl * Math.cos(slope) + cosEl * Math.sin(slope) * Math.cos(az - aspect), 0, 1);
    }
  }
  return out;
}

/** Replace non-finite values with the minimum so geometry stays valid; returns a copy. */
export function sanitise(a: Float32Array, fill: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Number.isFinite(a[i]) ? a[i] : fill;
  return out;
}

/** Bilinear resample of a grid to a new size (same extent, pixel-centre aligned). */
export function resampleGrid(src: Float32Array, sw: number, sh: number, dw: number, dh: number): Float32Array {
  const out = new Float32Array(dw * dh);
  const g: HeightGrid = { data: src, width: sw, height: sh };
  for (let r = 0; r < dh; r++) {
    const sr = ((r + 0.5) * sh) / dh - 0.5;
    for (let c = 0; c < dw; c++) {
      const sc = ((c + 0.5) * sw) / dw - 0.5;
      out[r * dw + c] = sampleBilinear(g, sc, sr);
    }
  }
  return out;
}

export function formatMetres(v: number, digits = 2) {
  if (!Number.isFinite(v)) return '—';
  return `${v.toFixed(digits)} m`;
}

export function formatDistance(m: number) {
  if (!Number.isFinite(m)) return '—';
  if (m >= 1000) return `${(m / 1000).toFixed(2)} km`;
  return `${m.toFixed(m < 10 ? 2 : 1)} m`;
}
