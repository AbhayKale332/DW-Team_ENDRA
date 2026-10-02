/** Measure tool maths (Length · Area · Height) on the height grid. Points are fractional pixels (pixel centres at
 *  integers), heights come from a sampler, results are metres. Pure, so the shapes are unit tested. */

export interface Pt {
  col: number;
  row: number;
}

/** Metres per pixel: one number, or separate column (x) and row (y) spacing for a non-square pixel. */
export type Gsd = number | { x: number; y: number };

/** Height in metres at fractional pixel coordinates (e.g. sampleBilinear on the scene grid). */
export type HeightAt = (col: number, row: number) => number;

const gx = (g: Gsd) => (typeof g === 'number' ? g : g.x);
const gy = (g: Gsd) => (typeof g === 'number' ? g : g.y);
const same = (a: Pt, b: Pt) => Math.abs(a.col - b.col) < 1e-9 && Math.abs(a.row - b.row) < 1e-9;

/** Drops repeated consecutive points and, for a ring, a last point that repeats the first (a self-closed input). */
export function cleanPoints(pts: Pt[], ring = false): Pt[] {
  const out: Pt[] = [];
  for (const p of pts) if (!out.length || !same(out[out.length - 1], p)) out.push(p);
  if (ring && out.length > 1 && same(out[0], out[out.length - 1])) out.pop();
  return out;
}

/** Horizontal (map) distance between two points, metres. */
export function segmentLength(a: Pt, b: Pt, gsd: Gsd): number {
  return Math.hypot((b.col - a.col) * gx(gsd), (b.row - a.row) * gy(gsd));
}

/** Length along the terrain between two points: the 3D length of the profile, ~1 sample per pixel. */
export function surfaceSegmentLength(a: Pt, b: Pt, gsd: Gsd, heightAt: HeightAt): number {
  const px = Math.hypot(b.col - a.col, b.row - a.row);
  const n = Math.max(1, Math.ceil(px));
  const step = segmentLength(a, b, gsd) / n;
  let len = 0;
  let prev = heightAt(a.col, a.row);
  for (let k = 1; k <= n; k++) {
    const t = k / n;
    const h = heightAt(a.col + (b.col - a.col) * t, a.row + (b.row - a.row) * t);
    // a no-data sample counts as level ground rather than poisoning the total
    const dh = Number.isFinite(h) && Number.isFinite(prev) ? h - prev : 0;
    len += Math.hypot(step, dh);
    if (Number.isFinite(h)) prev = h;
  }
  return len;
}

export interface PathLength {
  /** Map length, metres. */
  horizontal: number;
  /** Length along the terrain surface, metres (equals `horizontal` without a sampler). */
  surface: number;
  /** Horizontal length of each segment, in order (the closing one last for a ring). */
  segments: number[];
}

/** Horizontal and surface length of a polyline; `closed` adds the segment back to the first point (perimeter). */
export function pathLength(pts: Pt[], gsd: Gsd, heightAt?: HeightAt, closed = false): PathLength {
  const p = cleanPoints(pts, closed);
  const pairs: Array<[Pt, Pt]> = [];
  for (let i = 0; i < p.length - 1; i++) pairs.push([p[i], p[i + 1]]);
  if (closed && p.length > 2) pairs.push([p[p.length - 1], p[0]]);
  const segments = pairs.map(([a, b]) => segmentLength(a, b, gsd));
  const horizontal = segments.reduce((s, v) => s + v, 0);
  const surface = heightAt ? pairs.reduce((s, [a, b]) => s + surfaceSegmentLength(a, b, gsd, heightAt), 0) : horizontal;
  return { horizontal, surface, segments };
}

/** Planimetric polygon area in m² (shoelace). Fewer than 3 distinct points is 0; a self-intersecting ring nets out. */
export function polygonArea(pts: Pt[], gsd: Gsd): number {
  const p = cleanPoints(pts, true);
  if (p.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const a = p[i];
    const b = p[(i + 1) % p.length];
    s += a.col * b.row - b.col * a.row;
  }
  return (Math.abs(s) / 2) * gx(gsd) * gy(gsd);
}

/** Even-odd point-in-polygon test in pixel coordinates. */
export function insidePolygon(pts: Pt[], col: number, row: number): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i];
    const b = pts[j];
    if (a.row > row !== b.row > row && col < ((b.col - a.col) * (row - a.row)) / (b.row - a.row) + a.col) inside = !inside;
  }
  return inside;
}

/** Area of the terrain surface inside the polygon, m²: the planimetric area times the mean slope factor
 *  √(1 + |∇h|²) over a grid of samples (at most `maxSamples`, so large polygons stay fast). Walls make it large. */
export function surfaceArea(pts: Pt[], gsd: Gsd, heightAt: HeightAt, maxSamples = 40_000): number {
  const p = cleanPoints(pts, true);
  const plan = polygonArea(p, gsd);
  if (plan <= 0) return 0;
  let c0 = Infinity;
  let c1 = -Infinity;
  let r0 = Infinity;
  let r1 = -Infinity;
  for (const q of p) {
    c0 = Math.min(c0, q.col);
    c1 = Math.max(c1, q.col);
    r0 = Math.min(r0, q.row);
    r1 = Math.max(r1, q.row);
  }
  const stride = Math.max(1, Math.sqrt(((c1 - c0) * (r1 - r0)) / maxSamples));
  const factor = (col: number, row: number) => {
    const dx = (heightAt(col + 0.5, row) - heightAt(col - 0.5, row)) / gx(gsd);
    const dy = (heightAt(col, row + 0.5) - heightAt(col, row - 0.5)) / gy(gsd);
    return Number.isFinite(dx) && Number.isFinite(dy) ? Math.sqrt(1 + dx * dx + dy * dy) : NaN;
  };
  let sum = 0;
  let n = 0;
  for (let row = r0 + stride / 2; row < r1; row += stride) {
    for (let col = c0 + stride / 2; col < c1; col += stride) {
      if (!insidePolygon(p, col, row)) continue;
      const f = factor(col, row);
      if (!Number.isFinite(f)) continue;
      sum += f;
      n++;
    }
  }
  if (!n) {
    // a sliver no sample landed in: the slope at its vertex mean
    const f = factor(p.reduce((s, q) => s + q.col, 0) / p.length, p.reduce((s, q) => s + q.row, 0) / p.length);
    return Number.isFinite(f) ? plan * f : plan;
  }
  return plan * (sum / n);
}

export interface HeightDiff {
  /** Height at the first and second point, metres. */
  a: number;
  b: number;
  /** b − a: positive when the second point is higher. */
  dh: number;
  horizontal: number;
  /** Straight line between the two points in 3D. */
  distance3d: number;
  slopeDeg: number;
  slopePct: number;
}

/** Height difference, distance and slope between two points with known heights. Coincident points have no slope. */
export function heightDiff(a: Pt, b: Pt, ha: number, hb: number, gsd: Gsd): HeightDiff {
  const horizontal = segmentLength(a, b, gsd);
  const dh = hb - ha;
  return {
    a: ha,
    b: hb,
    dh,
    horizontal,
    distance3d: Math.hypot(horizontal, dh),
    slopeDeg: horizontal > 0 ? (Math.atan2(Math.abs(dh), horizontal) * 180) / Math.PI : 0,
    slopePct: horizontal > 0 ? (Math.abs(dh) / horizontal) * 100 : 0,
  };
}

const fixed = (v: number, d: number) => v.toFixed(d).replace(/^-(0\.?0*)$/, '$1');

/** Metres → "8.25 m", "412.3 m", "1.24 km". */
export function formatLength(m: number): string {
  if (!Number.isFinite(m)) return '—';
  const a = Math.abs(m);
  if (a >= 1000) return `${fixed(m / 1000, a >= 1e5 ? 1 : 2)} km`;
  return `${fixed(m, a < 10 ? 2 : 1)} m`;
}

/** Square metres → m² below a hectare, ha below a km², then km². */
export function formatArea(m2: number): string {
  if (!Number.isFinite(m2)) return '—';
  const a = Math.abs(m2);
  if (a >= 1e6) return `${fixed(m2 / 1e6, a >= 1e8 ? 1 : 2)} km²`;
  if (a >= 1e4) return `${fixed(m2 / 1e4, 2)} ha`;
  return `${fixed(m2, a < 10 ? 2 : a < 1000 ? 1 : 0)} m²`;
}

/** A signed height difference: "+3.20 m" / "−3.20 m" (true minus sign). */
export function formatSigned(m: number, digits = 2): string {
  if (!Number.isFinite(m)) return '—';
  const s = fixed(m, digits);
  return s.startsWith('-') ? `−${s.slice(1)} m` : `${Number(s) === 0 ? '±' : '+'}${s} m`;
}

/** Plain-text read-out for the clipboard: a title, "label: value" rows, then the caveats. */
export function measureText(title: string, rows: Array<[string, string]>, notes: string[] = []): string {
  return [title, ...rows.map(([k, v]) => `${k}: ${v}`), ...notes.map((n) => `Note: ${n}`)].join('\n');
}
