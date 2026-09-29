import type { HeightGrid, Scene } from '@/domain/types';
import { sampleBilinear, slopeAspectAt } from './heights';
import { lonLatAt } from './georef';

export interface GridPt {
  col: number;
  row: number;
}

export interface ProbeReadout {
  col: number;
  row: number;
  height: number;
  slope: number;
  aspect: number;
  lonLat: [number, number] | null;
  reference: number | null;
  error: number | null;
}

export function probeAt(scene: Scene, p: GridPt, reference?: Float32Array | null): ProbeReadout {
  const height = sampleBilinear(scene.heights, p.col, p.row);
  const { slope, aspect } = slopeAspectAt(scene.heights, p.col, p.row, scene.gsd);
  let ref: number | null = null;
  if (reference) {
    const v = sampleBilinear({ data: reference, width: scene.heights.width, height: scene.heights.height }, p.col, p.row);
    ref = Number.isFinite(v) ? v : null;
  }
  return {
    col: p.col,
    row: p.row,
    height,
    slope,
    aspect,
    lonLat: scene.georef ? lonLatAt(scene.georef, p.col, p.row) : null,
    reference: ref,
    error: ref === null ? null : height - ref,
  };
}

export interface MeasureResult {
  ground: number;
  distance3d: number;
  dh: number;
  slopeDeg: number;
  slopePct: number;
  bearing: number;
  a: number;
  b: number;
}

export function measure(scene: Scene, a: GridPt, b: GridPt): MeasureResult {
  const ha = sampleBilinear(scene.heights, a.col, a.row);
  const hb = sampleBilinear(scene.heights, b.col, b.row);
  const dx = (b.col - a.col) * scene.gsd;
  const dz = (b.row - a.row) * scene.gsd;
  const ground = Math.hypot(dx, dz);
  const dh = hb - ha;
  let bearing = (Math.atan2(dx, -dz) * 180) / Math.PI;
  if (bearing < 0) bearing += 360;
  return {
    ground,
    distance3d: Math.hypot(ground, dh),
    dh,
    slopeDeg: ground > 0 ? (Math.atan2(Math.abs(dh), ground) * 180) / Math.PI : 0,
    slopePct: ground > 0 ? (Math.abs(dh) / ground) * 100 : 0,
    bearing,
    a: ha,
    b: hb,
  };
}

export interface ProfileSample {
  d: number;
  h: number;
  ref: number | null;
  col: number;
  row: number;
}

/** Samples heights along a polyline, ~1 sample per pixel (min 256 per segment for smooth charts). */
export function profile(scene: Scene, pts: GridPt[], reference?: Float32Array | null): ProfileSample[] {
  const out: ProfileSample[] = [];
  if (pts.length < 2) return out;
  const refGrid: HeightGrid | null = reference ? { data: reference, width: scene.heights.width, height: scene.heights.height } : null;
  let dist = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    const segPx = Math.hypot(b.col - a.col, b.row - a.row);
    const n = Math.max(256, Math.ceil(segPx));
    const segLen = segPx * scene.gsd;
    for (let k = i === 0 ? 0 : 1; k <= n; k++) {
      const t = k / n;
      const col = a.col + (b.col - a.col) * t;
      const row = a.row + (b.row - a.row) * t;
      const r = refGrid ? sampleBilinear(refGrid, col, row) : null;
      out.push({ d: dist + segLen * t, h: sampleBilinear(scene.heights, col, row), ref: r !== null && Number.isFinite(r) ? r : null, col, row });
    }
    dist += segLen;
  }
  return out;
}

export function profileCsv(samples: ProfileSample[]) {
  const head = 'distance_m,col,row,height_m,reference_m';
  const rows = samples.map((s) => `${s.d.toFixed(3)},${s.col.toFixed(2)},${s.row.toFixed(2)},${s.h.toFixed(3)},${s.ref === null ? '' : s.ref.toFixed(3)}`);
  return [head, ...rows].join('\n');
}

export function compassPoint(deg: number) {
  if (!Number.isFinite(deg)) return '—';
  const names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  return names[Math.round(deg / 45) % 8];
}
