import type { HeightGrid } from '@/domain/types';
import { sampleBilinear } from './heights';

/** Scene ↔ grid conversions for the terrain coordinate contract (see workers/terrainMesh.ts). */
export interface TerrainFrame {
  grid: HeightGrid;
  gsd: number;
  base: number;
  /** Current vertical scale applied to the mesh (exaggeration, or ~0 in flat 2D views). */
  scaleY: number;
}

export function gridToWorld(f: TerrainFrame, col: number, row: number, h?: number): [number, number, number] {
  const { grid, gsd, base, scaleY } = f;
  const height = h ?? sampleBilinear(grid, col, row);
  return [(col - (grid.width - 1) / 2) * gsd, (height - base) * scaleY, (row - (grid.height - 1) / 2) * gsd];
}

export function worldToGrid(f: TerrainFrame, x: number, z: number): [number, number] {
  const { grid, gsd } = f;
  return [x / gsd + (grid.width - 1) / 2, z / gsd + (grid.height - 1) / 2];
}

export function heightAtWorld(f: TerrainFrame, x: number, z: number): number | null {
  const [c, r] = worldToGrid(f, x, z);
  if (c < 0 || r < 0 || c > f.grid.width - 1 || r > f.grid.height - 1) return null;
  return sampleBilinear(f.grid, c, r);
}

/** Intersect a world-space ray with the height field. Returns fractional grid coordinates or null. */
export function pickHeightfield(
  f: TerrainFrame,
  origin: { x: number; y: number; z: number },
  dir: { x: number; y: number; z: number },
  maxHeight: number,
): { col: number; row: number; point: [number, number, number] } | null {
  const { grid, gsd, base } = f;
  const sy = Math.max(f.scaleY, 1e-6);
  const hx = ((grid.width - 1) / 2) * gsd;
  const hz = ((grid.height - 1) / 2) * gsd;
  const top = (maxHeight - base) * sy + 1e-3;
  // Slab test against the terrain's bounding box.
  let t0 = 0;
  let t1 = Infinity;
  const slab = (o: number, d: number, lo: number, hi: number) => {
    if (Math.abs(d) < 1e-12) return o >= lo && o <= hi;
    let a = (lo - o) / d;
    let b = (hi - o) / d;
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
    return t0 <= t1;
  };
  if (!slab(origin.x, dir.x, -hx, hx) || !slab(origin.y, dir.y, -1e-3, top) || !slab(origin.z, dir.z, -hz, hz)) return null;

  const surf = (t: number) => {
    const x = origin.x + dir.x * t;
    const z = origin.z + dir.z * t;
    const [c, r] = worldToGrid(f, x, z);
    const h = (sampleBilinear(grid, c, r) - base) * sy;
    return origin.y + dir.y * t - h;
  };
  // March in steps of ~half a pixel of horizontal travel.
  const horiz = Math.hypot(dir.x, dir.z);
  const dt = horiz > 1e-6 ? (gsd * 0.5) / horiz : ((top + 1) / Math.max(Math.abs(dir.y), 1e-6)) / 64;
  const maxSteps = 60000;
  let prevT = t0;
  if (surf(t0) <= 0) return finish(t0);
  let steps = 0;
  for (let t = t0 + dt; t <= t1 + dt && steps < maxSteps; t += dt, steps++) {
    const tt = Math.min(t, t1);
    const cur = surf(tt);
    if (cur <= 0) {
      let a = prevT;
      let b = tt;
      for (let i = 0; i < 24; i++) {
        const m = (a + b) / 2;
        if (surf(m) > 0) a = m;
        else b = m;
      }
      return finish(b);
    }
    prevT = tt;
    if (tt >= t1) break;
  }
  return null;

  function finish(t: number) {
    const x = origin.x + dir.x * t;
    const z = origin.z + dir.z * t;
    const [col, row] = worldToGrid(f, x, z);
    return { col, row, point: [x, origin.y + dir.y * t, z] as [number, number, number] };
  }
}
