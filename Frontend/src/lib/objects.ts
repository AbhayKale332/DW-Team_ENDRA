import type { BuildingObject, GridPoint, SceneObjects, TreeObject, WaterObject } from '@/domain/types';

/** Defensive ceilings — well above the backend's own caps (20k trees, 5k buildings, 500 water bodies). */
const LIMITS = { trees: 50_000, buildings: 20_000, water: 5_000, ringPoints: 10_000 };

type Json = Record<string, unknown>;
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function ring(v: unknown, sx: number, sy: number): GridPoint[] | null {
  if (!Array.isArray(v) || v.length < 3) return null;
  const out: GridPoint[] = [];
  for (const p of v.slice(0, LIMITS.ringPoints)) {
    if (!Array.isArray(p)) return null;
    const x = num(p[0]);
    const y = num(p[1]);
    if (x === null || y === null) return null;
    out.push([x * sx, y * sy]);
  }
  return out;
}

/** `objects.json` (schema version 1, written by the Space's infer/objects.py) -> SceneObjects on the given height grid.
 *  Malformed entries are dropped rather than failing the scene; an unknown schema version returns null.
 *  If the file describes a different grid size, positions are rescaled onto `grid` (heights and radii are metric already). */
export function parseObjects(json: unknown, grid: { width: number; height: number; gsd?: number | null }): SceneObjects | null {
  if (!isObj(json)) return null;
  if (json.version !== 1) {
    console.warn(`objects.json version ${String(json.version)} is not supported; 3D objects disabled`);
    return null;
  }
  const g = isObj(json.grid) ? json.grid : {};
  const gw = num(g.width) ?? grid.width;
  const gh = num(g.height) ?? grid.height;
  if (gw <= 0 || gh <= 0) return null;
  const sx = grid.width / gw;
  const sy = grid.height / gh;
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x <= grid.width && y <= grid.height;

  const trees: TreeObject[] = [];
  for (const t of Array.isArray(json.trees) ? json.trees.slice(0, LIMITS.trees) : []) {
    if (!isObj(t)) continue;
    const x = num(t.x);
    const y = num(t.y);
    const h = num(t.h);
    const r = num(t.r);
    if (x === null || y === null || h === null || r === null || h <= 0 || r <= 0) continue;
    if (!inside(x * sx, y * sy)) continue;
    trees.push({ x: x * sx, y: y * sy, h, r });
  }
  trees.sort((a, b) => b.h - a.h);

  const buildings: BuildingObject[] = [];
  for (const b of Array.isArray(json.buildings) ? json.buildings.slice(0, LIMITS.buildings) : []) {
    if (!isObj(b)) continue;
    const poly = ring(b.poly, sx, sy);
    const h = num(b.h);
    if (!poly || h === null || h <= 0) continue;
    buildings.push({ poly, h, hMax: Math.max(h, num(b.h_max) ?? h), areaM2: num(b.area_m2) ?? 0 });
  }
  buildings.sort((a, b) => b.areaM2 - a.areaM2);

  const water: WaterObject[] = [];
  for (const w of Array.isArray(json.water) ? json.water.slice(0, LIMITS.water) : []) {
    if (!isObj(w)) continue;
    const poly = ring(w.poly, sx, sy);
    if (poly) water.push({ poly, areaM2: num(w.area_m2) ?? 0 });
  }
  water.sort((a, b) => b.areaM2 - a.areaM2);

  const cut = isObj(json.truncated) ? json.truncated : {};
  return {
    version: 1,
    width: grid.width,
    height: grid.height,
    gsd: grid.gsd ?? num(g.gsd_m) ?? 0.5,
    trees,
    buildings,
    water,
    truncated: { trees: cut.trees === true, buildings: cut.buildings === true, water: cut.water === true },
  };
}

/** SceneObjects -> the backend's `objects.json` shape, so projects store the same format the Space publishes. */
export function serializeObjects(o: SceneObjects): Json {
  return {
    version: 1,
    grid: { width: o.width, height: o.height, gsd_m: o.gsd },
    trees: o.trees,
    buildings: o.buildings.map((b) => ({ poly: b.poly, h: b.h, h_max: b.hMax, area_m2: b.areaM2 })),
    water: o.water.map((w) => ({ poly: w.poly, area_m2: w.areaM2 })),
    counts: { trees: o.trees.length, buildings: o.buildings.length, water: o.water.length },
    truncated: o.truncated,
  };
}

export interface ObjectSummary {
  trees: number;
  treeMeanH: number | null;
  treeMaxH: number | null;
  buildings: number;
  buildingMedianH: number | null;
  buildingMaxH: number | null;
  water: number;
  truncated: boolean;
}

export function summariseObjects(o: SceneObjects): ObjectSummary {
  const th = o.trees.map((t) => t.h); // tallest first
  const bh = o.buildings.map((b) => b.h).sort((a, b) => a - b);
  const mid = bh.length ? (bh.length % 2 ? bh[(bh.length - 1) / 2] : (bh[bh.length / 2 - 1] + bh[bh.length / 2]) / 2) : null;
  return {
    trees: th.length,
    treeMeanH: th.length ? th.reduce((a, b) => a + b, 0) / th.length : null,
    treeMaxH: th.length ? th[0] : null,
    buildings: bh.length,
    buildingMedianH: mid,
    buildingMaxH: bh.length ? bh[bh.length - 1] : null,
    water: o.water.length,
    truncated: o.truncated.trees || o.truncated.buildings || o.truncated.water,
  };
}

/** Fetch + parse an optional objects.json; null when absent or unusable. */
export async function fetchObjects(url: string, grid: { width: number; height: number; gsd?: number | null }, init?: RequestInit): Promise<SceneObjects | null> {
  try {
    const r = await fetch(url, init);
    if (!r.ok) return null;
    return parseObjects(await r.json(), grid);
  } catch {
    return null;
  }
}
