import type { TreeObject } from '@/domain/types';

/** Model variants (indices into `createTreeModels()`). */
export const TREE_BROADLEAF = 0;
export const TREE_CLUSTER = 1;
export const TREE_CONIFER = 2;
export const TREE_VARIANTS = 3;

/** Height / crown-diameter ratio from which a tree reads as a conifer (a narrow spire rather than a dome). */
const CONIFER_RATIO = 1.8;
const MIN_RADIUS_M = 0.5;
const MIN_HEIGHT_M = 1;

export interface TreeInstance {
  variant: number;
  /** Tree base in world coordinates (scene frame: x east, y up relative to `base`, z south), metres. */
  x: number;
  y: number;
  z: number;
  /** Crown radius (x, z) and top height (y), metres — the models are unit-sized. */
  sx: number;
  sy: number;
  sz: number;
  /** Yaw, radians. */
  rot: number;
  /** Brightness multiplier for the instance colour, ~0.8–1.15. */
  tint: number;
}

/** Deterministic 0..1 from an integer (mulberry32 step). */
function hash01(n: number) {
  let t = (n + 0x6d2b79f5) | 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

/** Trees (grid pixels, corner origin) -> placed, scaled model instances.
 *
 *  World mapping follows the terrain contract (workers/terrainMesh.ts): pixel centre (col, row) sits at
 *  x = (col − (W−1)/2)·gsd, so a corner-origin position p lands at (p − W/2)·gsd.
 *  `groundAt(col, row)` returns the absolute ground height under a pixel-centre position.
 *  Randomness is seeded by position, so a tree keeps its look when the list order changes. */
export function layoutTrees(
  trees: TreeObject[],
  grid: { width: number; height: number; gsd: number; base: number },
  groundAt: (col: number, row: number) => number,
): TreeInstance[] {
  const { width: W, height: H, gsd, base } = grid;
  return trees.map((t) => {
    const seed = (Math.round(t.x * 16) * 73856093) ^ (Math.round(t.y * 16) * 19349663);
    const r = Math.max(MIN_RADIUS_M, t.r);
    const h = Math.max(MIN_HEIGHT_M, t.h);
    const slender = h / (2 * r) >= CONIFER_RATIO;
    const ground = groundAt(t.x - 0.5, t.y - 0.5);
    const squash = 0.94 + 0.12 * hash01(seed + 3); // a little crown asymmetry
    return {
      variant: slender ? TREE_CONIFER : hash01(seed) < 0.5 ? TREE_BROADLEAF : TREE_CLUSTER,
      x: (t.x - W / 2) * gsd,
      y: (Number.isFinite(ground) ? ground : base) - base,
      z: (t.y - H / 2) * gsd,
      sx: r * squash,
      sy: h,
      sz: r / squash,
      rot: hash01(seed + 1) * Math.PI * 2,
      tint: 0.82 + 0.33 * hash01(seed + 2),
    };
  });
}
