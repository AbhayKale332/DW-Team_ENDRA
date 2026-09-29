import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import type { ClassMap, SceneObjects, TreeObject } from '@/domain/types';
import { fillFromEdges, objectSurfaceHeights, treeCrownMask } from './objectSurface';
import { layoutTrees, TREE_CONIFER, TREE_VARIANTS } from './treeLayout';
import { createTreeModels } from '@/features/viewport/scene/treeModels';

const W = 40;
const H = 30;
const GSD = 0.5;
const TREE = 6;
const NAMES = ['other', 'ground', 'low_veg', 'building', 'water', 'road', 'tree'];

/** Flat ground at `ground` m with one conical crown (radius `rPx`, top `top` m) centred on pixel (cr, cc). */
function scene(trees: TreeObject[], opts: { withClasses?: boolean; ground?: number; extraTreeBlob?: boolean } = {}) {
  const ground = opts.ground ?? 0;
  const data = new Float32Array(W * H).fill(ground);
  const cls = new Uint8Array(W * H).fill(1);
  const paint = (cr: number, cc: number, rPx: number, top: number) => {
    for (let r = 0; r < H; r++)
      for (let c = 0; c < W; c++) {
        const d = Math.hypot(r - cr, c - cc);
        if (d > rPx) continue;
        data[r * W + c] = ground + top * (1 - 0.5 * (d / rPx));
        cls[r * W + c] = TREE;
      }
  };
  paint(10, 10, 4, 8);
  if (opts.extraTreeBlob) paint(20, 32, 3, 6); // canopy the backend did not list as a tree
  const classes: ClassMap | null = opts.withClasses === false ? null : { data: cls, width: W, height: H, names: NAMES };
  const objects: SceneObjects = {
    version: 1,
    width: W,
    height: H,
    gsd: GSD,
    trees,
    buildings: [],
    water: [],
    truncated: { trees: false, buildings: false, water: false },
  };
  return { heights: { data, width: W, height: H }, classes, objects, gsd: GSD };
}

// the painted crown: centre pixel (10, 10) → corner-origin (10.5, 10.5); radius 4 px = 2 m (area-equivalent ≈ that)
const CROWN: TreeObject = { x: 10.5, y: 10.5, h: 8, r: 2 };

describe('terrain under trees', () => {
  it('masks the listed crown, not other canopy or ground', () => {
    const s = scene([CROWN], { extraTreeBlob: true });
    const m = treeCrownMask(s)!;
    expect(m[10 * W + 10]).toBe(1); // crown centre
    expect(m[10 * W + 13]).toBe(1); // crown edge
    expect(m[20 * W + 32]).toBe(0); // unlisted canopy stays in the mesh
    expect(m[25 * W + 2]).toBe(0); // ground
  });

  it('without a class map, claims a disc slightly wider than the crown', () => {
    const m = treeCrownMask(scene([CROWN], { withClasses: false }))!;
    expect(m[10 * W + 10]).toBe(1);
    expect(m[10 * W + 20]).toBe(0);
  });

  it('fills the crown with the surrounding ground', () => {
    const s = scene([CROWN], { ground: 3 });
    const out = objectSurfaceHeights(s);
    expect(out[10 * W + 10]).toBeCloseTo(3);
    expect(Math.max(...Array.from(out))).toBeCloseTo(3);
    // raw heights untouched — probe and validation read these
    expect(s.heights.data[10 * W + 10]).toBeCloseTo(11);
    // cached per scene
    expect(objectSurfaceHeights(s)).toBe(out);
  });

  it('prefers the lowest neighbour, so a canopy fringe cannot leak a plateau inward', () => {
    const src = new Float32Array([0, 5, 5, 5, 9, 5, 5, 5, 5]);
    const mask = new Uint8Array([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    expect(fillFromEdges(src, 3, 3, mask)[4]).toBe(5);
  });

  it('falls back to the scene minimum when everything is masked', () => {
    const out = fillFromEdges(new Float32Array([4, 2, 7, 3]), 2, 2, new Uint8Array([1, 1, 1, 1]));
    expect(Array.from(out)).toEqual([2, 2, 2, 2]);
  });

  it('returns raw heights when there are no trees', () => {
    const s = scene([]);
    expect(objectSurfaceHeights(s)).toBe(s.heights.data);
  });
});

describe('tree layout', () => {
  const grid = { width: W, height: H, gsd: GSD, base: 0 };

  it('places a corner-origin position on the terrain grid, standing on the ground', () => {
    // the grid centre (W/2, H/2) is the world origin
    const [t] = layoutTrees([{ x: W / 2, y: H / 2, h: 9, r: 3 }], grid, () => 2);
    expect(t.x).toBeCloseTo(0);
    expect(t.z).toBeCloseTo(0);
    expect(t.y).toBe(2);
    expect(t.sy).toBe(9);
    // pixel (0, 0)'s centre is (0.5, 0.5) corner-origin → x = (0 − (W−1)/2)·gsd, as in terrainMesh
    const [c] = layoutTrees([{ x: 0.5, y: 0.5, h: 5, r: 2 }], grid, () => 0);
    expect(c.x).toBeCloseTo((0 - (W - 1) / 2) * GSD);
    expect(c.z).toBeCloseTo((0 - (H - 1) / 2) * GSD);
  });

  it('samples the ground at the pixel-centre index of the tree', () => {
    const seen: number[][] = [];
    layoutTrees([{ x: 10.5, y: 4.5, h: 5, r: 2 }], grid, (c, r) => (seen.push([c, r]), 0));
    expect(seen).toEqual([[10, 4]]);
  });

  it('makes slender trees conifers and is deterministic', () => {
    const trees: TreeObject[] = [
      { x: 5, y: 5, h: 20, r: 2 },
      { x: 15, y: 5, h: 8, r: 4 },
    ];
    const a = layoutTrees(trees, grid, () => 0);
    expect(a[0].variant).toBe(TREE_CONIFER);
    expect(a[1].variant).not.toBe(TREE_CONIFER);
    expect(layoutTrees(trees, grid, () => 0)).toEqual(a);
    // order-independent: seeded by position, not index
    expect(layoutTrees([trees[1], trees[0]], grid, () => 0)).toEqual([a[1], a[0]]);
  });

  it('clamps degenerate sizes and non-finite ground', () => {
    const [t] = layoutTrees([{ x: 5, y: 5, h: 0.2, r: 0.1 }], { ...grid, base: 1 }, () => NaN);
    expect(t.sy).toBe(1);
    expect(Math.min(t.sx, t.sz)).toBeGreaterThanOrEqual(0.4);
    expect(t.y).toBe(0);
  });
});

describe('tree models', () => {
  it('are unit-sized: base at 0, top ≈ 1, crown radius ≈ 1, with colours and normals', () => {
    const models = createTreeModels();
    expect(models.length).toBe(TREE_VARIANTS);
    for (const g of models) {
      const b = g.boundingBox as THREE.Box3;
      expect(b.min.y).toBeCloseTo(0, 2);
      expect(b.max.y).toBeGreaterThan(0.95);
      expect(b.max.y).toBeLessThan(1.06);
      expect(Math.max(b.max.x, -b.min.x, b.max.z, -b.min.z)).toBeGreaterThan(0.85);
      expect(Math.max(b.max.x, -b.min.x, b.max.z, -b.min.z)).toBeLessThan(1.1);
      expect(g.getAttribute('color')).toBeTruthy();
      expect(g.getAttribute('normal')).toBeTruthy();
      g.dispose();
    }
  });
});
