import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseObjects, serializeObjects, summariseObjects } from './objects';

// Written by the Space's own infer/objects.py, so this pins the real contract.
const sample = JSON.parse(readFileSync('public/samples/synthetic-city/objects.json', 'utf8')) as Record<string, unknown>;
const GRID = { width: 384, height: 384, gsd: 0.5 };

describe('objects.json', () => {
  it('parses the Space output', () => {
    const o = parseObjects(sample, GRID)!;
    expect(o).not.toBeNull();
    expect(o.trees.length).toBe(19);
    expect(o.buildings.length).toBe(8);
    expect(o.water.length).toBe(0);
    // tallest tree first, largest building first
    expect(o.trees[0].h).toBeGreaterThanOrEqual(o.trees[o.trees.length - 1].h);
    expect(o.buildings[0].areaM2).toBeGreaterThanOrEqual(o.buildings[7].areaM2);
    for (const b of o.buildings) {
      expect(b.poly.length).toBeGreaterThanOrEqual(3);
      expect(b.hMax).toBeGreaterThanOrEqual(b.h);
    }
    for (const t of o.trees) {
      expect(t.x).toBeGreaterThanOrEqual(0);
      expect(t.x).toBeLessThanOrEqual(384);
      expect(t.r).toBeGreaterThan(0);
    }
  });

  it('drops malformed entries instead of failing', () => {
    const o = parseObjects(
      {
        version: 1,
        trees: [{ x: 10, y: 10, h: 8, r: 2 }, { x: 'a', y: 1, h: 1, r: 1 }, { x: 5, y: 5, h: -1, r: 1 }, { x: 9999, y: 5, h: 5, r: 1 }, null],
        buildings: [{ poly: [[0, 0], [10, 0], [10, 10]], h: 6 }, { poly: [[0, 0], [1, 1]], h: 6 }, { poly: [[0, 0], [1, 0], [1, 'x']], h: 6 }],
        water: 'nope',
      },
      GRID,
    )!;
    expect(o.trees).toEqual([{ x: 10, y: 10, h: 8, r: 2 }]);
    expect(o.buildings).toEqual([{ poly: [[0, 0], [10, 0], [10, 10]], h: 6, hMax: 6, areaM2: 0 }]);
    expect(o.water).toEqual([]);
    expect(o.truncated).toEqual({ trees: false, buildings: false, water: false });
  });

  it('rescales positions when the file describes a different grid; metres stay metres', () => {
    const o = parseObjects({ version: 1, grid: { width: 100, height: 50 }, trees: [{ x: 50, y: 25, h: 9, r: 3 }] }, { width: 200, height: 100 })!;
    expect(o.trees[0]).toEqual({ x: 100, y: 50, h: 9, r: 3 });
  });

  it('refuses an unknown schema version and non-objects', () => {
    expect(parseObjects({ ...sample, version: 2 }, GRID)).toBeNull();
    expect(parseObjects(null, GRID)).toBeNull();
    expect(parseObjects([1, 2], GRID)).toBeNull();
  });

  it('takes the GSD from the file when the caller has none', () => {
    expect(parseObjects({ version: 1, grid: { gsd_m: 0.3 } }, { width: 10, height: 10 })!.gsd).toBe(0.3);
  });

  it('round-trips through the project serializer', () => {
    const o = parseObjects(sample, GRID)!;
    expect(parseObjects(JSON.parse(JSON.stringify(serializeObjects(o))), GRID)).toEqual(o);
  });

  it('summarises counts and heights', () => {
    const s = summariseObjects(parseObjects(sample, GRID)!);
    expect(s.trees).toBe(19);
    expect(s.buildings).toBe(8);
    expect(s.treeMaxH).toBeGreaterThanOrEqual(s.treeMeanH!);
    expect(s.buildingMaxH).toBeGreaterThanOrEqual(s.buildingMedianH!);
    expect(s.truncated).toBe(false);
  });
});
