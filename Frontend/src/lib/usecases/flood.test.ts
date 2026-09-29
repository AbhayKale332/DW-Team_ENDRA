import { describe, expect, it } from 'vitest';
import type { BuildingObject } from '@/domain/types';
import type { AnalysisGrid } from './grid';
import { assessBuildings, buildFloodModel, floodStats, type FloodModel } from './flood';

const W = 60;
const H = 60;

/** Slope rising 0.2 m per cell to the east from a river in columns 0..1, with an isolated pit at (40, 30). */
function grid(withWater = true): AnalysisGrid {
  const ground = new Float32Array(W * H);
  const water = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      ground[y * W + x] = x * 0.2;
      if (x < 2 && withWater) water[y * W + x] = 1;
    }
  }
  ground[30 * W + 40] = -5; // pit: lower than the slope around it, but not connected to the river below the rim
  return { w: W, h: H, f: 1, cellM: 5, surface: ground.slice(), ground, hasTerrain: true, building: null, water, sceneW: W, sceneH: H };
}

const ok = (m: ReturnType<typeof buildFloodModel>): FloodModel => {
  if ('error' in m) throw new Error(m.error);
  return m;
};

describe('flood arrival', () => {
  const g = grid();
  const m = ok(buildFloodModel(g, { kind: 'water' }));

  it('floods uphill cells only as the level reaches them', () => {
    expect(m.arrival[10 * W + 10]).toBeCloseTo(2, 4);
    expect(m.arrival[10 * W + 30]).toBeCloseTo(6, 4);
    // monotone along the slope
    for (let x = 3; x < W; x++) expect(m.arrival[10 * W + x]).toBeGreaterThanOrEqual(m.arrival[10 * W + x - 1]);
  });

  it('keeps an unconnected pit dry until the water reaches its rim', () => {
    const pit = m.arrival[30 * W + 40];
    expect(pit).toBeGreaterThan(7); // rim is ~8 m (x = 40), not the pit floor at -5
    expect(pit).toBeCloseTo(7.8, 0);
  });

  it('a whole-scene bathtub ignores connectivity', () => {
    const b = ok(buildFloodModel(g, { kind: 'bathtub' }));
    expect(b.arrival[30 * W + 40]).toBe(-5);
  });

  it('flooded share grows with the rise', () => {
    const a = floodStats(m, 1);
    const b = floodStats(m, 5);
    expect(b.share).toBeGreaterThan(a.share);
    expect(b.maxDepthM).toBeGreaterThan(a.maxDepthM);
  });

  it('reports why it cannot run', () => {
    expect(buildFloodModel(grid(false), { kind: 'water' })).toHaveProperty('error');
    expect(buildFloodModel({ ...g, hasTerrain: false }, { kind: 'water' })).toHaveProperty('error');
  });
});

describe('building prioritisation', () => {
  const g = grid();
  const m = ok(buildFloodModel(g, { kind: 'water' }));
  const square = (x: number, y: number, s: number): BuildingObject['poly'] => [
    [x, y],
    [x + s, y],
    [x + s, y + s],
    [x, y + s],
  ];
  const near: BuildingObject = { poly: square(6, 20, 6), h: 9, hMax: 10, areaM2: 900 };
  const far: BuildingObject = { poly: square(50, 20, 6), h: 9, hMax: 10, areaM2: 900 };

  it('ranks the building that floods first as more urgent', () => {
    const [a, b] = assessBuildings([near, far], g, m, null, 1);
    expect(a.arrival).toBeLessThan(b.arrival);
    expect(a.score).toBeGreaterThan(b.score);
  });

  it('boosts a building holding a critical facility', () => {
    const [plain] = assessBuildings([near], g, m, null, 1);
    const [school] = assessBuildings([near], g, m, [{ id: 's', kind: 'school', name: 'Test School', tags: {}, col: 8, row: 22 }], 1);
    expect(school.facility).toContain('Test School');
    expect(school.score).toBeGreaterThan(plain.score + 15);
  });
});
