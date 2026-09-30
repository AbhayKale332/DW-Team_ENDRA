import { describe, expect, it } from 'vitest';
import { FloodSim, simGridFrom, WET_M, type SimGrid } from './floodSim';

const W = 40;
const H = 12;
const CELL = 5;

/** Slope rising 0.1 m per cell to the east; a pit at (30, 6) floored at -3 m, its rim the slope (~3 m). */
function slope(sources: 'west' | 'none' | 'pits'): SimGrid {
  const ground = new Float32Array(W * H);
  let src: Uint8Array | null = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) ground[y * W + x] = x * 0.1;
  ground[6 * W + 30] = -3;
  if (sources === 'west') for (let y = 0; y < H; y++) src[y * W] = 1;
  if (sources === 'pits') src = null;
  return { w: W, h: H, cellM: CELL, ground, sources: src };
}

const run = (sim: FloodSim, level: number, simS: number) => {
  let t = 0;
  while (t < simS) t += sim.step(level);
};

describe('flood hydraulics', () => {
  it('conserves water and never goes negative without sources', () => {
    const sim = new FloodSim(slope('none'));
    sim.settle(-Infinity);
    for (let y = 2; y < 6; y++) for (let x = 5; x < 9; x++) sim.pour(y * W + x, 2); // a 2 m column released on the slope
    const v0 = sim.volume();
    run(sim, -Infinity, 600);
    expect(sim.volume()).toBeCloseTo(v0, 3);
    for (const d of sim.depth) expect(d).toBeGreaterThanOrEqual(0);
    // it ran downhill: the west edge now holds water, the release spot is lower than it was
    expect(sim.depth[3 * W]).toBeGreaterThan(0.05);
    expect(sim.depth[3 * W + 6]).toBeLessThan(2);
  });

  it('sends a front out from the source instead of filling instantly', () => {
    const sim = new FloodSim(slope('west'));
    sim.settle(0);
    run(sim, 2, 10);
    expect(sim.depth[6 * W + 3]).toBeGreaterThan(WET_M); // near the source: reached
    expect(sim.depth[6 * W + 15]).toBe(0); // 75 m away, 1.5 m up: not yet
    run(sim, 2, 3000);
    expect(sim.depth[6 * W + 15]).toBeGreaterThan(WET_M);
  });

  it('settles flat at the level, where the arrival model would', () => {
    const sim = new FloodSim(slope('west'));
    sim.settle(0);
    run(sim, 2, 6000);
    for (let x = 0; x < 20; x++) {
      const i = 6 * W + x;
      if (x < 19) expect(sim.z[i] + sim.depth[i]).toBeCloseTo(2, 1);
    }
    expect(sim.depth[6 * W + 25]).toBeLessThan(WET_M); // 2.5 m ground: above the level (the run-up surge leaves a film)
    expect(sim.depth[6 * W + 30]).toBe(0); // the pit is behind a 3 m rim
    const rest = new FloodSim(slope('west'));
    rest.settle(2);
    let diff = 0;
    for (let i = 0; i < W * H; i++) diff = Math.max(diff, Math.abs(rest.depth[i] - sim.depth[i]));
    expect(diff).toBeLessThan(0.05);
  });

  it('whole-scene level fills every hollow from its own low point', () => {
    const sim = new FloodSim(slope('pits'));
    sim.settle(-10);
    run(sim, 0.5, 3000);
    expect(sim.z[6 * W + 30] + sim.depth[6 * W + 30]).toBeCloseTo(0.5, 1); // the pit fills to the level
    expect(sim.depth[6 * W + 2]).toBeGreaterThan(0.2); // the low west side too
  });

  it('packs dry surroundings so nothing is drawn far from water', () => {
    const sim = new FloodSim(slope('west'));
    sim.settle(1);
    const out = new Float32Array(W * H * 4);
    const probes = Int32Array.from([6 * W + 2, 6 * W + 35]);
    const stages = new Float32Array(2);
    const st = sim.writeFrame(out, 0, true, probes, stages);
    expect(out[(6 * W + 2) * 4]).toBeCloseTo(0.8, 4); // depth
    expect(out[(6 * W + 2) * 4 + 1]).toBeCloseTo(1, 4); // surface
    expect(out[(6 * W + 35) * 4]).toBe(-1);
    expect(stages[0]).toBeCloseTo(1, 4);
    expect(stages[1]).toBeNaN();
    expect(st.wetCells).toBe(10 * H); // x = 0..9 are below 1 m; x = 10 is exactly at it (dry)
  });

  it('pools the analysis grid, keeping the sources', () => {
    const g = slope('west');
    const seedMask = g.sources!;
    const p = simGridFrom({ w: W, h: H, cellM: CELL, ground: g.ground, seedMask }, 4);
    expect(p.w).toBe(10);
    expect(p.h).toBe(3);
    expect(p.cellM).toBe(20);
    expect(p.ground[0]).toBeCloseTo(0.15, 5);
    expect(p.sources![0]).toBe(1);
    expect(p.sources![1]).toBe(0);
  });
});
