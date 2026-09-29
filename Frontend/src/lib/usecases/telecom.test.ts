import { describe, expect, it } from 'vitest';
import type { AnalysisGrid } from './grid';
import { computeCoverage, DEFAULT_PARAMS, ENV_MODEL, fspl, knifeEdge, suggestSites, RE_OFFSET_DB, type Tower } from './telecom';

function grid(w: number, h: number, surf: (x: number, y: number) => number, cellM = 5): AnalysisGrid {
  const surface = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) surface[y * w + x] = surf(x, y);
  return { w, h, f: 1, cellM, surface, ground: new Float32Array(w * h), hasTerrain: false, building: null, water: null, sceneW: w, sceneH: h };
}
const tower = (col: number, row: number, extra: Partial<Tower> = {}): Tower => ({ id: 't', col, row, heightM: 25, eirpDbm: 46, ...extra });
const P = { ...DEFAULT_PARAMS, env: 'rural' as const };

describe('knife edge', () => {
  it('matches ITU-R P.526 anchors', () => {
    expect(knifeEdge(-1)).toBe(0);
    expect(knifeEdge(0)).toBeCloseTo(6.03, 1);
    expect(knifeEdge(2.4)).toBeGreaterThan(20);
  });
});

describe('open flat ground', () => {
  const g = grid(120, 120, () => 0);
  const r = computeCoverage(g, [tower(20, 60)], P);

  it('is line of sight everywhere and follows free-space loss', () => {
    expect(r.summary.blockedStructures + r.summary.blockedTerrain).toBe(0);
    const i = 60 * 120 + 70; // 50 cells = 250 m east
    const d = 250;
    const env = ENV_MODEL.rural;
    const expected = 46 - RE_OFFSET_DB - (fspl(P.freqMHz, d) + 10 * (env.n - 2) * Math.log10(d / 100));
    expect(r.rsrp[i]).toBeCloseTo(expected, 0);
  });

  it('gets weaker with distance', () => {
    expect(r.rsrp[60 * 120 + 40]).toBeGreaterThan(r.rsrp[60 * 120 + 110]);
  });
});

describe('a building in the way', () => {
  // 40 m tall slab across the whole north-south extent at x = 50..54, tower at x = 20
  const g = grid(120, 120, (x) => (x >= 50 && x < 54 ? 40 : 0));
  const r = computeCoverage(g, [tower(20, 60, { heightM: 15 })], P);

  it('shadows the area behind it and costs signal', () => {
    const behind = 60 * 120 + 62;
    const beside = 60 * 120 + 46; // same side, closer: no obstacle
    expect(r.nlos[behind]).toBe(1);
    expect(r.nlos[beside]).toBe(0);
    expect(r.rsrp[behind]).toBeLessThan(r.rsrp[beside] - 15);
    expect(r.summary.blockedStructures).toBeGreaterThan(0.3);
  });

  it('suggests a site on or behind the building, not next to the tower', () => {
    const picks = suggestSites(g, [tower(20, 60, { heightM: 15 })], P, 2, { heightM: 25, eirpDbm: 46 });
    expect(picks.length).toBeGreaterThan(0);
    expect(picks[0].col).toBeGreaterThanOrEqual(50); // on the slab (a rooftop mast sees both sides) or behind it
    expect(picks[0].gain).toBeGreaterThan(0.05);
    const after = computeCoverage(g, [tower(20, 60, { heightM: 15 }), tower(picks[0].col, picks[0].row)], P);
    expect(after.summary.share[0] + after.summary.share[1]).toBeLessThan(r.summary.share[0] + r.summary.share[1]);
  });
});
