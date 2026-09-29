import { describe, expect, it } from 'vitest';
import { anchorToDem, blockMean, cellCount } from './anchor';

function scene(W: number, H: number, terrain: (x: number, y: number) => number, towers: Array<[number, number, number, number]>) {
  const ndsm = new Float32Array(W * H);
  for (const [x0, y0, size, h] of towers) for (let y = y0; y < y0 + size; y++) for (let x = x0; x < x0 + size; x++) ndsm[y * W + x] = h;
  const surface = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) surface[y * W + x] = terrain(x, y) + ndsm[y * W + x];
  return { ndsm, surface };
}

const FULL = { structureShare: 1 };

describe('anchorToDem', () => {
  const W = 120;
  const H = 90;
  const k = 30;
  const plane = (x: number, y: number) => 100 + 0.05 * x + 0.02 * y;

  it('makes every cell of the DSM average to the DEM', () => {
    const { ndsm, surface } = scene(W, H, plane, [[40, 30, 10, 30]]);
    const r = anchorToDem(ndsm, W, H, k, blockMean(surface, W, H, k), FULL); // a surface-model DEM: cells include the tower
    expect(r.cellMeanRmseM).toBeLessThan(0.05);
  });

  it('does not count a tower twice', () => {
    const { ndsm, surface } = scene(W, H, () => 12, [[45, 30, 10, 30]]);
    const r = anchorToDem(ndsm, W, H, k, blockMean(surface, W, H, k), FULL);
    const top = r.dsm[35 * W + 50];
    expect(top).toBeGreaterThan(40);
    expect(top).toBeLessThan(44); // ~42 m; DEM + nDSM would be ~45+
    expect(Math.abs(r.dtm[5 * W + 5] - 12)).toBeLessThan(1.5);
  });

  it('reproduces a plane without edge ripple', () => {
    const { ndsm, surface } = scene(W, H, plane, []);
    const r = anchorToDem(ndsm, W, H, k, blockMean(surface, W, H, k), FULL);
    let worst = 0;
    for (let i = 0; i < r.dsm.length; i++) worst = Math.max(worst, Math.abs(r.dsm[i] - surface[i]));
    expect(worst).toBeLessThan(0.25);
  });

  it('keeps NaN where the nDSM is NaN', () => {
    const { ndsm, surface } = scene(W, H, plane, []);
    ndsm[10] = NaN;
    const r = anchorToDem(ndsm, W, H, k, blockMean(surface, W, H, k), FULL);
    expect(Number.isNaN(r.dsm[10])).toBe(true);
    expect(cellCount(W, k)).toBe(4);
  });
});

describe('structureShare 0 (default): smooth terrain, no sunken buildings', () => {
  it('keeps the terrain equal to the DEM under a building', () => {
    const W = 120;
    const H = 90;
    const k = 30;
    const ndsm = new Float32Array(W * H);
    for (let y = 30; y < 60; y++) for (let x = 30; x < 60; x++) ndsm[y * W + x] = 20; // fills one whole cell
    const flat = new Float32Array(W * H).fill(12);
    const r = anchorToDem(ndsm, W, H, k, blockMean(flat, W, H, k)); // DEM says 12 m everywhere
    expect(Math.abs(r.dtm[45 * W + 45] - 12)).toBeLessThan(0.5);
    expect(r.dsm[45 * W + 45]).toBeGreaterThan(31);
    expect(r.dsm[5 * W + 5]).toBeCloseTo(12, 0);
    // full structure share would have dug the terrain under it
    const full = anchorToDem(ndsm, W, H, k, blockMean(flat, W, H, k), { structureShare: 1 });
    expect(full.dtm[45 * W + 45]).toBeLessThan(0);
  });
});
