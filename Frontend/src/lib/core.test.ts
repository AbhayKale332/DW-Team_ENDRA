import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseNpy, writeNpyF32 } from './npy';
import { computeStats, resampleGrid, sampleBilinear, slopeAspectAt } from './heights';
import { computeMetrics, validate } from './metrics';
import { decodeRG16, encodeRG16 } from './rg16';
import { mapToPixel, pixelToMap, rescaleTransform } from './georef';
import { pickHeightfield } from './pick';
import { measure } from './analysis';
import { buildTerrain } from '@/workers/terrainMesh';
import { parseStatus } from '@/api/gradio/parseStatus';
import { classifyStatusText, toDepthWizardError } from '@/api/errors';
import type { Scene } from '@/domain/types';

const buf = (p: string) => {
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

describe('npy', () => {
  it('parses the bundled sample and round-trips', () => {
    const a = parseNpy(buf('public/samples/synthetic-city/ndsm_m.npy'));
    expect(a.shape).toEqual([384, 384]);
    const s = computeStats(a.data);
    expect(s.min).toBeCloseTo(0, 3);
    expect(s.max).toBeCloseTo(32.5196, 3);
    const again = parseNpy(writeNpyF32(a.data, a.shape).slice().buffer);
    expect(again.shape).toEqual([384, 384]);
    expect(again.data[12345]).toBe(a.data[12345]);
  });
});

describe('RG16', () => {
  it('encodes and decodes within quantisation error', () => {
    const h = Float32Array.from([0, 1.234, 17.5, 32.5]);
    const out = decodeRG16(encodeRG16(h, 0, 32.5), 0, 32.5);
    h.forEach((v, i) => expect(Math.abs(out[i] - v)).toBeLessThan(32.5 / 65535));
  });
});

describe('status markdown', () => {
  const md = [
    '**Height** 0.00 – 32.52 m · mean 4.76 m · median 0.11 m',
    '**GSD** 0.750 m/px (effective, user) · **scene** 1600 × 1200 px',
    '**68.9 %** of pixels below 1 m (a nadir urban scene is usually 40–70 %)',
    'Product `rDSM` · 9 artefacts written',
    'Scene resampled 3200 → 1600 px (GSD adjusted to 0.750 m/px to match).',
  ].join('\n\n');
  it('extracts the typed fields', () => {
    const s = parseStatus(md);
    expect(s.heightHigh).toBeCloseTo(32.52);
    expect(s.gsd).toBeCloseTo(0.75);
    expect(s.gsdSource).toBe('user');
    expect([s.sceneW, s.sceneH]).toEqual([1600, 1200]);
    expect(s.pctBelow1m).toBeCloseTo(68.9);
    expect(s.product).toBe('rDSM');
    expect(s.resampleNote).toContain('3200 → 1600');
  });
  it('classifies failure texts', () => {
    expect(classifyStatusText('Your ZeroGPU quota for today is used up.').kind).toBe('quota');
    expect(classifyStatusText('Upload an image first.').kind).toBe('no-image');
    expect(classifyStatusText('Inference failed: boom').detail).toBe('boom');
    expect(toDepthWizardError(new Error('HTTP 401')).kind).toBe('auth');
  });
});

describe('metrics', () => {
  it('scores the synthetic sample against its exact reference', () => {
    const pred = parseNpy(buf('public/samples/synthetic-city/ndsm_m.npy')).data;
    const gt = parseNpy(buf('public/samples/synthetic-city/gt_ndsm_m.npy')).data;
    const m = computeMetrics(pred, gt);
    expect(m.n).toBe(384 * 384);
    expect(m.rmse).toBeGreaterThan(0);
    expect(m.r).toBeGreaterThan(0.8);
    const self = validate(gt, gt, { removeOffset: false });
    expect(self.all.rmse).toBeCloseTo(0, 6);
    expect(self.all.r).toBeCloseTo(1, 6);
    expect(self.strata.reduce((a, s) => a + s.n, 0)).toBe(self.all.n);
  });
  it('removes a constant ground offset', () => {
    const pred = Float32Array.from([0, 0, 0, 10]);
    const ref = Float32Array.from([100, 100, 100, 110]);
    expect(validate(pred, ref, { removeOffset: true }).all.rmse).toBeCloseTo(0, 6);
  });
});

describe('geometry contract', () => {
  const W = 5;
  const H = 4;
  const heights = new Float32Array(W * H).map((_, i) => i % W); // ramp east
  it('places vertices at pixel centres with pixel-centre UVs', () => {
    const out = buildTerrain({ heights, width: W, height: H, gsd: 2, base: 0, budget: 1e6, wallThreshold: 0, skirtDepth: 1 });
    const p = out.terrain.positions;
    const uv = out.terrain.uvs;
    // vertex (row 0, col 0)
    expect([p[0], p[1], p[2]]).toEqual([-4, 0, -3]);
    expect(uv[0]).toBeCloseTo(0.5 / W);
    expect(uv[1]).toBeCloseTo(0.5 / H);
    // vertex (row 3, col 4) is last regular-grid vertex
    const k = (3 * W + 4) * 3;
    expect([p[k], p[k + 1], p[k + 2]]).toEqual([4, 4, 3]);
    // all terrain normals point up
    for (let i = 1; i < out.terrain.normals.length; i += 3) expect(out.terrain.normals[i]).toBeGreaterThan(0);
  });
  it('adds wall cells where heights jump', () => {
    const step = new Float32Array(16).map((_, i) => (i % 4 >= 2 ? 10 : 0));
    const out = buildTerrain({ heights: step, width: 4, height: 4, gsd: 1, base: 0, budget: 1e6, wallThreshold: 2.5, skirtDepth: 1 });
    expect(out.wallCells).toBe(3);
  });
  it('picks the height field from above', () => {
    const grid = { data: heights, width: W, height: H };
    const hit = pickHeightfield({ grid, gsd: 2, base: 0, scaleY: 1 }, { x: 0, y: 100, z: 0 }, { x: 0, y: -1, z: 0 }, 4);
    expect(hit).not.toBeNull();
    expect(hit!.col).toBeCloseTo(2, 3);
    expect(hit!.row).toBeCloseTo(1.5, 3);
    expect(hit!.point[1]).toBeCloseTo(sampleBilinear(grid, 2, 1.5), 2);
  });
  it('computes slope from gradients', () => {
    const grid = { data: heights, width: W, height: H };
    const { slope, aspect } = slopeAspectAt(grid, 2, 1, 1); // rises 1 m per metre to the east
    expect(slope).toBeCloseTo(45, 3);
    expect(aspect).toBeCloseTo(270, 3); // downslope towards the west
  });
  it('measures distance and slope', () => {
    const scene = { heights: { data: heights, width: W, height: H }, gsd: 1 } as unknown as Scene;
    const m = measure(scene, { col: 0, row: 0 }, { col: 4, row: 0 });
    expect(m.ground).toBeCloseTo(4);
    expect(m.dh).toBeCloseTo(4);
    expect(m.slopeDeg).toBeCloseTo(45);
    expect(m.bearing).toBeCloseTo(90);
  });
});

describe('georeferencing', () => {
  it('rescales a transform for a downsampled grid and inverts it', () => {
    const t: [number, number, number, number, number, number] = [0.5, 0, 500000, 0, -0.5, 2000000];
    const r = rescaleTransform(t, 3200, 2400, 1600, 1200);
    expect(r[0]).toBeCloseTo(1);
    expect(r[4]).toBeCloseTo(-1);
    const [x, y] = pixelToMap(r, 10, 20);
    expect([x, y]).toEqual([500010.5, 1999979.5]);
    const [c, rr] = mapToPixel(r, x, y);
    expect(c).toBeCloseTo(10);
    expect(rr).toBeCloseTo(20);
  });
  it('resamples same-extent grids', () => {
    const src = Float32Array.from([0, 1, 2, 3]);
    const out = resampleGrid(src, 2, 2, 4, 4);
    expect(out.length).toBe(16);
    expect(out[0]).toBeCloseTo(0);
    expect(out[15]).toBeCloseTo(3);
  });
});
