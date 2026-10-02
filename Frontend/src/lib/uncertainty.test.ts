import { describe, expect, it } from 'vitest';
import { strToU8 } from 'fflate';
import type { Scene, UncertaintyGrid } from '@/domain/types';
import { writeNpyF32 } from './npy';
import { readProjectFiles } from './dwproj';
import { validate, validateUncertainty } from './metrics';
import { probeAt } from './analysis';
import { describeAt } from './hoverInfo';
import { parseUncertainty, plusMinus, sigmaAt, sparsification, withConfidentCut } from './uncertainty';

const npy = (data: number[] | Float32Array, h: number, w: number) => {
  const u = writeNpyF32(Float32Array.from(data), [h, w]);
  return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer;
};

/** Seeded uniform 0..1 (mulberry32), so the "random σ" case is the same every run. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Heteroscedastic errors: |error| spread over two orders of magnitude, as a height map's are. */
function errors(n: number, seed = 1) {
  const r = rng(seed);
  return Float64Array.from({ length: n }, () => (r() - 0.5) * (0.2 + 6 * r()));
}

describe('σ file (ndsm_std_m.npy)', () => {
  it('parses a σ grid of the height grid size', () => {
    const g = parseUncertainty(npy([0.5, 1, 2, 4, 0, 3], 2, 3), 3, 2);
    expect(g).not.toBeNull();
    expect([g!.width, g!.height]).toEqual([3, 2]);
    expect(Array.from(g!.data)).toEqual([0.5, 1, 2, 4, 0, 3]);
  });
  it('is absent, never zero, when the backend did not send it or sent something unusable', () => {
    expect(parseUncertainty(null, 3, 2)).toBeNull();
    expect(parseUncertainty(undefined, 3, 2)).toBeNull();
    expect(parseUncertainty(npy([1, 1, 1, 1], 2, 2), 3, 2)).toBeNull(); // wrong size
    expect(parseUncertainty(npy([0, 0, 0, 0, 0, 0], 2, 3), 3, 2)).toBeNull(); // no spread at all
    expect(parseUncertainty(npy([NaN, NaN, NaN, NaN, NaN, NaN], 2, 3), 3, 2)).toBeNull();
    expect(parseUncertainty(new Uint8Array([1, 2, 3]).buffer, 3, 2)).toBeNull(); // not .npy
  });
  it('turns negative and non-finite values into no-data', () => {
    const g = parseUncertainty(npy([1, -2, Infinity, 3], 2, 2), 2, 2)!;
    expect(Number.isNaN(g.data[1])).toBe(true);
    expect(Number.isNaN(g.data[2])).toBe(true);
    expect(g.data[3]).toBe(3);
  });
  it("takes the backend's confident cut, else max(1 m, median σ), and drops σ under clouds", () => {
    const grid = { data: Float32Array.from([2, 2, 4, 4]), width: 2, height: 2 };
    expect(withConfidentCut(grid, { uncertainty: { confident_threshold_m: 1.7 } }).confidentM).toBe(1.7);
    expect(withConfidentCut(grid, {}).confidentM).toBeGreaterThan(1.9);
    expect(withConfidentCut({ ...grid, data: Float32Array.from([0.1, 0.2, 0.3, 0.2]) }, {}).confidentM).toBe(1);
    const masked = withConfidentCut(grid, {}, Uint8Array.from([0, 255, 0, 0]));
    expect(Number.isNaN(masked.data[1])).toBe(true);
    expect(grid.data[1]).toBe(2); // the input is not modified
  });
});

describe('projects carry σ', () => {
  const manifest = { format: 'dwproj', version: 2, name: 't', imageName: 'input.png', gsd: 0.5, gsdSource: 'user', georef: null, meta: {}, statusLines: [], provenance: {}, view: {}, bookmarks: [], tools: {}, reference: null };
  const base = () => ({
    'manifest.json': strToU8(JSON.stringify(manifest)),
    'input.png': new Uint8Array([137, 80, 78, 71]),
    'ndsm_m.npy': new Uint8Array(npy([0, 1, 2, 3, 4, 5], 2, 3)),
  });
  it('reads ndsm_std_m.npy when present', () => {
    const p = readProjectFiles({ ...base(), 'ndsm_std_m.npy': new Uint8Array(npy([1, 1, 2, 2, 3, 3], 2, 3)) });
    expect(p.uncertainty?.data[4]).toBe(3);
  });
  it('opens without it', () => {
    const p = readProjectFiles(base());
    expect(p.uncertainty).toBeNull();
    expect(p.heights.width).toBe(3);
  });
});

describe('sparsification (Poggi et al. 2020)', () => {
  it('matches the hand-worked case and the Python evaluator (eval/sparsification.py)', () => {
    // errors 0, 0, 3, 4: RMSE 2.5, and 0 once the two big ones are gone
    const err2 = Float64Array.from([0, 0, 9, 16]);
    const perfect = sparsification(err2, Float64Array.from([0, 0, 3, 4]), 2)!;
    expect(perfect.bySigma).toEqual([2.5, 0, 0]);
    expect(perfect.ause).toBeCloseTo(0, 12);
    expect(perfect.aurg).toBeCloseTo(2.5 - 0.625, 12);
    const flat = sparsification(err2, Float64Array.from([1, 1, 1, 1]), 2)!;
    expect(flat.bySigma).toEqual([2.5, 2.5, 0]);
    expect(flat.ause).toBeCloseTo(1.25, 12);
    expect(flat.aurg).toBeCloseTo(0.625, 12);
  });
  it('gives AUSE 0 when σ ranks pixels exactly as their error', () => {
    const e = errors(5000);
    const err2 = e.map((v) => v * v);
    const s = sparsification(err2, e.map(Math.abs))!;
    expect(Math.abs(s.ause)).toBeLessThan(1e-9);
    expect(s.bySigma).toEqual(s.oracle);
    expect(s.aurg).toBeGreaterThan(0.3 * s.rmse);
    expect(s.bySigma[0]).toBeCloseTo(s.rmse, 9);
  });
  it('gives AUSE > 0 and no gain for a random σ, and negative gain for an inverted one', () => {
    const e = errors(5000);
    const err2 = e.map((v) => v * v);
    const r = rng(7);
    const random = sparsification(err2, Float64Array.from(e, () => r()))!;
    const noisy = sparsification(err2, Float64Array.from(e, (v) => Math.abs(v) + 0.3 * (r() - 0.5)))!;
    const inverted = sparsification(err2, e.map((v) => -Math.abs(v)))!;
    expect(random.ause).toBeGreaterThan(0.1);
    expect(noisy.ause).toBeGreaterThan(0);
    expect(noisy.ause).toBeLessThan(random.ause);
    expect(Math.abs(random.aurg)).toBeLessThan(0.1 * random.rmse);
    expect(inverted.aurg).toBeLessThan(0);
  });
  it('keeps exactly the pixels at or under each percentile (brute force)', () => {
    const e = errors(301, 3);
    const err2 = e.map((v) => v * v);
    const r = rng(11);
    // ties on purpose: σ on a coarse 0.5 m grid
    const sig = Float64Array.from(e, (v) => Math.round((Math.abs(v) + r()) * 2) / 2);
    const K = 10;
    const sorted = sig.slice().sort();
    const pct = (p: number) => {
      const pos = p * (sorted.length - 1);
      const i = Math.floor(pos);
      return i + 1 < sorted.length ? sorted[i] + (sorted[i + 1] - sorted[i]) * (pos - i) : sorted[i];
    };
    const brute = Array.from({ length: K }, (_, j) => {
      const cut = pct(1 - j / K);
      let s = 0;
      let n = 0;
      sig.forEach((v, i) => {
        if (v <= cut) {
          s += err2[i];
          n++;
        }
      });
      return Math.sqrt(s / n);
    });
    const got = sparsification(err2, sig, K)!;
    got.bySigma.slice(0, K).forEach((v, j) => expect(v).toBeCloseTo(brute[j], 10));
    expect(got.bySigma[K]).toBe(0);
  });
  it('needs at least two pixels', () => {
    expect(sparsification(new Float64Array(1), new Float64Array(1))).toBeNull();
  });
});

describe('validation with σ', () => {
  // ground truth 0; the prediction is 0.5 m off where σ is 0.5, 4 m off where σ is 3; confident cut 1.5 m
  const n = 100;
  const pred = Float32Array.from({ length: n }, (_, i) => (i < 60 ? 0.5 : 4));
  const ref = new Float32Array(n);
  const sigma: UncertaintyGrid = { data: Float32Array.from({ length: n }, (_, i) => (i < 60 ? 0.5 : 3)), width: 10, height: 10, confidentM: 1.5 };

  it('reports the RMSE on confident pixels with their share', () => {
    const u = validateUncertainty(pred, ref, sigma)!;
    expect(u.confidentM).toBe(1.5);
    expect(u.confident.n).toBe(60);
    expect(u.confident.rmse).toBeCloseTo(0.5, 6);
    expect(u.coverage).toBeCloseTo(0.6, 9);
    expect(u.sparsification?.n).toBe(n);
    expect(u.sparsification!.aurg).toBeGreaterThan(0);
  });
  it('counts compared pixels without σ as not confident, and applies the ground offset', () => {
    const s = { ...sigma, data: sigma.data.slice() };
    s.data.fill(NaN, 0, 10);
    const u = validateUncertainty(pred, ref, s)!;
    expect(u.confident.n).toBe(50);
    expect(u.coverage).toBeCloseTo(0.5, 9);
    expect(validateUncertainty(pred, Float32Array.from(ref, () => 10), sigma, 10)!.confident.rmse).toBeCloseTo(0.5, 6);
    expect(validateUncertainty(pred, ref, { ...sigma, data: new Float32Array(n).fill(NaN) })).toBeNull();
  });
  it('is part of validate() only when the scene has σ', () => {
    expect(validate(pred, ref, { removeOffset: false }).uncertainty).toBeNull();
    expect(validate(pred, ref, { removeOffset: false, sigma: null }).uncertainty).toBeNull();
    const v = validate(pred, ref, { removeOffset: false, sigma });
    expect(v.uncertainty?.confident.rmse).toBeCloseTo(0.5, 6);
    expect(v.all.n).toBe(n);
  });
});

describe('probe read-out with σ', () => {
  const W = 4;
  const H = 3;
  const heights = { data: new Float32Array(W * H).fill(5), width: W, height: H };
  const scene = (uncertainty: UncertaintyGrid | null) =>
    ({ heights, uncertainty, gsd: 0.5, product: 'nDSM', georef: null, classes: null, objects: null, stats: {} }) as unknown as Scene;
  const sig: UncertaintyGrid = { data: new Float32Array(W * H).fill(0.8), width: W, height: H, confidentM: 1 };

  it('reads h ± σ where σ exists', () => {
    expect(probeAt(scene(sig), { col: 1, row: 1 }).sigma).toBeCloseTo(0.8, 6);
    expect(describeAt(scene(sig), 1, 1).rows.find(([k]) => k === 'Height here')?.[1]).toBe('5.0 m ± 0.8 m');
    expect(plusMinus(0.8)).toBe(' ± 0.8 m');
  });
  it('reads the plain height without σ, or where σ is no-data', () => {
    expect(probeAt(scene(null), { col: 1, row: 1 }).sigma).toBeNull();
    expect(describeAt(scene(null), 1, 1).rows.find(([k]) => k === 'Height here')?.[1]).toBe('5.0 m');
    const holes = { ...sig, data: sig.data.slice().fill(NaN) };
    expect(sigmaAt(scene(holes), { col: 1, row: 1 })).toBeNull();
    expect(plusMinus(null)).toBe('');
  });
});
