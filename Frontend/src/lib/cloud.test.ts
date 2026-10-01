import { describe, expect, it } from 'vitest';
import { detectClouds, fillHeights, fillImage, pushPull, resampleMask } from './cloud';

/** Deterministic noise in [-1, 1]. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s / 2 ** 32) * 2 - 1;
  };
}

const W = 256;
const H = 256;
const DISC = { x: 170, y: 150, r: 50 };

/** Textured green fields with small white roofs (dark-edged, like shadowed buildings) and, optionally, a cloud:
 *  a smooth near-white disc with a soft edge. */
function scene({ cloud }: { cloud: boolean }) {
  const rand = rng(7);
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const t = rand() * 28;
      rgba[i] = 62 + t;
      rgba[i + 1] = 112 + t;
      rgba[i + 2] = 52 + t;
      rgba[i + 3] = 255;
    }
  }
  const roofs: Array<[number, number]> = [];
  for (let k = 0; k < 10; k++) roofs.push([20 + (k % 5) * 22, 30 + Math.floor(k / 5) * 60]);
  for (const [rx, ry] of roofs) {
    for (let y = ry - 1; y < ry + 8; y++) {
      for (let x = rx - 1; x < rx + 8; x++) {
        const i = (y * W + x) * 4;
        const edge = x < rx || y < ry || x >= rx + 7 || y >= ry + 7;
        const v = edge ? 30 : 240;
        rgba[i] = v;
        rgba[i + 1] = v;
        rgba[i + 2] = v;
      }
    }
  }
  if (cloud) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const d = Math.hypot(x - DISC.x, y - DISC.y);
        const a = Math.min(1, Math.max(0, (DISC.r + 4 - d) / 8)); // soft, hazy edge
        if (!a) continue;
        const i = (y * W + x) * 4;
        const c = [236, 239, 243];
        for (let ch = 0; ch < 3; ch++) rgba[i + ch] = rgba[i + ch] * (1 - a) + (c[ch] + rand() * 2) * a;
      }
    }
  }
  return { rgba, roofs };
}

describe('detectClouds', () => {
  it('masks a cloud and leaves white roofs alone', () => {
    const { rgba, roofs } = scene({ cloud: true });
    const r = detectClouds(rgba, W, H);
    const disc = (Math.PI * DISC.r ** 2) / (W * H);
    expect(r.coverage).toBeGreaterThan(disc * 0.9);
    expect(r.coverage).toBeLessThan(disc * 1.6);
    expect(r.mask[DISC.y * W + DISC.x]).toBe(255);
    for (const [rx, ry] of roofs) expect(r.mask[(ry + 3) * W + rx + 3]).toBeLessThan(128);
  });

  it('finds nothing in a clear image', () => {
    const r = detectClouds(scene({ cloud: false }).rgba, W, H);
    expect(r.coverage).toBe(0);
    expect(r.mask.every((v) => v === 0)).toBe(true);
  });
});

describe('fillImage', () => {
  it('paints the cloud with the surrounding colours', () => {
    const { rgba } = scene({ cloud: true });
    const r = detectClouds(rgba, W, H);
    const out = fillImage(rgba, r.mask, W, H);
    for (let i = 0; i < W * H; i++) {
      if (r.mask[i] < 200) continue;
      const lum = (out[i * 4] + out[i * 4 + 1] + out[i * 4 + 2]) / 3;
      expect(lum).toBeLessThan(160);
    }
    // clear pixels are untouched
    const far = (10 * W + 240) * 4;
    expect(Array.from(out.slice(far, far + 3))).toEqual(Array.from(rgba.slice(far, far + 3)));
  });
});

describe('fillHeights', () => {
  const n = 64;
  const rand = rng(3);
  const heights = new Float32Array(n * n);
  const mask = new Uint8Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = y * n + x;
      heights[i] = 0.2 + rand() * 0.2;
      if (Math.hypot(x - 32, y - 32) < 10) {
        mask[i] = 255;
        heights[i] = 30; // what the model makes of a cloud
      }
      if (x >= 30 && x <= 34 && y >= 43 && y <= 47) heights[i] = 25; // a tower right beside it
    }
  }
  heights[5] = NaN;

  it('fills the cloud from the ground around it, not from the tower', () => {
    const out = fillHeights(heights, mask, n, n);
    for (let i = 0; i < n * n; i++) {
      if (!mask[i]) continue;
      expect(Number.isFinite(out[i])).toBe(true);
      expect(out[i]).toBeLessThan(1.5);
      expect(out[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('leaves every cell outside the cloud as it was', () => {
    const out = fillHeights(heights, mask, n, n);
    for (let i = 0; i < n * n; i++) if (!mask[i]) expect(Object.is(out[i], heights[i]) || out[i] === heights[i]).toBe(true);
  });

  it('is a no-op without cloud', () => {
    expect(Array.from(fillHeights(heights, new Uint8Array(n * n), n, n))).toEqual(Array.from(heights));
  });
});

describe('pushPull', () => {
  it('keeps known cells and interpolates between them', () => {
    const v = new Float32Array([0, 0, 0, 0, 10]);
    const wt = new Float32Array([1, 0, 0, 0, 1]);
    const out = pushPull(v, wt, 5, 1, 1);
    expect(out[0]).toBe(0);
    expect(out[4]).toBe(10);
    for (let i = 1; i < 4; i++) {
      expect(out[i]).toBeGreaterThan(0);
      expect(out[i]).toBeLessThan(10);
    }
  });
});

describe('resampleMask', () => {
  it('keeps a block when scaling up', () => {
    const m = new Uint8Array([0, 0, 0, 255]);
    const up = resampleMask(m, 2, 2, 8, 8);
    expect(up[0]).toBe(0);
    expect(up[63]).toBe(255);
  });
});
