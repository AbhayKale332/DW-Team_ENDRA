import { describe, expect, it } from 'vitest';
import type { BuildingObject, ClassMap, GridPoint, SceneObjects } from '@/domain/types';
import { dilate, footprintMask, objectSurfaceHeights, rasterPolygon, smoothGround } from './objectSurface';
import { buildingGeometry, waterGeometry, type MeshArrays, type ObjectFrame } from './objectGeometry';
import { sampleBilinear } from './heights';

const W = 40;
const H = 30;
const GSD = 0.5;
const NAMES = ['other', 'ground', 'low_veg', 'building', 'water', 'road', 'tree'];
const BUILDING = 3;

const rect = (x0: number, y0: number, x1: number, y1: number): GridPoint[] => [
  [x0, y0],
  [x1, y0],
  [x1, y1],
  [x0, y1],
];

function objects(buildings: BuildingObject[], water: SceneObjects['water'] = []): SceneObjects {
  return { version: 1, width: W, height: H, gsd: GSD, trees: [], buildings, water, truncated: { trees: false, buildings: false, water: false } };
}

describe('footprint rasterising', () => {
  it('fills exactly the pixels whose centres are inside', () => {
    const m = new Uint8Array(W * H);
    rasterPolygon(m, W, H, rect(10, 10, 20, 15));
    let n = 0;
    for (let r = 0; r < H; r++)
      for (let c = 0; c < W; c++) {
        const inside = c >= 10 && c <= 19 && r >= 10 && r <= 14;
        expect(m[r * W + c]).toBe(inside ? 1 : 0);
        n += m[r * W + c];
      }
    expect(n).toBe(50);
  });

  it('dilates by k pixels in a square', () => {
    const m = new Uint8Array(W * H);
    m[10 * W + 10] = 1;
    const d = dilate(m, W, H, 1);
    let n = 0;
    d.forEach((v) => (n += v));
    expect(n).toBe(9);
    expect(d[9 * W + 9]).toBe(1);
    expect(d[11 * W + 11]).toBe(1);
    expect(d[12 * W + 10]).toBe(0);
  });

  it('grows building footprints by ~1 m', () => {
    const m = footprintMask({ heights: { data: new Float32Array(W * H), width: W, height: H }, classes: null, objects: objects([{ poly: rect(10, 10, 20, 15), h: 9, hMax: 9, areaM2: 12.5 }]), gsd: GSD })!;
    expect(m[10 * W + 8]).toBe(1); // 2 px = 1 m outside
    expect(m[10 * W + 7]).toBe(0);
  });
});

describe('ground under objects', () => {
  it('smooths ground-like pixels only, over ground-like neighbours only', () => {
    const src = new Float32Array([0, 1, 0, 1, 50, 1, 0, 1, 0]);
    const ground = new Uint8Array([1, 1, 1, 1, 0, 1, 1, 1, 1]);
    const out = smoothGround(src, 3, 3, ground, 1);
    expect(out[4]).toBe(50); // the roof pixel is untouched…
    expect(out[0]).toBeCloseTo((0 + 1 + 1) / 3); // …and never averaged into its neighbours
  });

  it('replaces a building with the surrounding ground; raw heights stay', () => {
    const data = new Float32Array(W * H).fill(0.2);
    const cls = new Uint8Array(W * H).fill(1);
    for (let r = 10; r < 15; r++)
      for (let c = 10; c < 20; c++) {
        data[r * W + c] = 12;
        cls[r * W + c] = BUILDING;
      }
    const classes: ClassMap = { data: cls, width: W, height: H, names: NAMES };
    const s = { heights: { data, width: W, height: H }, classes, objects: objects([{ poly: rect(10, 10, 20, 15), h: 12, hMax: 12, areaM2: 12.5 }]), gsd: GSD };
    const out = objectSurfaceHeights(s);
    expect(out[12 * W + 15]).toBeCloseTo(0.2);
    expect(Math.max(...Array.from(out))).toBeCloseTo(0.2);
    expect(data[12 * W + 15]).toBe(12);
  });

  it('flattens only the classes switched on', () => {
    const data = new Float32Array(W * H).fill(0);
    for (let r = 10; r < 15; r++) for (let c = 10; c < 20; c++) data[r * W + c] = 12;
    const s = { heights: { data, width: W, height: H }, classes: null, objects: objects([{ poly: rect(10, 10, 20, 15), h: 12, hMax: 12, areaM2: 12.5 }]), gsd: GSD };
    const treesOnly = objectSurfaceHeights(s, { buildings: false, trees: true, water: false });
    expect(treesOnly).toBe(data); // nothing applies: the raw heights themselves
    const withBuildings = objectSurfaceHeights(s, { buildings: true, trees: false, water: false });
    expect(withBuildings[12 * W + 15]).toBeCloseTo(0);
  });
});

/** Face normal of triangle t (indices into `a`). */
function faceNormal(a: MeshArrays, t: number): [number, number, number] {
  const p = (k: number) => [a.positions[a.index[t + k] * 3], a.positions[a.index[t + k] * 3 + 1], a.positions[a.index[t + k] * 3 + 2]];
  const [A, B, C] = [p(0), p(1), p(2)];
  const u = [B[0] - A[0], B[1] - A[1], B[2] - A[2]];
  const v = [C[0] - A[0], C[1] - A[1], C[2] - A[2]];
  return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
}

describe('building and water geometry', () => {
  const frame: ObjectFrame = { width: W, height: H, gsd: GSD, base: 0, groundAt: () => 0.5 };

  for (const [label, poly] of [
    ['clockwise ring', rect(10, 10, 20, 15)],
    ['counter-clockwise ring', rect(10, 10, 20, 15).reverse()],
  ] as Array<[string, GridPoint[]]>) {
    it(`roofs face up and walls face out (${label})`, () => {
      const { roof, walls } = buildingGeometry([{ poly, h: 12, hMax: 13, areaM2: 12.5 }], frame);
      expect(roof.index.length).toBe(6); // a rectangle = 2 triangles
      for (let i = 1; i < roof.positions.length; i += 3) expect(roof.positions[i]).toBe(12);
      for (let t = 0; t < roof.index.length; t += 3) expect(faceNormal(roof, t)[1]).toBeGreaterThan(0);
      // footprint centre in world: ((15 − W/2)·gsd, (12.5 − H/2)·gsd)
      const cx = (15 - W / 2) * GSD;
      const cz = (12.5 - H / 2) * GSD;
      expect(walls.index.length).toBe(4 * 6);
      for (let t = 0; t < walls.index.length; t += 3) {
        const n = faceNormal(walls, t);
        const i0 = walls.index[t] * 3;
        const i1 = walls.index[t + 1] * 3;
        const mx = (walls.positions[i0] + walls.positions[i1]) / 2 - cx;
        const mz = (walls.positions[i0 + 2] + walls.positions[i1 + 2]) / 2 - cz;
        expect(n[0] * mx + n[2] * mz).toBeGreaterThan(0);
        // stored normals agree with the winding
        expect(n[0] * walls.normals[i0] + n[2] * walls.normals[i0 + 2]).toBeGreaterThan(0);
      }
      // walls run from just below the ground to the roof
      let lo = Infinity;
      for (let i = 1; i < walls.positions.length; i += 3) lo = Math.min(lo, walls.positions[i]);
      expect(lo).toBeCloseTo(0.5 - 0.3);
    });
  }

  it('maps roof uv to the image and positions to the terrain frame', () => {
    const { roof } = buildingGeometry([{ poly: rect(10, 10, 20, 15), h: 12, hMax: 12, areaM2: 12.5 }], frame);
    expect(roof.positions[0]).toBeCloseTo((10 - W / 2) * GSD);
    expect(roof.positions[2]).toBeCloseTo((10 - H / 2) * GSD);
    expect(roof.uvs[0]).toBeCloseTo(10 / W);
    expect(roof.uvs[1]).toBeCloseTo(10 / H);
  });

  it('keeps a roof above ground even when the model reads it low', () => {
    const { roof } = buildingGeometry([{ poly: rect(10, 10, 20, 15), h: 0.4, hMax: 0.4, areaM2: 12.5 }], frame);
    expect(roof.positions[1]).toBeGreaterThanOrEqual(0.5 + 1 - 1e-6);
  });

  it('lays water flat just above its shore', () => {
    const w = waterGeometry([{ poly: rect(2, 2, 8, 6), areaM2: 6 }], frame);
    expect(w.index.length).toBe(6);
    for (let i = 1; i < w.positions.length; i += 3) expect(w.positions[i]).toBeCloseTo(0.55);
    for (let t = 0; t < w.index.length; t += 3) expect(faceNormal(w, t)[1]).toBeGreaterThan(0);
  });

  /** A flat-roofed block map (metres per pixel of `n`×`n`), blurred by a separable Gaussian as the height head does. */
  function blurred(n: number, gsd: number, sigmaM: number, at: (x: number, y: number) => number) {
    const raw = new Float32Array(n * n);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) raw[r * n + c] = at((c + 0.5 - n / 2) * gsd, (r + 0.5 - n / 2) * gsd);
    const s = sigmaM / gsd;
    const R = Math.ceil(3 * s);
    const k = Array.from({ length: 2 * R + 1 }, (_, i) => Math.exp(-((i - R) ** 2) / (2 * s * s)));
    const ks = k.reduce((a, b) => a + b);
    const pass = (src: Float32Array, dc: number, dr: number) => {
      const out = new Float32Array(n * n);
      for (let r = 0; r < n; r++)
        for (let c = 0; c < n; c++) {
          let a = 0;
          for (let i = -R; i <= R; i++) a += src[Math.min(n - 1, Math.max(0, r + i * dr)) * n + Math.min(n - 1, Math.max(0, c + i * dc))] * k[i + R];
          out[r * n + c] = a / ks;
        }
      return out;
    };
    return pass(pass(raw, 1, 0), 0, 1);
  }
  const towerFrame = (h: Float32Array, n: number, gsd: number): ObjectFrame => ({
    width: n,
    height: n,
    gsd,
    base: 0,
    groundAt: () => 0,
    aglAt: (c, r) => sampleBilinear({ data: h, width: n, height: n }, c, r),
  });
  const square = (n: number, gsd: number, x0: number, y0: number, x1: number, y1: number) => rect(n / 2 + x0 / gsd, n / 2 + y0 / gsd, n / 2 + x1 / gsd, n / 2 + y1 / gsd);

  it('stands a blurred tower at its DSM peak, not at the median of its dome', () => {
    const n = 120;
    const gsd = 0.5;
    const h = blurred(n, gsd, 4, (x, y) => (Math.abs(x) < 8 && Math.abs(y) < 8 ? 90 : 0));
    const peak = Math.max(...h);
    // the backend's median of this dome is ~20 m under the peak
    const { roof } = buildingGeometry([{ poly: square(n, gsd, -8, -8, 8, 8), h: 65, hMax: 80, areaM2: 256 }], towerFrame(h, n, gsd));
    for (let i = 1; i < roof.positions.length; i += 3) expect(roof.positions[i]).toBeGreaterThan(peak - 2.5);
  });

  it('raises a tower on a podium without lifting the podium', () => {
    const n = 160;
    const gsd = 0.5;
    const h = blurred(n, gsd, 4, (x, y) => (x > 5 && x < 25 && Math.abs(y) < 10 ? 80 : Math.abs(x) < 30 && Math.abs(y) < 15 ? 10 : 0));
    const peak = Math.max(...h);
    const { roof } = buildingGeometry([{ poly: square(n, gsd, -30, -15, 30, 15), h: 12, hMax: 60, areaM2: 1800 }], towerFrame(h, n, gsd));
    let tower = -Infinity;
    let podium = -Infinity;
    for (let i = 0; i < roof.positions.length; i += 3) {
      const [x, y, z] = [roof.positions[i], roof.positions[i + 1], roof.positions[i + 2]];
      if (x > 8 && x < 22 && Math.abs(z) < 7) tower = Math.max(tower, y);
      if (x < -10 && Math.abs(z) < 10) podium = Math.max(podium, y);
    }
    expect(tower).toBeGreaterThan(peak - 2.5);
    expect(podium).toBeLessThan(14);
  });

  it('keeps each tower of a merged block at its own height', () => {
    // one class-map footprint, several towers: none may take a taller neighbour's height
    const n = 220;
    const gsd = 0.5;
    const towers = [
      { x0: -48, x1: -28, y0: -18, y1: 2, h: 90 },
      { x0: -26, x1: -10, y0: -18, y1: -2, h: 60 },
      { x0: -6, x1: 10, y0: -10, y1: 6, h: 40 },
      { x0: 30, x1: 48, y0: 0, y1: 18, h: 30 },
    ];
    const h = blurred(n, gsd, 4, (x, y) => towers.find((t) => x > t.x0 && x < t.x1 && y > t.y0 && y < t.y1)?.h ?? (Math.abs(x) < 50 && Math.abs(y) < 20 ? 12 : 0));
    const { roof } = buildingGeometry([{ poly: square(n, gsd, -50, -20, 50, 20), h: 14, hMax: 70, areaM2: 4000 }], towerFrame(h, n, gsd));
    for (const t of towers) {
      const cx = (t.x0 + t.x1) / 2;
      const cz = (t.y0 + t.y1) / 2;
      const dsm = sampleBilinear({ data: h, width: n, height: n }, n / 2 + cx / gsd - 0.5, n / 2 + cz / gsd - 0.5);
      // the roof vertex nearest the tower's centre
      let best = Infinity;
      let y = NaN;
      for (let i = 0; i < roof.positions.length; i += 3) {
        const d = Math.hypot(roof.positions[i] - cx, roof.positions[i + 2] - cz);
        if (d < best) [best, y] = [d, roof.positions[i + 1]];
      }
      expect(Math.abs(y - dsm)).toBeLessThan(3);
    }
  });
});

