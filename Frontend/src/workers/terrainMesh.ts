/** Terrain geometry builder (pure; runs inside terrain.worker). Ported and extended from the v5 viewer.
 *
 *  Coordinate contract (shared with viz/mesh.py and every tool in the app):
 *    grid pixel (col, row)  →  x = (col − (W−1)/2)·gsd   (east)
 *                               z = (row − (H−1)/2)·gsd   (south)
 *                               y = h − base             (up, metres)
 *    uv = ((col + 0.5)/W, (row + 0.5)/H)  — pixel-centre exact, row 0 at v≈0 (glTF convention,
 *    textures uploaded with flipY = false). */

export interface TerrainBuildInput {
  heights: Float32Array;
  width: number;
  height: number;
  gsd: number;
  base: number;
  /** Max vertices of the regular grid (decimation is chosen to fit). */
  budget: number;
  /** Height jump (m) above which grid edges become vertical walls; 0 disables. */
  wallThreshold: number;
  /** Depth of the base skirt below the lowest point, metres. */
  skirtDepth: number;
}

export interface TerrainGeometryData {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  shade: Float32Array;
  index: Uint32Array;
}

export interface TerrainBuildOutput {
  terrain: TerrainGeometryData;
  skirt: TerrainGeometryData;
  step: number;
  vertices: number;
  wallCells: number;
}

class GrowF32 {
  a: Float32Array;
  n = 0;
  constructor(cap: number) {
    this.a = new Float32Array(Math.max(16, cap));
  }
  push3(x: number, y: number, z: number) {
    if (this.n + 3 > this.a.length) this.grow();
    this.a[this.n++] = x;
    this.a[this.n++] = y;
    this.a[this.n++] = z;
  }
  push2(x: number, y: number) {
    if (this.n + 2 > this.a.length) this.grow();
    this.a[this.n++] = x;
    this.a[this.n++] = y;
  }
  push1(x: number) {
    if (this.n + 1 > this.a.length) this.grow();
    this.a[this.n++] = x;
  }
  grow() {
    const b = new Float32Array(this.a.length * 2);
    b.set(this.a);
    this.a = b;
  }
  done() {
    return this.a.slice(0, this.n);
  }
}

class GrowU32 {
  a: Uint32Array;
  n = 0;
  constructor(cap: number) {
    this.a = new Uint32Array(Math.max(16, cap));
  }
  push3(a: number, b: number, c: number) {
    if (this.n + 3 > this.a.length) {
      const nb = new Uint32Array(this.a.length * 2);
      nb.set(this.a);
      this.a = nb;
    }
    this.a[this.n++] = a;
    this.a[this.n++] = b;
    this.a[this.n++] = c;
  }
  done() {
    return this.a.slice(0, this.n);
  }
}

export function chooseStep(W: number, H: number, budget: number) {
  let step = 1;
  while (Math.ceil(H / step) * Math.ceil(W / step) > budget) step++;
  return step;
}

function computeNormals(pos: Float32Array, idx: Uint32Array): Float32Array {
  const nrm = new Float32Array(pos.length);
  for (let i = 0; i < idx.length; i += 3) {
    const a = idx[i] * 3;
    const b = idx[i + 1] * 3;
    const c = idx[i + 2] * 3;
    const ux = pos[b] - pos[a];
    const uy = pos[b + 1] - pos[a + 1];
    const uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a];
    const vy = pos[c + 1] - pos[a + 1];
    const vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const k of [a, b, c]) {
      nrm[k] += nx;
      nrm[k + 1] += ny;
      nrm[k + 2] += nz;
    }
  }
  for (let i = 0; i < nrm.length; i += 3) {
    const l = Math.hypot(nrm[i], nrm[i + 1], nrm[i + 2]) || 1;
    nrm[i] /= l;
    nrm[i + 1] /= l;
    nrm[i + 2] /= l;
  }
  return nrm;
}

export function buildTerrain(inp: TerrainBuildInput): TerrainBuildOutput {
  const { heights, width: W, height: H, gsd, base } = inp;
  const step = chooseStep(W, H, inp.budget);
  const gw = Math.floor((W - 1) / step) + 1;
  const gh = Math.floor((H - 1) / step) + 1;
  const cx0 = (W - 1) / 2;
  const cz0 = (H - 1) / 2;
  const thr = inp.wallThreshold;
  const useWalls = thr > 0;

  const hAt = (r: number, c: number) => {
    const v = heights[Math.min(H - 1, r * step) * W + Math.min(W - 1, c * step)];
    return (Number.isFinite(v) ? v : base) - base;
  };

  const pos = new GrowF32(gw * gh * 3 * 1.2);
  const uv = new GrowF32(gw * gh * 2 * 1.2);
  const shade = new GrowF32(gw * gh * 1.2);
  const idx = new GrowU32((gw - 1) * (gh - 1) * 6 * 1.1);

  // vertex in grid-pixel space (col, row may be fractional for cell centres / edge midpoints)
  const add = (col: number, row: number, y: number, s: number) => {
    pos.push3((col - cx0) * gsd, y, (row - cz0) * gsd);
    uv.push2((col + 0.5) / W, (row + 0.5) / H);
    shade.push1(s);
    return pos.n / 3 - 1;
  };

  for (let r = 0; r < gh; r++) for (let c = 0; c < gw; c++) add(Math.min(W - 1, c * step), Math.min(H - 1, r * step), hAt(r, c), 1);

  const V = (r: number, c: number) => r * gw + c;
  const WALL = 0.8;
  let wallCells = 0;
  const h = [0, 0, 0, 0];
  const split = [false, false, false, false];
  const g = [0, 1, 2, 3];
  const find = (i: number): number => (g[i] === i ? i : (g[i] = find(g[i])));

  for (let r = 0; r < gh - 1; r++) {
    for (let c = 0; c < gw - 1; c++) {
      // ring: top-left, top-right, bottom-right, bottom-left
      const rc: Array<[number, number]> = [
        [r, c],
        [r, c + 1],
        [r + 1, c + 1],
        [r + 1, c],
      ];
      for (let i = 0; i < 4; i++) h[i] = hAt(rc[i][0], rc[i][1]);
      let any = false;
      for (let i = 0; i < 4; i++) {
        split[i] = useWalls && Math.abs(h[i] - h[(i + 1) % 4]) > thr;
        any = any || split[i];
      }
      if (!any) {
        const a = V(r, c);
        const b = V(r, c + 1);
        const d = V(r + 1, c + 1);
        const e = V(r + 1, c);
        idx.push3(a, e, b);
        idx.push3(b, e, d);
        continue;
      }
      wallCells++;
      for (let i = 0; i < 4; i++) g[i] = i;
      for (let i = 0; i < 4; i++) if (!split[i]) g[find(i)] = find((i + 1) % 4);
      const gsum = [0, 0, 0, 0];
      const gn = [0, 0, 0, 0];
      for (let i = 0; i < 4; i++) {
        const k = find(i);
        gsum[k] += h[i];
        gn[k]++;
      }
      // pixel-space coordinates of the cell
      const colOf = (cc: number) => Math.min(W - 1, cc * step);
      const rowOf = (rr: number) => Math.min(H - 1, rr * step);
      const ccol = (colOf(c) + colOf(c + 1)) / 2;
      const crow = (rowOf(r) + rowOf(r + 1)) / 2;
      const centre = [-1, -1, -1, -1];
      for (let k = 0; k < 4; k++) if (gn[k]) centre[k] = add(ccol, crow, gsum[k] / gn[k], 1);
      const mid: Array<[number, number]> = [];
      for (let i = 0; i < 4; i++) {
        const [y0, x0] = rc[i];
        const [y1, x1] = rc[(i + 1) % 4];
        const mcol = (colOf(x0) + colOf(x1)) / 2;
        const mrow = (rowOf(y0) + rowOf(y1)) / 2;
        if (split[i]) mid.push([add(mcol, mrow, h[i], 1), add(mcol, mrow, h[(i + 1) % 4], 1)]);
        else {
          const v = add(mcol, mrow, (h[i] + h[(i + 1) % 4]) / 2, 1);
          mid.push([v, v]);
        }
      }
      for (let i = 0; i < 4; i++) {
        const p = V(rc[i][0], rc[i][1]);
        const mPrev = mid[(i + 3) % 4][1];
        const mNext = mid[i][0];
        const ctr = centre[find(i)];
        idx.push3(p, mPrev, ctr);
        idx.push3(p, ctr, mNext);
      }
      const P = pos.a;
      for (let i = 0; i < 4; i++) {
        const j = (i + 1) % 4;
        const cp = centre[find(i)];
        const cq = centre[find(j)];
        const mp = mid[i][0];
        const mq = mid[i][1];
        const yp0 = P[mp * 3 + 1];
        const yq0 = P[mq * 3 + 1];
        const yp1 = P[cp * 3 + 1];
        const yq1 = P[cq * 3 + 1];
        if (yp0 === yq0 && yp1 === yq1) continue;
        const [y0, x0] = rc[i];
        const [y1, x1] = rc[j];
        const mcol = (colOf(x0) + colOf(x1)) / 2;
        const mrow = (rowOf(y0) + rowOf(y1)) / 2;
        const a = add(mcol, mrow, yp0, WALL);
        const b = add(mcol, mrow, yq0, WALL);
        const d = add(ccol, crow, yq1, WALL);
        const e = add(ccol, crow, yp1, WALL);
        idx.push3(a, b, d);
        idx.push3(a, d, e);
      }
    }
  }

  const positions = pos.done();
  const index = idx.done();
  const terrain: TerrainGeometryData = {
    positions,
    normals: computeNormals(positions, index),
    uvs: uv.done(),
    shade: shade.done(),
    index,
  };

  return { terrain, skirt: buildSkirt(hAt, gw, gh, step, W, H, gsd, inp.skirtDepth), step, vertices: positions.length / 3, wallCells };
}

/** Vertical curtain around the grid perimeter down to −depth, giving the terrain a solid "diorama" base. */
function buildSkirt(
  hAt: (r: number, c: number) => number,
  gw: number,
  gh: number,
  step: number,
  W: number,
  H: number,
  gsd: number,
  depth: number,
): TerrainGeometryData {
  const cx0 = (W - 1) / 2;
  const cz0 = (H - 1) / 2;
  const pos: number[] = [];
  const nrm: number[] = [];
  const uv: number[] = [];
  const sh: number[] = [];
  const idx: number[] = [];
  const side = (pts: Array<[number, number]>, nx: number, nz: number) => {
    const start = pos.length / 3;
    for (const [r, c] of pts) {
      const col = Math.min(W - 1, c * step);
      const row = Math.min(H - 1, r * step);
      const x = (col - cx0) * gsd;
      const z = (row - cz0) * gsd;
      const u = (col + 0.5) / W;
      const v = (row + 0.5) / H;
      pos.push(x, hAt(r, c), z, x, -depth, z);
      nrm.push(nx, 0, nz, nx, 0, nz);
      uv.push(u, v, u, v);
      sh.push(0.5, 0.5);
    }
    for (let i = 0; i < pts.length - 1; i++) {
      const a = start + i * 2;
      const b = a + 2;
      // wind so the face points outward along (nx, nz)
      idx.push(a, a + 1, b, b, a + 1, b + 1);
    }
  };
  const north: Array<[number, number]> = [];
  const south: Array<[number, number]> = [];
  const west: Array<[number, number]> = [];
  const east: Array<[number, number]> = [];
  for (let c = gw - 1; c >= 0; c--) north.push([0, c]);
  for (let c = 0; c < gw; c++) south.push([gh - 1, c]);
  for (let r = 0; r < gh; r++) west.push([r, 0]);
  for (let r = gh - 1; r >= 0; r--) east.push([r, gw - 1]);
  side(north, 0, -1);
  side(south, 0, 1);
  side(west, -1, 0);
  side(east, 1, 0);
  // bottom cap
  const b0 = pos.length / 3;
  const x0 = -cx0 * gsd;
  const x1 = (Math.min(W - 1, (gw - 1) * step) - cx0) * gsd;
  const z0 = -cz0 * gsd;
  const z1 = (Math.min(H - 1, (gh - 1) * step) - cz0) * gsd;
  pos.push(x0, -depth, z0, x1, -depth, z0, x1, -depth, z1, x0, -depth, z1);
  nrm.push(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
  uv.push(0, 0, 1, 0, 1, 1, 0, 1);
  sh.push(0.4, 0.4, 0.4, 0.4);
  idx.push(b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3);
  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(nrm),
    uvs: new Float32Array(uv),
    shade: new Float32Array(sh),
    index: new Uint32Array(idx),
  };
}
