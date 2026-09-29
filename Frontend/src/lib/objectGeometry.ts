import { ShapeUtils, Vector2 } from 'three';
import type { BuildingObject, GridPoint, WaterObject } from '@/domain/types';

/** Building and water geometry as plain arrays, in the terrain's world frame (workers/terrainMesh.ts):
 *  a corner-origin grid point (px, py) lands at x = (px − W/2)·gsd, z = (py − H/2)·gsd, and y is metres
 *  above the scene base. Pure (no GPU, no DOM), shared by the viewport and the GLB exporter. */

export interface ObjectFrame {
  width: number;
  height: number;
  gsd: number;
  /** Scene base height (world y = 0), metres. */
  base: number;
  /** Absolute ground height under a pixel-centre position (the flattened surface in objects mode). */
  groundAt: (col: number, row: number) => number;
  /** Heights are absolute elevations (an anchored DSM), so a roof is the ground under it plus the building's height.
   *  Off (nDSM / rDSM, where ground is ~0 and `base` is the scene minimum) a roof height is used as is. */
  roofOnGround?: boolean;
  /** Height above ground of the model's raw height map at a pixel-centre position. Lets a footprint whose heights
   *  vary a lot (a tower beside low blocks) be built stepped instead of as one prism at its median. */
  aglAt?: (col: number, row: number) => number;
}

export interface MeshArrays {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  index: Uint32Array;
}

/** Window pitch of the facade texture: one tile per bay (m) and per storey (m). */
export const FACADE_BAY_M = 3.2;
export const FACADE_STOREY_M = 3.0;
/** Walls start this far below the lowest ground sample, so a footprint on a slope never shows a gap. */
const BURY_M = 0.3;
/** Water sits this far above its shoreline ground, clear of z-fighting with the flattened mesh. */
const WATER_LIFT_M = 0.05;

class Builder {
  p: number[] = [];
  n: number[] = [];
  t: number[] = [];
  i: number[] = [];
  vertex(x: number, y: number, z: number, nx: number, ny: number, nz: number, u: number, v: number) {
    this.p.push(x, y, z);
    this.n.push(nx, ny, nz);
    this.t.push(u, v);
    return this.p.length / 3 - 1;
  }
  done(): MeshArrays {
    return { positions: new Float32Array(this.p), normals: new Float32Array(this.n), uvs: new Float32Array(this.t), index: new Uint32Array(this.i) };
  }
}

const worldX = (f: ObjectFrame, px: number) => (px - f.width / 2) * f.gsd;
const worldZ = (f: ObjectFrame, py: number) => (py - f.height / 2) * f.gsd;

/** Lowest ground under a ring (vertices + centroid), absolute metres; NaN-safe. */
function groundUnder(poly: GridPoint[], f: ObjectFrame) {
  let lo = Infinity;
  let cx = 0;
  let cy = 0;
  for (const [x, y] of poly) {
    const g = f.groundAt(x - 0.5, y - 0.5);
    if (Number.isFinite(g)) lo = Math.min(lo, g);
    cx += x / poly.length;
    cy += y / poly.length;
  }
  const g = f.groundAt(cx - 0.5, cy - 0.5);
  if (Number.isFinite(g)) lo = Math.min(lo, g);
  return Number.isFinite(lo) ? lo : f.base;
}

/** Horizontal polygon at world height `y`, facing up; uv = image position (u = px/W, v = py/H). */
function cap(b: Builder, poly: GridPoint[], y: number, f: ObjectFrame) {
  const contour = poly.map(([x, z]) => new Vector2(worldX(f, x), worldZ(f, z)));
  const tris = ShapeUtils.triangulateShape(contour, []);
  if (!tris.length) return false;
  const start = b.p.length / 3;
  poly.forEach(([px, py], k) => b.vertex(contour[k].x, y, contour[k].y, 0, 1, 0, px / f.width, py / f.height));
  for (const [a, c, d] of tris) {
    const A = contour[a];
    const C = contour[c];
    const D = contour[d];
    // y component of (C − A) × (D − A): positive means this winding faces +y
    const up = (C.y - A.y) * (D.x - A.x) - (C.x - A.x) * (D.y - A.y);
    if (up >= 0) b.i.push(start + a, start + c, start + d);
    else b.i.push(start + a, start + d, start + c);
  }
  return true;
}

/** Cell height quantum and size of the stepped roofs, metres. */
const STEP_M = 2;
const CELL_M = 2.4;
const MAX_CELLS = 6000;
/** Cells of margin read round a footprint, so a roof edge's ramp can be followed down to the ground. */
const MARGIN_CELLS = 3;
/** Slope (m per m) above which a cell is on the ramp the height head smears a roof edge into, not on a roof. */
const RAMP_SLOPE = 1;

function inRing(poly: GridPoint[], x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

interface Stepped {
  x0: number;
  y0: number;
  c: number;
  nx: number;
  ny: number;
  /** Height above ground per cell (quantised), NaN outside the footprint. */
  h: Float32Array;
}

/** Undo the blur of the height head on a footprint's cell heights (`v`, NaN where unknown; `inside` marks the
 *  footprint's cells; `cellM` is the cell size in metres).
 *
 *  The head smears each flat roof into a dome and each roof edge into a ramp, so a tower reaches its real height
 *  only at its apex. Cells are split into roofs (gentle slope) and ramps (steep). Each connected roof takes a high
 *  percentile of its heights, so a dome's top reads as the tower's peak. A ramp cell is traced uphill to the roof
 *  it rises to and downhill to the surface it falls to, and joins whichever it is nearer in height: a blurred step
 *  crosses that midpoint where the real edge is. A ramp that falls away outside the footprint is the building's own
 *  outer wall, so it joins the roof. Every cell stays with its *own* tower: nothing is compared with the highest
 *  roof nearby, which may be a different, taller tower. */
export function sharpenRoof(v: Float32Array, inside: Uint8Array, nx: number, ny: number, cellM: number): Float32Array {
  const N = nx * ny;
  const out = new Float32Array(v);
  const ok = (k: number) => Number.isFinite(v[k]);
  const at = (i: number, j: number) => (i >= 0 && j >= 0 && i < nx && j < ny && ok(j * nx + i) ? v[j * nx + i] : NaN);
  // central-difference slope, one-sided where a neighbour is unknown
  const ramp = new Uint8Array(N);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (!ok(k)) continue;
      const d = (a: number, b: number, span: number) => (Number.isFinite(a) && Number.isFinite(b) ? (a - b) / (span * cellM) : 0);
      const l = at(i - 1, j);
      const r = at(i + 1, j);
      const u = at(i, j - 1);
      const w = at(i, j + 1);
      const gx = Number.isFinite(l) && Number.isFinite(r) ? d(r, l, 2) : Number.isFinite(r) ? d(r, v[k], 1) : d(v[k], l, 1);
      const gy = Number.isFinite(u) && Number.isFinite(w) ? d(w, u, 2) : Number.isFinite(w) ? d(w, v[k], 1) : d(v[k], u, 1);
      ramp[k] = Math.hypot(gx, gy) > RAMP_SLOPE ? 1 : 0;
    }
  }
  // roofs: connected gentle cells inside the footprint whose neighbours differ by a gentle step
  const region = new Int32Array(N).fill(-1);
  const level: number[] = [];
  const stack: number[] = [];
  for (let k0 = 0; k0 < N; k0++) {
    if (!inside[k0] || ramp[k0] || !ok(k0) || region[k0] >= 0) continue;
    const id = level.length;
    const members: number[] = [];
    region[k0] = id;
    stack.push(k0);
    while (stack.length) {
      const k = stack.pop()!;
      members.push(v[k]);
      const i = k % nx;
      const j = (k - i) / nx;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di;
        const jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
        const n = jj * nx + ii;
        if (region[n] >= 0 || !inside[n] || ramp[n] || !ok(n) || Math.abs(v[n] - v[k]) > RAMP_SLOPE * cellM) continue;
        region[n] = id;
        stack.push(n);
      }
    }
    members.sort((a, b) => a - b);
    level.push(members[Math.min(members.length - 1, Math.floor(members.length * 0.9))]);
  }
  // steepest walk from k, uphill (sign 1, inside the footprint only) or downhill (sign −1, anywhere), until a
  // gentle cell or an extremum; returns [height reached, whether it ended outside the footprint]
  const walk = (k: number, sign: 1 | -1): [number, boolean] => {
    for (let guard = 0; guard < N; guard++) {
      if (!ramp[k]) return [region[k] >= 0 ? level[region[k]] : v[k], !inside[k]];
      const i = k % nx;
      const j = (k - i) / nx;
      let best = -1;
      let bestV = v[k];
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          const ii = i + di;
          const jj = j + dj;
          if ((!di && !dj) || ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
          const n = jj * nx + ii;
          if (!ok(n) || (sign > 0 && !inside[n])) continue;
          if (sign * (v[n] - bestV) > 0) {
            best = n;
            bestV = v[n];
          }
        }
      }
      if (best < 0) return [v[k], !inside[k]];
      k = best;
    }
    return [v[k], !inside[k]];
  };
  for (let k = 0; k < N; k++) {
    if (!inside[k] || !ok(k)) continue;
    if (!ramp[k]) {
      if (region[k] >= 0) out[k] = level[region[k]];
      continue;
    }
    const [top] = walk(k, 1);
    const [foot, outside] = walk(k, -1);
    out[k] = outside || v[k] >= (top + foot) / 2 ? Math.max(top, v[k]) : Math.min(foot, v[k]);
  }
  return out;
}

/** The footprint's roof from the height map: `roof` is the sharpened median height above ground (NaN when the
 *  footprint has too few cells to say), `stepped` its cells when their heights vary too much for one prism. */
function roofCells(poly: GridPoint[], f: ObjectFrame): { roof: number; stepped: Stepped | null } {
  const none = { roof: NaN, stepped: null };
  if (!f.aglAt) return none;
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const [x, y] of poly) {
    x0 = Math.min(x0, x);
    x1 = Math.max(x1, x);
    y0 = Math.min(y0, y);
    y1 = Math.max(y1, y);
  }
  let c = Math.max(1, Math.round(CELL_M / f.gsd));
  while (Math.ceil((x1 - x0) / c) * Math.ceil((y1 - y0) / c) > MAX_CELLS) c *= 2;
  const nx = Math.ceil((x1 - x0) / c);
  const ny = Math.ceil((y1 - y0) / c);
  // a margin of cells round the footprint, so the sharpening sees the ground a roof edge ramps down to
  const reach = MARGIN_CELLS;
  const mx = nx + 2 * reach;
  const my = ny + 2 * reach;
  const raw = new Float32Array(mx * my).fill(NaN);
  const inside = new Uint8Array(mx * my);
  const q = c / 4;
  const sample = (cx: number, cy: number) => {
    const s = [0, -q, q].flatMap((dx) => [0, -q, q].map((dy) => f.aglAt!(cx + dx - 0.5, cy + dy - 0.5))).filter(Number.isFinite).sort((a, b) => a - b);
    return s.length ? s[s.length >> 1] : NaN;
  };
  for (let j = 0; j < my; j++) {
    for (let i = 0; i < mx; i++) {
      const cx = x0 + (i - reach + 0.5) * c;
      const cy = y0 + (j - reach + 0.5) * c;
      const k = j * mx + i;
      inside[k] = i >= reach && j >= reach && i < reach + nx && j < reach + ny && inRing(poly, cx, cy) ? 1 : 0;
      // outside the grid there is nothing to read (sampleBilinear would clamp to the edge)
      if (cx < 0 || cy < 0 || cx > f.width || cy > f.height) continue;
      raw[k] = sample(cx, cy);
    }
  }
  const sharp = sharpenRoof(raw, inside, mx, my, c * f.gsd);
  const h = new Float32Array(nx * ny).fill(NaN);
  const vals: number[] = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = (j + reach) * mx + i + reach;
      if (!inside[k]) continue;
      const v = Number.isFinite(sharp[k]) ? sharp[k] : 0;
      h[j * nx + i] = Math.max(1, Math.round(v / STEP_M) * STEP_M);
      vals.push(v);
    }
  }
  if (vals.length < 4) return none;
  vals.sort((a, b) => a - b);
  const roof = vals[vals.length >> 1];
  const q5 = Math.max(1, Math.round(vals[Math.floor(vals.length * 0.05)] / STEP_M) * STEP_M);
  const q95 = Math.max(1, Math.round(vals[Math.floor(vals.length * 0.95)] / STEP_M) * STEP_M);
  // uniform enough: one prism at the median is cleaner and cheaper
  if (q95 - q5 < Math.max(4, 0.3 * roof)) return { roof, stepped: null };
  return { roof, stepped: cleanSteps({ x0, y0, c, nx, ny, h }) };
}

/** Post-process the cell heights so the stepped roof reads as a building, not as noise: a rare level joins the
 *  nearest common one only when that is close in height (a far one may be another tower of the block), the
 *  tallest level is always kept, and single stray cells merge into their neighbours. */
function cleanSteps(s: Stepped): Stepped | null {
  const { nx, ny } = s;
  const inside = (i: number, j: number) => i >= 0 && j >= 0 && i < nx && j < ny && !Number.isNaN(s.h[j * nx + i]);
  let h = s.h;
  // 1. roof levels: those covering a real share of the footprint, plus the tallest
  const count = new Map<number, number>();
  let total = 0;
  let top = -Infinity;
  for (const v of h) {
    if (Number.isNaN(v)) continue;
    count.set(v, (count.get(v) ?? 0) + 1);
    total++;
    if (v > top) top = v;
  }
  const keep = [...count]
    .filter(([v, n]) => v === top || n >= Math.max(2, 0.03 * total))
    .map(([v]) => v)
    .sort((a, b) => a - b);
  const snapped = new Float32Array(h);
  for (let k = 0; k < snapped.length; k++) {
    if (Number.isNaN(h[k])) continue;
    let near = h[k];
    let gap = Math.max(2 * STEP_M, 0.1 * h[k]);
    for (const l of keep) {
      const d = Math.abs(l - h[k]);
      // on a tie prefer the higher level: the head reads roofs low, not high
      if (d < gap || (d === gap && l > near)) [near, gap] = [l, d];
    }
    snapped[k] = near;
  }
  h = snapped;
  // 2. merge single stray cells into the most common neighbouring level
  const label = new Int32Array(h.length).fill(-1);
  let next = 0;
  for (let k = 0; k < h.length; k++) {
    if (Number.isNaN(h[k]) || label[k] >= 0) continue;
    const members: number[] = [k];
    label[k] = next;
    for (let m = 0; m < members.length; m++) {
      const idx = members[m];
      const i = idx % nx;
      const j = (idx - i) / nx;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        if (!inside(i + di, j + dj)) continue;
        const n = (j + dj) * nx + i + di;
        if (label[n] < 0 && h[n] === h[k]) {
          label[n] = next;
          members.push(n);
        }
      }
    }
    // a lone cell of the tallest level stays: it is the building's real height, not noise
    if (members.length < 2 && h[k] !== top) {
      const votes = new Map<number, number>();
      for (const idx of members) {
        const i = idx % nx;
        const j = (idx - i) / nx;
        for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          if (!inside(i + di, j + dj)) continue;
          const v = h[(j + dj) * nx + i + di];
          if (v !== h[k]) votes.set(v, (votes.get(v) ?? 0) + 1);
        }
      }
      let best = NaN;
      let most = 0;
      for (const [v, n] of votes) if (n > most) [best, most] = [v, n];
      if (!Number.isNaN(best)) for (const idx of members) h[idx] = best;
    }
    next++;
  }
  if (new Set([...h].filter((v) => !Number.isNaN(v))).size < 2) return null;
  return { ...s, h };
}

/** Roof cells and the walls between cells of different height (and down to the ground at the footprint edge). */
function emitStepped(roof: Builder, walls: Builder, s: Stepped, f: ObjectFrame, yBase: number) {
  const roofY = (hc: number) => (f.roofOnGround ? yBase + BURY_M + hc : hc - f.base);
  const cellH = (i: number, j: number) => (i < 0 || j < 0 || i >= s.nx || j >= s.ny ? NaN : s.h[j * s.nx + i]);
  const wall = (xa: number, za: number, xb: number, zb: number, yLo: number, yHi: number, nx: number, nz: number) => {
    const len = Math.hypot(xb - xa, zb - za);
    if (len < 1e-6 || yHi - yLo < 1e-3) return;
    const u0 = (xa + za) / FACADE_BAY_M;
    const u1 = u0 + len / FACADE_BAY_M;
    const v0 = (yLo - yBase) / FACADE_STOREY_M;
    const v1 = (yHi - yBase) / FACADE_STOREY_M;
    const a = walls.vertex(xa, yLo, za, nx, 0, nz, u0, v0);
    const b = walls.vertex(xb, yLo, zb, nx, 0, nz, u1, v0);
    const c = walls.vertex(xb, yHi, zb, nx, 0, nz, u1, v1);
    const d = walls.vertex(xa, yHi, za, nx, 0, nz, u0, v1);
    if (-(zb - za) * nx + (xb - xa) * nz >= 0) walls.i.push(a, b, c, a, c, d);
    else walls.i.push(a, c, b, a, d, c);
  };
  for (let j = 0; j < s.ny; j++) {
    for (let i = 0; i < s.nx; i++) {
      const hc = cellH(i, j);
      if (Number.isNaN(hc)) continue;
      const pxa = s.x0 + i * s.c;
      const pya = s.y0 + j * s.c;
      const xa = worldX(f, pxa);
      const xb = worldX(f, pxa + s.c);
      const za = worldZ(f, pya);
      const zb = worldZ(f, pya + s.c);
      const y = roofY(hc);
      const start = roof.p.length / 3;
      const uv = (px: number, py: number) => [px / f.width, py / f.height] as const;
      roof.vertex(xa, y, za, 0, 1, 0, ...uv(pxa, pya));
      roof.vertex(xb, y, za, 0, 1, 0, ...uv(pxa + s.c, pya));
      roof.vertex(xb, y, zb, 0, 1, 0, ...uv(pxa + s.c, pya + s.c));
      roof.vertex(xa, y, zb, 0, 1, 0, ...uv(pxa, pya + s.c));
      roof.i.push(start, start + 2, start + 1, start, start + 3, start + 2);
      const low = (ni: number, nj: number) => {
        const nh = cellH(ni, nj);
        return Number.isNaN(nh) ? yBase : Math.min(y, roofY(nh));
      };
      if (Number.isNaN(cellH(i + 1, j)) || cellH(i + 1, j) < hc) wall(xb, za, xb, zb, low(i + 1, j), y, 1, 0);
      if (Number.isNaN(cellH(i - 1, j)) || cellH(i - 1, j) < hc) wall(xa, zb, xa, za, low(i - 1, j), y, -1, 0);
      if (Number.isNaN(cellH(i, j + 1)) || cellH(i, j + 1) < hc) wall(xb, zb, xa, zb, low(i, j + 1), y, 0, 1);
      if (Number.isNaN(cellH(i, j - 1)) || cellH(i, j - 1) < hc) wall(xa, za, xb, za, low(i, j - 1), y, 0, -1);
    }
  }
}

/** Buildings as flat-roofed prisms: `roof` (uv = the optical image, so each roof shows the real one) and
 *  `walls` (uv in facade tiles: u along the perimeter per bay, v up per storey). */
export function buildingGeometry(buildings: BuildingObject[], f: ObjectFrame): { roof: MeshArrays; walls: MeshArrays } {
  const roof = new Builder();
  const walls = new Builder();
  for (const bld of buildings) {
    const poly = bld.poly;
    if (poly.length < 3) continue;
    const yBase = groundUnder(poly, f) - f.base - BURY_M;
    // the roof is read from the height map, sharpened, so a tower stands at its real height rather than at the
    // median of its blurred dome; the backend's median (`bld.h`) is the fallback. Never below the ground it stands on.
    const cells = roofCells(poly, f);
    if (cells.stepped) {
      emitStepped(roof, walls, cells.stepped, f, yBase);
      continue;
    }
    const hRoof = Number.isFinite(cells.roof) ? Math.max(cells.roof, bld.h) : bld.h;
    const yTop = f.roofOnGround ? yBase + BURY_M + Math.max(hRoof, 1) : Math.max(hRoof - f.base, yBase + BURY_M + 1);
    if (!cap(roof, poly, yTop, f)) continue;
    const outward = signedArea(poly) < 0 ? -1 : 1;
    let along = 0;
    for (let k = 0; k < poly.length; k++) {
      const [px0, py0] = poly[k];
      const [px1, py1] = poly[(k + 1) % poly.length];
      const x0 = worldX(f, px0);
      const z0 = worldZ(f, py0);
      const x1 = worldX(f, px1);
      const z1 = worldZ(f, py1);
      const len = Math.hypot(x1 - x0, z1 - z0);
      if (len < 1e-6) continue;
      // outward normal: (dz, −dx) for a positively oriented ring, flipped otherwise
      const nx = (outward * (z1 - z0)) / len;
      const nz = (-outward * (x1 - x0)) / len;
      const u0 = along / FACADE_BAY_M;
      const u1 = (along + len) / FACADE_BAY_M;
      const vTop = (yTop - yBase) / FACADE_STOREY_M;
      const a = walls.vertex(x0, yBase, z0, nx, 0, nz, u0, 0);
      const b = walls.vertex(x1, yBase, z1, nx, 0, nz, u1, 0);
      const c = walls.vertex(x1, yTop, z1, nx, 0, nz, u1, vTop);
      const d = walls.vertex(x0, yTop, z0, nx, 0, nz, u0, vTop);
      // (b − a) × (c − a) is (−dz, 0, dx)·h: keep the winding whose face normal matches the outward one
      const fx = -(z1 - z0);
      const fz = x1 - x0;
      if (fx * nx + fz * nz >= 0) walls.i.push(a, b, c, a, c, d);
      else walls.i.push(a, c, b, a, d, c);
      along += len;
    }
  }
  return { roof: roof.done(), walls: walls.done() };
}

/** Signed area in world (x, z) — same orientation as grid (px, py). Positive = counter-clockwise with z up-page. */
function signedArea(poly: GridPoint[]) {
  let a = 0;
  for (let k = 0; k < poly.length; k++) {
    const [x0, y0] = poly[k];
    const [x1, y1] = poly[(k + 1) % poly.length];
    a += x0 * y1 - x1 * y0;
  }
  return a / 2;
}

/** Water bodies as flat surfaces just above their shoreline ground. */
export function waterGeometry(water: WaterObject[], f: ObjectFrame): MeshArrays {
  const b = new Builder();
  for (const w of water) {
    if (w.poly.length < 3) continue;
    cap(b, w.poly, groundUnder(w.poly, f) - f.base + WATER_LIFT_M, f);
  }
  return b.done();
}
