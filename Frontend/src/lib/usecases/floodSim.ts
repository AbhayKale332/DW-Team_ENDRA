/** Flood hydraulics: how the water gets to the level the arrival model (flood.ts) settles at.
 *
 *  The local inertial shallow-water scheme of Bates, Horritt & Fewtrell (2010), the one in LISFLOOD-FP, with the
 *  momentum smoothing of de Almeida et al. (2012) that stops it oscillating cell to cell. Water depth h lives on
 *  cells, discharge per unit width q on the faces between them:
 *
 *      q̃ = θ·q + (1 − θ)/2·(q_prev + q_next)                                  the face and its neighbours in line
 *      q ← (q̃ − g·h_flow·Δt·∂η/∂x) / (1 + g·Δt·n²·|q| / h_flow^(7/3))      momentum, Manning friction semi-implicit
 *      h ← h + Δt·(Σq_in − Σq_out) / Δx                                      mass
 *
 *  η = z + h is the water surface and h_flow = max(η_i, η_j) − max(z_i, z_j) the depth that can pass a face.
 *  Δt = α·Δx/√(g·h_max) keeps it stable, and outflows are capped at the water a cell holds so depth never goes
 *  negative. Source cells hold the surface at the flood level (a stage boundary): water pours out of them as the
 *  level rises, runs downhill, fills hollows and settles flat, where it matches the arrival model on this grid.
 *  With no single source (whole-scene level) every local low point is one. Scene edges are closed walls. */
import { arrivalLevels } from './flood';

export const GRAVITY = 9.81;
/** Manning roughness, s/m^(1/3): a floodplain of pasture, streets and buildings. */
export const MANNING_N = 0.035;
/** Courant factor of the time step (Bates et al. recommend 0.7). */
const ALPHA = 0.7;
/** Weight of a face's own momentum against its neighbours' (de Almeida et al.: 0.7 damps checkerboarding). */
const THETA = 0.7;
const DT_MAX_S = 10;
/** Water thinner than this cannot pass a face, metres. */
const H_FLOW_MIN = 1e-3;
/** Depth that counts as wet for display and statistics, metres. */
export const WET_M = 0.01;
/** The simulation grid's longest side: the analysis grid is pooled down to it so a step stays a few ms. */
export const SIM_MAX_SIDE = 256;

export interface SimGrid {
  w: number;
  h: number;
  cellM: number;
  /** Bed elevation, absolute metres; NaN = no data (a wall). */
  ground: Float32Array;
  /** 1 on stage-boundary cells; null = every local low point. */
  sources: Uint8Array | null;
}

/** Analysis cells per simulation cell for a grid of w × h. */
export const simFactor = (w: number, h: number, maxSide = SIM_MAX_SIDE) => Math.max(1, Math.ceil(Math.max(w, h) / maxSide));

/** Mean-pool the analysis ground by `s`; a pooled cell is a source when any of its cells is. */
export function simGridFrom(src: { w: number; h: number; cellM: number; ground: Float32Array; seedMask: Uint8Array | null }, s: number): SimGrid {
  const w = Math.ceil(src.w / s);
  const h = Math.ceil(src.h / s);
  const sum = new Float64Array(w * h);
  const cnt = new Uint32Array(w * h);
  const sources = src.seedMask ? new Uint8Array(w * h) : null;
  for (let y = 0; y < src.h; y++) {
    const row = Math.floor(y / s) * w;
    for (let x = 0; x < src.w; x++) {
      const i = y * src.w + x;
      const g = src.ground[i];
      if (!Number.isFinite(g)) continue;
      const a = row + Math.floor(x / s);
      sum[a] += g;
      cnt[a]++;
      if (sources && src.seedMask![i]) sources[a] = 1;
    }
  }
  const ground = new Float32Array(w * h);
  for (let a = 0; a < w * h; a++) ground[a] = cnt[a] ? sum[a] / cnt[a] : NaN;
  return { w, h, cellM: src.cellM * s, ground, sources };
}

export interface FrameStats {
  wetCells: number;
  validCells: number;
  sumDepthM: number;
  maxDepthM: number;
}

export class FloodSim {
  readonly w: number;
  readonly h: number;
  readonly dx: number;
  /** Bed elevation (0 where there is no data; see `valid`). */
  readonly z: Float32Array;
  readonly valid: Uint8Array;
  readonly depth: Float32Array;
  /** Discharge per unit width across the east face of each cell (m²/s, + = eastward). */
  readonly qx: Float32Array;
  /** Across the south face (+ = southward). */
  readonly qy: Float32Array;
  /** The fluxes at the start of the step, for the neighbour smoothing. */
  private readonly qx0: Float32Array;
  private readonly qy0: Float32Array;
  readonly sources: Int32Array;
  /** Simulated seconds since the last settle(). */
  time = 0;
  private readonly ground: Float32Array;
  private readonly pits: boolean;
  private hMax = 0;
  /** Bounding box of every cell that has held water (it only grows: cells outside never had flow). */
  private bx0 = Infinity;
  private by0 = Infinity;
  private bx1 = -Infinity;
  private by1 = -Infinity;
  private prev: Float32Array;

  constructor(g: SimGrid) {
    const { w, h } = g;
    const n = w * h;
    this.w = w;
    this.h = h;
    this.dx = g.cellM;
    this.ground = g.ground;
    this.z = new Float32Array(n);
    this.valid = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(g.ground[i])) continue;
      this.z[i] = g.ground[i];
      this.valid[i] = 1;
    }
    this.depth = new Float32Array(n);
    this.qx = new Float32Array(n);
    this.qy = new Float32Array(n);
    this.qx0 = new Float32Array(n);
    this.qy0 = new Float32Array(n);
    this.prev = new Float32Array(n);
    this.pits = !g.sources;
    const src: number[] = [];
    for (let i = 0; i < n; i++) {
      if (!this.valid[i]) continue;
      if (g.sources ? g.sources[i] : this.isPit(i)) src.push(i);
    }
    this.sources = Int32Array.from(src);
  }

  private isPit(i: number): boolean {
    const { w, h, z, valid } = this;
    const x = i % w;
    const y = (i - x) / w;
    const lower = (j: number) => valid[j] && z[j] < z[i];
    return !((x > 0 && lower(i - 1)) || (x < w - 1 && lower(i + 1)) || (y > 0 && lower(i - w)) || (y < h - 1 && lower(i + w)));
  }

  private grow(i: number) {
    const x = i % this.w;
    const y = (i - x) / this.w;
    if (x < this.bx0) this.bx0 = x;
    if (x > this.bx1) this.bx1 = x;
    if (y < this.by0) this.by0 = y;
    if (y > this.by1) this.by1 = y;
  }

  /** Water at rest at `level`: every cell the sources reach below it, flat. Clears the flow and the clock. */
  settle(level: number) {
    const arrival = arrivalLevels(this.ground, this.w, this.h, this.pits ? null : this.sources);
    const { depth, z, valid } = this;
    this.qx.fill(0);
    this.qy.fill(0);
    this.bx0 = this.by0 = Infinity;
    this.bx1 = this.by1 = -Infinity;
    this.hMax = 0;
    for (let i = 0; i < depth.length; i++) {
      depth[i] = valid[i] && arrival[i] <= level ? level - z[i] : 0;
      if (depth[i] > 0) this.grow(i);
      if (depth[i] > this.hMax) this.hMax = depth[i];
    }
    for (const s of this.sources) this.grow(s);
    this.time = 0;
  }

  /** Add `m` metres of water to cell i. */
  pour(i: number, m: number) {
    if (!this.valid[i]) return;
    this.depth[i] += m;
    this.grow(i);
    if (this.depth[i] > this.hMax) this.hMax = this.depth[i];
  }

  private applySources(level: number) {
    const { depth, z } = this;
    for (const s of this.sources) {
      const d = level > z[s] ? level - z[s] : 0;
      depth[s] = d;
      if (d > this.hMax) this.hMax = d;
    }
  }

  /** One time step with the stage boundary at `level`. Returns the step length, seconds. */
  step(level: number): number {
    const { w, h, dx, z, valid, depth: d, qx, qy } = this;
    this.applySources(level);
    const dt = Math.min(DT_MAX_S, (ALPHA * dx) / Math.sqrt(GRAVITY * Math.max(this.hMax, 0.01)));
    // the wet box and one ring of dry cells around it, where water can move next
    const x0 = Math.max(0, this.bx0 - 1);
    const x1 = Math.min(w - 1, this.bx1 + 1);
    const y0 = Math.max(0, this.by0 - 1);
    const y1 = Math.min(h - 1, this.by1 + 1);
    if (x1 < x0 || y1 < y0) return dt;
    const gdt = GRAVITY * dt;
    const fric = gdt * MANNING_N * MANNING_N;
    const side = (1 - THETA) / 2;
    const { qx0, qy0 } = this;
    qx0.set(qx);
    qy0.set(qy);

    // momentum: east faces, then south faces (a missing neighbour face counts as this one)
    const xe = Math.min(x1, w - 2);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= xe; x++) {
        const i = y * w + x;
        const j = i + 1;
        if (!valid[i] || !valid[j]) continue;
        const zi = z[i];
        const zj = z[j];
        const ei = zi + d[i];
        const ej = zj + d[j];
        const hf = (ei > ej ? ei : ej) - (zi > zj ? zi : zj);
        if (hf <= H_FLOW_MIN) {
          qx[i] = 0;
          continue;
        }
        const q0 = qx0[i];
        const qs = THETA * q0 + side * ((x > 0 ? qx0[i - 1] : q0) + (x < w - 2 ? qx0[i + 1] : q0));
        qx[i] = (qs - (gdt * hf * (ej - ei)) / dx) / (1 + (fric * Math.abs(q0)) / (hf * hf * Math.cbrt(hf)));
      }
    }
    const ye = Math.min(y1, h - 2);
    for (let y = y0; y <= ye; y++) {
      for (let i = y * w + x0, end = y * w + x1; i <= end; i++) {
        const j = i + w;
        if (!valid[i] || !valid[j]) continue;
        const zi = z[i];
        const zj = z[j];
        const ei = zi + d[i];
        const ej = zj + d[j];
        const hf = (ei > ej ? ei : ej) - (zi > zj ? zi : zj);
        if (hf <= H_FLOW_MIN) {
          qy[i] = 0;
          continue;
        }
        const q0 = qy0[i];
        const qs = THETA * q0 + side * ((y > 0 ? qy0[i - w] : q0) + (y < h - 2 ? qy0[i + w] : q0));
        qy[i] = (qs - (gdt * hf * (ej - ei)) / dx) / (1 + (fric * Math.abs(q0)) / (hf * hf * Math.cbrt(hf)));
      }
    }

    // a cell cannot give more water than it holds: scale its outflows down (each face drains exactly one cell)
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * w + x;
        const e = qx[i];
        const wq = x > 0 ? qx[i - 1] : 0;
        const s = qy[i];
        const nq = y > 0 ? qy[i - w] : 0;
        const out = (e > 0 ? e : 0) - (wq < 0 ? wq : 0) + (s > 0 ? s : 0) - (nq < 0 ? nq : 0);
        if (out <= 0) continue;
        const cap = (d[i] * dx) / dt;
        if (out <= cap) continue;
        const k = cap / out;
        if (e > 0) qx[i] = e * k;
        if (wq < 0) qx[i - 1] = wq * k;
        if (s > 0) qy[i] = s * k;
        if (nq < 0) qy[i - w] = nq * k;
      }
    }

    // mass
    const r = dt / dx;
    let hMax = 0;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * w + x;
        if (!valid[i]) continue;
        const net = (x > 0 ? qx[i - 1] : 0) - qx[i] + (y > 0 ? qy[i - w] : 0) - qy[i];
        if (net === 0) {
          if (d[i] > hMax) hMax = d[i];
          continue;
        }
        const v = d[i] + r * net;
        const nd = v > 0 ? v : 0;
        d[i] = nd;
        if (nd > hMax) hMax = nd;
        if (nd > 0 && (x < this.bx0 || x > this.bx1 || y < this.by0 || y > this.by1)) this.grow(i);
      }
    }
    this.hMax = hMax;
    this.applySources(level);
    this.time += dt;
    return dt;
  }

  /** Step toward `level` for up to `budgetMs` of compute or `maxSimS` of simulated time.
   *  Returns the simulated seconds covered and the largest depth change over them, metres. */
  advance(level: number, budgetMs: number, maxSimS: number): { simS: number; maxChangeM: number } {
    const t0 = performance.now();
    this.prev.set(this.depth);
    let simS = 0;
    do simS += this.step(level);
    while (simS < maxSimS && performance.now() - t0 < budgetMs);
    let maxChangeM = 0;
    const { depth, prev } = this;
    for (let i = 0; i < depth.length; i++) {
      const c = Math.abs(depth[i] - prev[i]);
      if (c > maxChangeM) maxChangeM = c;
    }
    return { simS, maxChangeM };
  }

  /** Total water volume, m³. */
  volume(): number {
    let v = 0;
    for (let i = 0; i < this.depth.length; i++) v += this.depth[i];
    return v * this.dx * this.dx;
  }

  /** Pack the state for the GPU, 4 floats per cell:
   *    R depth (m); −1 where no water touches the cell or its 8 neighbours (nothing to draw there)
   *    G water surface height in the render frame: (absolute ? z : 0) + depth − base. Dry cells next to water get
   *      their neighbours' surface, so the drawn surface runs on flat until the ground rises through it
   *    B, A depth-averaged velocity east / south (m/s)
   *  and read the surface at `probes` (cell indices) into `stages` (absolute; NaN = dry). */
  writeFrame(out: Float32Array, base: number, absolute: boolean, probes?: Int32Array, stages?: Float32Array): FrameStats {
    const { w, h, z, valid, depth: d, qx, qy } = this;
    const surf = (i: number) => (absolute ? z[i] : 0) + d[i] - base;
    let wetCells = 0;
    let validCells = 0;
    let sumDepthM = 0;
    let maxDepthM = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const o = i * 4;
        const di = d[i];
        if (valid[i]) validCells++;
        if (valid[i] && di > WET_M) {
          wetCells++;
          sumDepthM += di;
          if (di > maxDepthM) maxDepthM = di;
          const hu = Math.max(di, 0.05);
          out[o] = di;
          out[o + 1] = surf(i);
          out[o + 2] = (0.5 * ((x > 0 ? qx[i - 1] : 0) + qx[i])) / hu;
          out[o + 3] = (0.5 * ((y > 0 ? qy[i - w] : 0) + qy[i])) / hu;
          continue;
        }
        let near = -Infinity;
        for (let ny = Math.max(0, y - 1); ny <= Math.min(h - 1, y + 1); ny++) {
          for (let nx = Math.max(0, x - 1); nx <= Math.min(w - 1, x + 1); nx++) {
            const j = ny * w + nx;
            if (valid[j] && d[j] > WET_M) near = Math.max(near, surf(j));
          }
        }
        const dry = near === -Infinity;
        out[o] = dry ? -1 : valid[i] ? di : 0;
        out[o + 1] = dry ? (absolute && valid[i] ? z[i] : 0) - base - 2 : near;
        out[o + 2] = 0;
        out[o + 3] = 0;
      }
    }
    if (probes && stages) for (let k = 0; k < probes.length; k++) stages[k] = d[probes[k]] > WET_M ? z[probes[k]] + d[probes[k]] : NaN;
    return { wetCells, validCells, sumDepthM, maxDepthM };
  }
}
