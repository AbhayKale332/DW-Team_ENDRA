/** Telecom tower coverage from a DSM: a planning-grade radio estimate, not a drive-test.
 *
 *  Per pixel:  PL = FSPL(f, d) + 10 (n - 2) log10(d / 100 m) + J(v) [+ NLOS clutter] [+ in-building]
 *              RSRP ~= EIRP - 28 dB - PL          (28 dB ~ per-resource-element share of a 10 MHz carrier)
 *
 *  J(v) is the ITU-R P.526 single knife-edge diffraction loss over the dominant obstacle on the path, found
 *  from the running maximum elevation angle along the ray from the antenna. The obstacles are the model's surface
 *  (DSM), so buildings and terrain really block the signal. No antenna pattern, foliage or reflection modelling. */
import type { AnalysisGrid } from './grid';

export type Environment = 'rural' | 'suburban' | 'urban';

export interface Tower {
  id: string;
  /** Scene-grid pixel-centre coordinates. */
  col: number;
  row: number;
  /** Antenna height above the surface at its position (a roof counts as the surface), metres. */
  heightM: number;
  eirpDbm: number;
}

export interface TelecomParams {
  freqMHz: number;
  /** Receiver height above the surface, metres. */
  rxHeightM: number;
  env: Environment;
}

export const ENV_MODEL: Record<Environment, { n: number; nlosDb: number; indoorDb: number; label: string }> = {
  rural: { n: 2.2, nlosDb: 6, indoorDb: 10, label: 'Rural / open' },
  suburban: { n: 2.7, nlosDb: 12, indoorDb: 15, label: 'Suburban' },
  urban: { n: 3.2, nlosDb: 18, indoorDb: 20, label: 'Dense urban' },
};

export const FREQ_PRESETS = [700, 900, 1800, 2100, 3500];
/** Class limits on the estimated per-resource-element level, dBm. */
export const CLASS_LIMITS = { strong: -85, fair: -95, weak: -105 };
export const CLASS_LABELS = ['No service', 'Weak', 'Fair', 'Strong'] as const;
export const RE_OFFSET_DB = 28;

export const DEFAULT_TOWER = { heightM: 25, eirpDbm: 46 };
export const DEFAULT_PARAMS: TelecomParams = { freqMHz: 1800, rxHeightM: 1.5, env: 'urban' };

/** Free-space path loss, dB (f in MHz, d in metres). */
export const fspl = (fMHz: number, dM: number) => 32.44 + 20 * Math.log10(fMHz) + 20 * Math.log10(Math.max(dM, 1) / 1000);

/** ITU-R P.526 single knife-edge loss, dB, for Fresnel-Kirchhoff parameter v. */
export function knifeEdge(v: number): number {
  if (v <= -0.78) return 0;
  const a = v - 0.1;
  return 6.9 + 20 * Math.log10(Math.sqrt(a * a + 1) + a);
}

const wavelength = (fMHz: number) => 299.792458 / fMHz;

/** Perimeter cells, each also at a half-cell offset along the border: ~2 rays per border pixel leaves no gaps. */
function borderTargets(w: number, h: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let x = 0; x < w; x++) {
    out.push([x, 0], [x + 0.5, 0], [x, h - 1], [x + 0.5, h - 1]);
  }
  for (let y = 1; y < h - 1; y++) {
    out.push([0, y], [0, y + 0.5], [w - 1, y], [w - 1, y + 0.5]);
  }
  return out;
}

export interface Sweep {
  /** Smallest knife-edge loss over the rays that reached each cell, dB. Infinity = not reached. */
  loss: Float32Array;
  /** Smallest clearance of the dominant edge over the sight line (> 0 = blocked); -Infinity = nothing in the way. */
  clearance: Float32Array;
}

/** Radial sweep from (tx, ty) over `surf` (analysis-grid pixel coordinates, integers). */
export function sweep(w: number, h: number, cellM: number, surf: Float32Array, tx: number, ty: number, zTx: number, rxH: number, freqMHz: number): Sweep {
  const loss = new Float32Array(w * h).fill(Infinity);
  const clearance = new Float32Array(w * h).fill(Infinity);
  const lambda = wavelength(freqMHz);
  const minD = cellM * 0.5;
  for (const [ex, ey] of borderTargets(w, h)) {
    const dx = ex - tx;
    const dy = ey - ty;
    const len = Math.max(Math.abs(dx), Math.abs(dy));
    if (len < 1) continue;
    const steps = Math.ceil(len);
    const sx = dx / len;
    const sy = dy / len;
    let maxTan = -Infinity;
    let dE = 0;
    let zE = 0;
    let last = -1;
    for (let i = 1; i <= steps; i++) {
      const cx = Math.round(tx + sx * i);
      const cy = Math.round(ty + sy * i);
      if (cx < 0 || cy < 0 || cx >= w || cy >= h) break;
      const idx = cy * w + cx;
      if (idx === last) continue;
      last = idx;
      const s = surf[idx];
      if (!Number.isFinite(s)) continue;
      const d = Math.hypot(cx - tx, cy - ty) * cellM;
      if (d < minD) continue;
      const zRx = s + rxH;
      let j = 0;
      let hc = -Infinity;
      if (maxTan > -Infinity) {
        hc = zE - (zTx + ((zRx - zTx) * dE) / d);
        const dd = Math.max(d - dE, minD);
        j = knifeEdge(hc * Math.sqrt((2 * d) / (lambda * dE * dd)));
      }
      if (j < loss[idx]) loss[idx] = j;
      if (hc < clearance[idx]) clearance[idx] = hc;
      const tan = (s - zTx) / d;
      if (tan > maxTan) {
        maxTan = tan;
        dE = d;
        zE = s;
      }
    }
  }
  return { loss, clearance };
}

export interface TowerField {
  /** Estimated level, dBm; -Infinity where unreachable. */
  rsrp: Float32Array;
  /** 0 line of sight, 1 blocked by structures (or by anything, without terrain), 2 blocked by terrain. */
  nlos: Uint8Array;
}

/** Signal field of one tower. `terrainSweep` decides whether a blocked path is blocked by terrain or structures. */
export function towerField(grid: AnalysisGrid, tower: Tower, p: TelecomParams, withTerrainSplit = true): TowerField {
  const { w, h, cellM, surface, ground } = grid;
  const [tx, ty] = [Math.round((tower.col + 0.5) / grid.f - 0.5), Math.round((tower.row + 0.5) / grid.f - 0.5)];
  const cx = Math.min(w - 1, Math.max(0, tx));
  const cy = Math.min(h - 1, Math.max(0, ty));
  const base = surface[cy * w + cx];
  const zTx = (Number.isFinite(base) ? base : ground[cy * w + cx]) + tower.heightM;
  const env = ENV_MODEL[p.env];
  const sw = sweep(w, h, cellM, surface, cx, cy, zTx, p.rxHeightM, p.freqMHz);
  const tsw = withTerrainSplit && grid.hasTerrain ? sweep(w, h, cellM, ground, cx, cy, zTx, p.rxHeightM, p.freqMHz) : null;
  const rsrp = new Float32Array(w * h).fill(-Infinity);
  const nlos = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!Number.isFinite(surface[i])) continue;
      const near = x === cx && y === cy;
      const d = near ? cellM * 0.5 : Math.hypot(x - cx, y - cy) * cellM;
      const j = near ? 0 : sw.loss[i];
      if (!Number.isFinite(j)) continue;
      const blocked = !near && sw.clearance[i] > 0;
      let pl = fspl(p.freqMHz, d) + Math.max(0, 10 * (env.n - 2) * Math.log10(d / 100)) + j;
      if (blocked) pl += env.nlosDb;
      if (grid.building && grid.building[i]) pl += env.indoorDb;
      rsrp[i] = tower.eirpDbm - RE_OFFSET_DB - pl;
      if (blocked) nlos[i] = tsw && tsw.clearance[i] > 0 ? 2 : 1;
    }
  }
  return { rsrp, nlos };
}

export const classOf = (dbm: number) => (dbm >= CLASS_LIMITS.strong ? 3 : dbm >= CLASS_LIMITS.fair ? 2 : dbm >= CLASS_LIMITS.weak ? 1 : 0);

export interface DeadZone {
  /** Analysis-grid centroid. */
  x: number;
  y: number;
  areaM2: number;
}

export interface CoverageSummary {
  /** Fraction of valid pixels per class: [none, weak, fair, strong]. */
  share: [number, number, number, number];
  /** Fraction of valid pixels with no line of sight to their best server, split by cause. */
  blockedStructures: number;
  blockedTerrain: number;
  /** Poorly served (none/weak) pixels that do have line of sight: distance-limited rather than blocked. */
  distanceLimited: number;
  /** Largest no-service patches, biggest first. */
  deadZones: DeadZone[];
  areaM2: number;
}

export interface CoverageResult {
  w: number;
  h: number;
  /** 0..3, or 255 where there is no data. */
  cls: Uint8Array;
  nlos: Uint8Array;
  rsrp: Float32Array;
  /** Index into the tower list of the best server; 255 = none. */
  server: Uint8Array;
  summary: CoverageSummary;
}

function deadZones(cls: Uint8Array, w: number, h: number, cellM: number, top = 3): DeadZone[] {
  const seen = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  const zones: DeadZone[] = [];
  for (let s = 0; s < cls.length; s++) {
    if (seen[s] || cls[s] !== 0) continue;
    let n = 0;
    let sx = 0;
    let sy = 0;
    let sp = 0;
    stack[sp++] = s;
    seen[s] = 1;
    while (sp) {
      const i = stack[--sp];
      const x = i % w;
      const y = (i - x) / w;
      n++;
      sx += x;
      sy += y;
      const push = (j: number) => {
        if (!seen[j] && cls[j] === 0) {
          seen[j] = 1;
          stack[sp++] = j;
        }
      };
      if (x > 0) push(i - 1);
      if (x < w - 1) push(i + 1);
      if (y > 0) push(i - w);
      if (y < h - 1) push(i + w);
    }
    zones.push({ x: sx / n, y: sy / n, areaM2: n * cellM * cellM });
  }
  return zones.sort((a, b) => b.areaM2 - a.areaM2).slice(0, top);
}

/** Merge tower fields into best-server coverage and summarise it. */
export function combineFields(grid: AnalysisGrid, fields: TowerField[]): CoverageResult {
  const { w, h, cellM, surface } = grid;
  const n = w * h;
  const rsrp = new Float32Array(n).fill(-Infinity);
  const nlos = new Uint8Array(n);
  const server = new Uint8Array(n).fill(255);
  const cls = new Uint8Array(n).fill(255);
  fields.forEach((f, k) => {
    for (let i = 0; i < n; i++) {
      if (f.rsrp[i] > rsrp[i]) {
        rsrp[i] = f.rsrp[i];
        nlos[i] = f.nlos[i];
        server[i] = k;
      }
    }
  });
  const count = [0, 0, 0, 0];
  let valid = 0;
  let bs = 0;
  let bt = 0;
  let dl = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(surface[i])) continue;
    valid++;
    const c = Number.isFinite(rsrp[i]) ? classOf(rsrp[i]) : 0;
    cls[i] = c;
    count[c]++;
    if (nlos[i] === 1) bs++;
    else if (nlos[i] === 2) bt++;
    else if (c <= 1) dl++;
  }
  const v = Math.max(valid, 1);
  return {
    w,
    h,
    cls,
    nlos,
    rsrp,
    server,
    summary: {
      share: [count[0] / v, count[1] / v, count[2] / v, count[3] / v],
      blockedStructures: bs / v,
      blockedTerrain: bt / v,
      distanceLimited: dl / v,
      deadZones: fields.length ? deadZones(cls, w, h, cellM) : [],
      areaM2: valid * cellM * cellM,
    },
  };
}

export function computeCoverage(grid: AnalysisGrid, towers: Tower[], p: TelecomParams): CoverageResult {
  return combineFields(grid, towers.map((t) => towerField(grid, t, p)));
}

export interface SiteSuggestion {
  col: number;
  row: number;
  /** Extra share of the scene that reaches at least "fair" service if a tower is added here. */
  gain: number;
}

/** Greedy siting: score candidate positions near poorly served areas by the area each would newly bring to
 *  "fair" or better, then pick up to `count`, re-scoring against what earlier picks already cover. */
export function suggestSites(grid: AnalysisGrid, towers: Tower[], p: TelecomParams, count = 3, tower: Pick<Tower, 'heightM' | 'eirpDbm'> = DEFAULT_TOWER): SiteSuggestion[] {
  const { w, h, surface } = grid;
  const current = computeCoverage(grid, towers, p);
  const bad = (i: number) => current.cls[i] === 0 || current.cls[i] === 1;
  // cells within `reach` of a poorly served cell
  const reach = Math.max(4, Math.round(w / 16));
  const near = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!bad(y * w + x)) continue;
      for (let yy = Math.max(0, y - reach); yy <= Math.min(h - 1, y + reach); yy += 2) {
        for (let xx = Math.max(0, x - reach); xx <= Math.min(w - 1, x + reach); xx += 2) near[yy * w + xx] = 1;
      }
    }
  }
  const stride = Math.max(6, Math.round(w / 22));
  const cand: Array<[number, number]> = [];
  for (let y = Math.floor(stride / 2); y < h; y += stride) {
    for (let x = Math.floor(stride / 2); x < w; x += stride) {
      // snap to the highest surface in the lattice cell: masts go on roofs and hilltops
      let bx = x;
      let by = y;
      let bh = -Infinity;
      for (let yy = y - stride / 2; yy < y + stride / 2; yy++) {
        for (let xx = x - stride / 2; xx < x + stride / 2; xx++) {
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const s = surface[Math.floor(yy) * w + Math.floor(xx)];
          if (Number.isFinite(s) && s > bh) {
            bh = s;
            bx = Math.floor(xx);
            by = Math.floor(yy);
          }
        }
      }
      if (bh > -Infinity && near[by * w + bx]) cand.push([bx, by]);
    }
  }
  if (!cand.length) return [];
  const fields = cand.slice(0, 60).map(([x, y]) => {
    const [col, row] = [(x + 0.5) * grid.f - 0.5, (y + 0.5) * grid.f - 0.5];
    return { col, row, x, y, f: towerField(grid, { id: 'cand', col, row, ...tower }, p, false) };
  });
  const cellFair = (dbm: number) => dbm >= CLASS_LIMITS.fair;
  const covered = new Uint8Array(w * h);
  let valid = 0;
  for (let i = 0; i < covered.length; i++) {
    if (Number.isFinite(surface[i])) valid++;
    if (current.cls[i] >= 2 && current.cls[i] !== 255) covered[i] = 1;
  }
  const picks: SiteSuggestion[] = [];
  const minSep = Math.max(stride * 2, Math.round(w / 8));
  for (let round = 0; round < count; round++) {
    let best = -1;
    let bestGain = 0;
    fields.forEach((c, k) => {
      if (picks.some((s) => Math.hypot(s.col - c.col, s.row - c.row) / grid.f < minSep)) return;
      let g = 0;
      for (let i = 0; i < covered.length; i++) if (!covered[i] && cellFair(c.f.rsrp[i]) && Number.isFinite(surface[i])) g++;
      if (g > bestGain) {
        bestGain = g;
        best = k;
      }
    });
    if (best < 0 || bestGain / valid < 0.005) break;
    const c = fields[best];
    for (let i = 0; i < covered.length; i++) if (cellFair(c.f.rsrp[i])) covered[i] = 1;
    picks.push({ col: c.col, row: c.row, gain: bestGain / valid });
  }
  return picks;
}
