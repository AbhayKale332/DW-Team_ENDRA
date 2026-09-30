/** Flood vulnerability and emergency prioritisation from terrain elevation.
 *
 *  Bathtub model with connectivity. One priority-flood pass from the flood source gives every cell its
 *  *arrival level*: the lowest water level at which it is reached through connected ground,
 *
 *      arrival[c] = max(ground[c], min over neighbours n of arrival[n])        (seeds: their own ground)
 *
 *  so for any water level L, a cell is flooded iff arrival <= L and its depth is L - ground. This is the state the
 *  water settles to at L, and what buildings are ranked by; how it gets there (flow, speed, the advancing front) is
 *  simulated by floodSim.ts. No rainfall, drainage or defences: an indicative screening.
 *
 *  Terrain comes from the DEM-anchored terrain (scene.terrain); without it there is nothing to flood. */
import type { BuildingObject } from '@/domain/types';
import type { Poi, PoiKind } from '../poi';
import type { AnalysisGrid } from './grid';

export type FloodSource = { kind: 'water' } | { kind: 'point'; col: number; row: number } | { kind: 'bathtub' };

export interface FloodModel {
  w: number;
  h: number;
  cellM: number;
  ground: Float32Array;
  /** Lowest water level (absolute, same datum as ground) at which each cell floods; Infinity = never. NaN = no data. */
  arrival: Float32Array;
  /** The "normal" water level the rise is measured from: the source's own elevation. */
  baseLevel: number;
  /** Largest rise offered by the slider, metres. */
  maxRise: number;
  minGround: number;
  maxGround: number;
  source: FloodSource['kind'];
  seeds: number;
  /** 1 on the cells the water comes from; null for the whole-scene level (no single source). */
  seedMask: Uint8Array | null;
}

export type FloodError = { error: string };

/** Min-heap of (key, index) pairs. */
class Heap {
  private keys: number[] = [];
  private vals: number[] = [];
  get size() {
    return this.keys.length;
  }
  push(k: number, v: number) {
    const ks = this.keys;
    const vs = this.vals;
    let i = ks.length;
    ks.push(k);
    vs.push(v);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (ks[p] <= k) break;
      ks[i] = ks[p];
      vs[i] = vs[p];
      i = p;
    }
    ks[i] = k;
    vs[i] = v;
  }
  pop(): [number, number] {
    const ks = this.keys;
    const vs = this.vals;
    const k0 = ks[0];
    const v0 = vs[0];
    const k = ks.pop() as number;
    const v = vs.pop() as number;
    const n = ks.length;
    if (n) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && ks[c + 1] < ks[c]) c++;
        if (ks[c] >= k) break;
        ks[i] = ks[c];
        vs[i] = vs[c];
        i = c;
      }
      ks[i] = k;
      vs[i] = v;
    }
    return [k0, v0];
  }
}

export function buildFloodModel(grid: AnalysisGrid, source: FloodSource): FloodModel | FloodError {
  if (!grid.hasTerrain) return { error: 'Flood mapping needs terrain elevation. Anchor the scene to a DEM first.' };
  const { w, h, ground } = grid;
  const n = w * h;
  let minGround = Infinity;
  let maxGround = -Infinity;
  for (let i = 0; i < n; i++) {
    const g = ground[i];
    if (!Number.isFinite(g)) continue;
    if (g < minGround) minGround = g;
    if (g > maxGround) maxGround = g;
  }
  if (!Number.isFinite(minGround)) return { error: 'The terrain has no valid elevations.' };

  const seeds: number[] = [];
  if (source.kind === 'water') {
    if (grid.water) for (let i = 0; i < n; i++) if (grid.water[i] && Number.isFinite(ground[i])) seeds.push(i);
    if (!seeds.length) return { error: 'No water body was detected in this scene. Pick a source point on the map, or use the whole-scene level.' };
  } else if (source.kind === 'point') {
    const cx = Math.min(w - 1, Math.max(0, Math.round((source.col + 0.5) / grid.f - 0.5)));
    const cy = Math.min(h - 1, Math.max(0, Math.round((source.row + 0.5) / grid.f - 0.5)));
    for (let y = Math.max(0, cy - 1); y <= Math.min(h - 1, cy + 1); y++) for (let x = Math.max(0, cx - 1); x <= Math.min(w - 1, cx + 1); x++) if (Number.isFinite(ground[y * w + x])) seeds.push(y * w + x);
    if (!seeds.length) return { error: 'The chosen point has no terrain data.' };
  }

  let baseLevel = minGround;
  let seedMask: Uint8Array | null = null;
  if (source.kind !== 'bathtub') {
    const sg = seeds.map((i) => ground[i]).sort((a, b) => a - b);
    baseLevel = sg[sg.length >> 1];
    seedMask = new Uint8Array(n);
    for (const s of seeds) seedMask[s] = 1;
  }
  const arrival = arrivalLevels(ground, w, h, source.kind === 'bathtub' ? null : seeds);
  let maxArrival = baseLevel;
  for (let i = 0; i < n; i++) if (Number.isFinite(arrival[i]) && arrival[i] > maxArrival) maxArrival = arrival[i];
  const maxRise = Math.min(30, Math.max(2, Math.ceil(maxArrival - baseLevel + 0.5)));
  return { w, h, cellM: grid.cellM, ground, arrival, baseLevel, maxRise, minGround, maxGround, source: source.kind, seeds: seeds.length, seedMask };
}

/** Priority-flood arrival levels (see the header) from `seeds`; null seeds = every cell is its own source, so the
 *  arrival is the ground itself. NaN ground stays NaN; cells the seeds never reach are Infinity. */
export function arrivalLevels(ground: Float32Array, w: number, h: number, seeds: ArrayLike<number> | null): Float32Array {
  const n = w * h;
  const arrival = new Float32Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) if (!Number.isFinite(ground[i])) arrival[i] = NaN;
  if (!seeds) {
    for (let i = 0; i < n; i++) if (Number.isFinite(ground[i])) arrival[i] = ground[i];
    return arrival;
  }
  const heap = new Heap();
  for (let k = 0; k < seeds.length; k++) {
    const s = seeds[k];
    if (!Number.isFinite(ground[s])) continue;
    arrival[s] = ground[s];
    heap.push(ground[s], s);
  }
  const done = new Uint8Array(n);
  while (heap.size) {
    const [k, i] = heap.pop();
    if (done[i]) continue;
    done[i] = 1;
    const x = i % w;
    const y = (i - x) / w;
    const visit = (j: number) => {
      if (done[j] || !Number.isFinite(ground[j])) return;
      const a = Math.max(ground[j], k);
      if (a < arrival[j]) {
        arrival[j] = a;
        heap.push(a, j);
      }
    };
    if (x > 0) visit(i - 1);
    if (x < w - 1) visit(i + 1);
    if (y > 0) visit(i - w);
    if (y < h - 1) visit(i + w);
  }
  return arrival;
}

export interface FloodStats {
  /** Fraction of valid cells under water at this rise. */
  share: number;
  areaM2: number;
  meanDepthM: number;
  maxDepthM: number;
}

/** Water state at `rise` metres above the base level. */
export function floodStats(m: FloodModel, rise: number): FloodStats {
  const level = m.baseLevel + rise;
  let valid = 0;
  let wet = 0;
  let sum = 0;
  let max = 0;
  for (let i = 0; i < m.arrival.length; i++) {
    const a = m.arrival[i];
    if (Number.isNaN(a)) continue;
    valid++;
    if (a <= level) {
      wet++;
      const d = level - m.ground[i];
      sum += d;
      if (d > max) max = d;
    }
  }
  return { share: valid ? wet / valid : 0, areaM2: wet * m.cellM * m.cellM, meanDepthM: wet ? sum / wet : 0, maxDepthM: max };
}

// ---------------------------------------------------------------------------------------------
// buildings

export type Tier = 'critical' | 'high' | 'watch' | 'none';
export const TIER_LABELS: Record<Tier, string> = { critical: 'Critical', high: 'High', watch: 'Watch', none: 'Not reached' };
export const TIER_COLORS: Record<Tier, string> = { critical: '#e03131', high: '#f76707', watch: '#f2c500', none: '#868e96' };

/** Facilities whose presence raises a building's priority (people who cannot easily self-evacuate, or who respond). */
export const CRITICAL_KINDS: ReadonlySet<PoiKind> = new Set(['hospital', 'clinic', 'school', 'kindergarten', 'university', 'fire_station', 'police', 'ambulance_station']);
export const STOREY_M = 3;
/** Score weights: urgency (how early it floods), occupancy proxy, critical facility. */
export const WEIGHTS = { urgency: 0.5, occupancy: 0.3, critical: 0.2 };
export const TIER_LIMITS = { critical: 60, high: 40 };

export interface BuildingRisk {
  /** Index into scene.objects.buildings. */
  index: number;
  /** Absolute level at which water first touches the footprint; Infinity = never within the model. */
  arrival: number;
  /** Metres of rise above the base level at which it floods. */
  rise: number;
  groundM: number;
  /** Roof height above the ground, metres: water deeper than this submerges the building. */
  heightM: number;
  areaM2: number;
  storeys: number;
  facility: string | null;
  score: number;
  tier: Tier;
  /** Footprint centroid, scene-grid corner-origin pixels. */
  cx: number;
  cy: number;
}

function inRing(ring: Array<[number, number]>, x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Rank buildings by exposure and vulnerability. `pois` may be null (no facilities loaded). */
export function assessBuildings(buildings: BuildingObject[], grid: AnalysisGrid, model: FloodModel, pois: Poi[] | null, gsd: number): BuildingRisk[] {
  const { w, h, f } = grid;
  const out: BuildingRisk[] = [];
  let maxOcc = 1;
  for (let index = 0; index < buildings.length; index++) {
    const b = buildings[index];
    const ring = b.poly.map(([x, y]) => [x / f, y / f] as [number, number]);
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    let cx = 0;
    let cy = 0;
    for (const [x, y] of b.poly) {
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
      cx += x;
      cy += y;
    }
    cx /= b.poly.length;
    cy /= b.poly.length;
    let arrival = Infinity;
    let gSum = 0;
    let gN = 0;
    const scan = (ax: number, ay: number) => {
      const i = ay * w + ax;
      const a = model.arrival[i];
      const g = model.ground[i];
      if (Number.isFinite(g)) {
        gSum += g;
        gN++;
      }
      if (!Number.isNaN(a) && a < arrival) arrival = a;
    };
    for (let ay = Math.max(0, Math.floor(y0 / f)); ay <= Math.min(h - 1, Math.floor(y1 / f)); ay++) {
      for (let ax = Math.max(0, Math.floor(x0 / f)); ax <= Math.min(w - 1, Math.floor(x1 / f)); ax++) if (inRing(ring, ax + 0.5, ay + 0.5)) scan(ax, ay);
    }
    if (!gN) scan(Math.min(w - 1, Math.max(0, Math.floor(cx / f))), Math.min(h - 1, Math.max(0, Math.floor(cy / f))));
    const storeys = Math.max(1, Math.round(b.h / STOREY_M));
    let facility: string | null = null;
    if (pois) {
      const reach = Math.max(30 / gsd, 0);
      for (const p of pois) {
        if (!CRITICAL_KINDS.has(p.kind)) continue;
        const px = p.col + 0.5;
        const py = p.row + 0.5;
        if (px < x0 - reach || px > x1 + reach || py < y0 - reach || py > y1 + reach) continue;
        if (inRing(b.poly, px, py) || Math.hypot(px - cx, py - cy) <= reach) {
          facility = p.name ? `${p.kind.replace(/_/g, ' ')}: ${p.name}` : p.kind.replace(/_/g, ' ');
          break;
        }
      }
    }
    maxOcc = Math.max(maxOcc, b.areaM2 * storeys);
    out.push({ index, arrival, rise: arrival - model.baseLevel, groundM: gN ? gSum / gN : NaN, heightM: b.h, areaM2: b.areaM2, storeys, facility, score: 0, tier: 'none', cx, cy });
  }
  for (const r of out) {
    if (!Number.isFinite(r.arrival) || r.rise > model.maxRise) continue;
    const urgency = 1 - Math.min(1, Math.max(0, r.rise / model.maxRise));
    const occupancy = Math.log1p(r.areaM2 * r.storeys) / Math.log1p(maxOcc);
    r.score = 100 * (WEIGHTS.urgency * urgency + WEIGHTS.occupancy * occupancy + WEIGHTS.critical * (r.facility ? 1 : 0));
    r.tier = r.score >= TIER_LIMITS.critical ? 'critical' : r.score >= TIER_LIMITS.high ? 'high' : 'watch';
  }
  return out;
}

/** Depth of water on a building's ground at `rise`, metres; 0 when dry. */
export function buildingDepth(r: BuildingRisk, model: FloodModel, rise: number): number {
  return r.arrival <= model.baseLevel + rise ? model.baseLevel + rise - r.groundM : 0;
}

/** Depth at a building from the simulated water surface there (`stage`, absolute; NaN = dry), metres. */
export function stageDepth(r: BuildingRisk, stage: number): number {
  return Number.isFinite(stage) ? Math.max(0, stage - r.groundM) : 0;
}
