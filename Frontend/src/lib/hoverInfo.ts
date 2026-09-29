import type { GridPoint, Scene } from '@/domain/types';
import { sampleBilinear, slopeAspectAt } from './heights';
import { formatLonLat, lonLatAt } from './georef';
import type { OsmFeature } from './osm';
import { classLabel } from '@/theme/classes';

/** What is under the pointer: a detected object (building / tree / water), an elevated area the object step
 *  did not extract, or plain surface — with its heights and whatever else is known about it. */

export type HoverKind = 'building' | 'tree' | 'water' | 'elevated' | 'surface';

export interface HoverInfo {
  kind: HoverKind;
  title: string;
  subtitle: string | null;
  /** The model's height at the pointer (raw heights, whatever the mesh shows), metres. */
  height: number;
  rows: Array<[string, string]>;
}

/** Raised at least this far above ground (nDSM metres) counts as an elevated area. */
export const ELEVATED_M = 2;
/** Typical storey height for the floor estimate. */
const STOREY_M = 3;
/** A street / river is "here" within this distance of the pointer, metres. */
const NEAR_LINE_M = 4;

const m = (v: number) => `${v.toFixed(1)} m`;
export const area = (m2: number) => (m2 >= 10_000 ? `${(m2 / 10_000).toFixed(2)} ha` : `${Math.round(m2).toLocaleString()} m²`);

/** Even-odd point-in-polygon. */
export function pointInRing(poly: ReadonlyArray<readonly [number, number]>, x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function bbox(poly: GridPoint[]) {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const [x, y] of poly) {
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x);
    y1 = Math.max(y1, y);
  }
  return [x0, y0, x1, y1] as const;
}

function segDist2(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const dx = bx - ax;
  const dy = by - ay;
  const l = dx * dx + dy * dy;
  const t = l ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l)) : 0;
  return (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2;
}

// ---------------------------------------------------------------------------------------------
// elevated areas: connected pixels ≥ ELEVATED_M, labelled once per scene

interface Regions {
  labels: Int32Array;
  count: number[];
  max: number[];
  sum: number[];
}
const regionCache = new WeakMap<object, Regions | null>();

/** Connected elevated areas (4-neighbour). Null for absolute DSMs without their nDSM, where "above ground" is unknown. */
export function elevatedRegions(scene: Pick<Scene, 'heights' | 'ndsm' | 'product'>): Regions | null {
  if (regionCache.has(scene)) return regionCache.get(scene) ?? null;
  // an anchored scene keeps its above-ground heights; a bare absolute DSM has none
  const above = scene.ndsm ?? (scene.product === 'DSM' ? null : scene.heights);
  if (!above) {
    regionCache.set(scene, null);
    return null;
  }
  const { data, width: W, height: H } = above;
  const labels = new Int32Array(W * H).fill(-1);
  const count: number[] = [];
  const max: number[] = [];
  const sum: number[] = [];
  const stack = new Int32Array(W * H);
  for (let s = 0; s < data.length; s++) {
    if (labels[s] !== -1 || !(data[s] >= ELEVATED_M)) continue;
    const id = count.length;
    let n = 0;
    let hi = -Infinity;
    let tot = 0;
    let top = 0;
    stack[top++] = s;
    labels[s] = id;
    while (top) {
      const i = stack[--top];
      const v = data[i];
      n++;
      tot += v;
      if (v > hi) hi = v;
      const c = i % W;
      const push = (j: number) => {
        if (labels[j] === -1 && data[j] >= ELEVATED_M) {
          labels[j] = id;
          stack[top++] = j;
        }
      };
      if (c > 0) push(i - 1);
      if (c < W - 1) push(i + 1);
      if (i >= W) push(i - W);
      if (i < W * (H - 1)) push(i + W);
    }
    count.push(n);
    max.push(hi);
    sum.push(tot);
  }
  const out = { labels, count, max, sum };
  regionCache.set(scene, out);
  return out;
}

// ---------------------------------------------------------------------------------------------

function osmRows(osm: OsmFeature[], col: number, row: number, gsd: number): Array<[string, string]> {
  const rows: Array<[string, string]> = [];
  const near2 = (NEAR_LINE_M / gsd) ** 2;
  let building: OsmFeature | null = null;
  let line: { f: OsmFeature; d2: number } | null = null;
  for (const f of osm) {
    if (f.closed && (f.kind === 'building' || f.kind === 'water')) {
      if (!building && f.kind === 'building' && pointInRing(f.pts, col, row)) building = f;
      if (f.kind === 'water' && f.name && pointInRing(f.pts, col, row)) rows.push(['OSM water', f.name]);
      continue;
    }
    if (f.kind === 'building') continue;
    for (let i = 0; i + 1 < f.pts.length; i++) {
      const d2 = segDist2(col, row, f.pts[i][0], f.pts[i][1], f.pts[i + 1][0], f.pts[i + 1][1]);
      if (d2 <= near2 && (!line || d2 < line.d2)) line = { f, d2 };
    }
  }
  if (building) {
    const t = building.tags;
    const type = t.building && t.building !== 'yes' ? t.building : null;
    rows.push(['OSM building', building.name ?? type ?? 'unnamed']);
    if (building.name && type) rows.push(['OSM type', type]);
    if (t['building:levels']) rows.push(['OSM levels', t['building:levels']]);
    if (t.height) rows.push(['OSM height', `${t.height} m`]);
    const addr = [t['addr:housenumber'], t['addr:street']].filter(Boolean).join(' ');
    if (addr) rows.push(['Address', addr]);
  }
  if (line) {
    const t = line.f.tags;
    const label = line.f.kind === 'rail' ? 'Railway' : line.f.kind === 'waterway' ? 'Waterway' : 'Street';
    const kind = t.highway ?? t.railway ?? t.waterway ?? '';
    rows.push([label, line.f.name ? `${line.f.name}${kind ? ` (${kind})` : ''}` : kind || 'unnamed']);
  }
  return rows;
}

/** Everything known about grid point (col, row) — pixel-centre coordinates, as the pick tools use. */
export function describeAt(scene: Scene, col: number, row: number, osm?: OsmFeature[] | null): HoverInfo {
  const { width: W, height: H } = scene.heights;
  const gsd = scene.gsd;
  const x = col + 0.5; // objects.json positions are corner-origin
  const y = row + 0.5;
  const height = sampleBilinear(scene.heights, col, row);
  const { slope } = slopeAspectAt(scene.heights, col, row, gsd);
  const pix = Math.min(H - 1, Math.max(0, Math.round(row))) * W + Math.min(W - 1, Math.max(0, Math.round(col)));
  const cls = scene.classes ? classLabel(scene.classes.names[scene.classes.data[pix]], scene.classes.data[pix]) : null;

  const tail: Array<[string, string]> = [
    scene.product === 'DSM' ? ['Elevation here', `${height.toFixed(1)} m a.s.l.`] : ['Height here', scene.product === 'rDSM' ? `${m(height)} (relative)` : m(height)],
    ...(scene.product === 'DSM' && scene.ndsm ? ([['Above ground', m(sampleBilinear(scene.ndsm, col, row))]] as Array<[string, string]>) : []),
    ['Slope', `${slope.toFixed(0)}°`],
  ];
  if (cls) tail.push(['Class map', cls]);
  if (scene.georef) {
    const ll = lonLatAt(scene.georef, col, row);
    if (ll) tail.push(['Location', formatLonLat(ll)]);
  }
  if (osm?.length) tail.push(...osmRows(osm, col, row, gsd));

  const o = scene.objects;
  if (o) {
    for (const b of o.buildings) {
      const [x0, y0, x1, y1] = bbox(b.poly);
      if (x < x0 || x > x1 || y < y0 || y > y1 || !pointInRing(b.poly, x, y)) continue;
      const floors = Math.max(1, Math.round(b.h / STOREY_M));
      return {
        kind: 'building',
        title: 'Building',
        subtitle: `≈ ${floors} floor${floors > 1 ? 's' : ''} (at ${STOREY_M} m each)`,
        height,
        rows: [
          ['Roof height', m(b.h)],
          ['Highest point', m(b.hMax)],
          ['Footprint', `${((x1 - x0) * gsd).toFixed(0)} × ${((y1 - y0) * gsd).toFixed(0)} m · ${area(b.areaM2)}`],
          ...tail,
        ],
      };
    }
    // the crown the pointer is most central in (crowns may overlap)
    let bestI = -1;
    let bestQ = 1;
    for (let i = 0; i < o.trees.length; i++) {
      const t = o.trees[i];
      const rPx = t.r / gsd;
      const q = ((t.x - x) ** 2 + (t.y - y) ** 2) / (rPx * rPx);
      if (q <= bestQ) {
        bestQ = q;
        bestI = i;
      }
    }
    if (bestI >= 0) {
      const t = o.trees[bestI];
      const conifer = t.h / (2 * t.r) >= 1.8;
      return {
        kind: 'tree',
        title: 'Tree',
        subtitle: conifer ? 'Narrow, conifer-like crown' : 'Spreading, broadleaf-like crown',
        height,
        rows: [
          ['Tree height', m(t.h)],
          ['Crown diameter', m(2 * t.r)],
          ['Crown area', area(Math.PI * t.r * t.r)],
          ...tail,
        ],
      };
    }
    for (const w of o.water) {
      const [x0, y0, x1, y1] = bbox(w.poly);
      if (x < x0 || x > x1 || y < y0 || y > y1 || !pointInRing(w.poly, x, y)) continue;
      return { kind: 'water', title: 'Water body', subtitle: null, height, rows: [['Surface area', area(w.areaM2)], ...tail] };
    }
  }

  const regions = height >= ELEVATED_M ? elevatedRegions(scene) : null;
  const id = regions ? regions.labels[pix] : -1;
  if (regions && id >= 0) {
    return {
      kind: 'elevated',
      title: 'Elevated area',
      subtitle: `Raised above ${ELEVATED_M} m, not extracted as an object`,
      height,
      rows: [
        ['Highest point', m(regions.max[id])],
        ['Mean height', m(regions.sum[id] / regions.count[id])],
        ['Extent', area(regions.count[id] * gsd * gsd)],
        ...tail,
      ],
    };
  }
  return { kind: 'surface', title: cls ?? (height < ELEVATED_M ? 'Ground level' : 'Surface'), subtitle: null, height, rows: tail };
}
