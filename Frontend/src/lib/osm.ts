import proj4 from 'proj4';
import type { Georef } from '@/domain/types';
import { lonLatAt, mapToPixel } from './georef';

/** OpenStreetMap features for a georeferenced scene, via the Overpass API.
 *
 *  Data © OpenStreetMap contributors, ODbL — the overlay must show that attribution while it is on.
 *  Positions are converted to height-grid pixel-centre coordinates (col, row), the same frame as the
 *  probe and pick tools, so everything downstream is CRS-free. */

export type OsmKind = 'building' | 'road' | 'path' | 'rail' | 'water' | 'waterway';

export interface OsmFeature {
  id: number;
  kind: OsmKind;
  name: string | null;
  tags: Record<string, string>;
  /** Grid pixel-centre coordinates; a closed way repeats its first point last (as OSM stores it). */
  pts: Array<[number, number]>;
  closed: boolean;
}

export const OSM_KIND_LABELS: Record<OsmKind, string> = {
  building: 'Buildings',
  road: 'Roads',
  path: 'Paths & tracks',
  rail: 'Railways',
  water: 'Water bodies',
  waterway: 'Rivers & canals',
};

export const OSM_KIND_COLORS: Record<OsmKind, string> = {
  building: '#ff5a36',
  road: '#ffd23f',
  path: '#f4f1e8',
  rail: '#b04adb',
  water: '#2fa8ff',
  waterway: '#2fa8ff',
};

/** Tried in order (see fetchFromMirrors). overpass-api.de answers 406 to every browser (it rejects browser
 *  User-Agents and any Referer), so it is only reachable through the app's own /overpass relay
 *  (server/overpass.mjs). The direct mirrors still work where the app is served without that relay. */
export const OSM_ENDPOINTS: ReadonlyArray<{ label: string; url: string }> = [
  { label: 'overpass-api.de', url: './overpass/overpass-api.de' },
  { label: 'overpass.private.coffee', url: './overpass/overpass.private.coffee' },
  { label: 'overpass.kumi.systems', url: 'https://overpass.kumi.systems/api/interpreter' },
  { label: 'overpass.private.coffee (direct)', url: 'https://overpass.private.coffee/api/interpreter' },
];

/** Larger than this (≈ 0.1 deg², ~30 × 30 km at mid latitudes) is refused: a public Overpass instance is shared. */
export const MAX_BBOX_DEG2 = 0.1;

/** Can this scene be placed on the globe (a CRS we can reproject from, and an affine)? */
export function canGeolocate(g: Georef | null | undefined): g is Georef {
  return !!g && (g.epsg === 4326 || !!g.proj4);
}

/** [south, west, north, east] of the grid's outer edges, degrees; null if a corner cannot be converted.
 *  `margin` grows the box by that many scene widths / heights on every side (the surroundings shown around it). */
export function sceneBBox(g: Georef, width: number, height: number, margin = 0): [number, number, number, number] | null {
  const mx = margin * width;
  const my = margin * height;
  const corners: Array<[number, number]> = [
    [-0.5 - mx, -0.5 - my],
    [width - 0.5 + mx, -0.5 - my],
    [-0.5 - mx, height - 0.5 + my],
    [width - 0.5 + mx, height - 0.5 + my],
  ];
  let s = Infinity;
  let w = Infinity;
  let n = -Infinity;
  let e = -Infinity;
  for (const [c, r] of corners) {
    const ll = lonLatAt(g, c, r);
    if (!ll) return null;
    w = Math.min(w, ll[0]);
    e = Math.max(e, ll[0]);
    s = Math.min(s, ll[1]);
    n = Math.max(n, ll[1]);
  }
  return [s, w, n, e];
}

/** Surroundings (basemap, facilities) reach this many scene sizes beyond each edge: a 5 × 5 window. */
export const CONTEXT_MARGIN = 2;

const bboxArea = ([s, w, n, e]: [number, number, number, number]) => (n - s) * (e - w);

/** The surroundings window: the widest margin up to CONTEXT_MARGIN whose box the public Overpass service accepts. */
export function contextBBox(g: Georef, width: number, height: number): { bbox: [number, number, number, number]; margin: number } | null {
  for (const margin of [CONTEXT_MARGIN, 1, 0.5, 0]) {
    const bbox = sceneBBox(g, width, height, margin);
    if (!bbox) return null;
    if (bboxArea(bbox) <= MAX_BBOX_DEG2) return { bbox, margin };
  }
  return null;
}

export function overpassQuery([s, w, n, e]: [number, number, number, number]): string {
  const b = `${s.toFixed(7)},${w.toFixed(7)},${n.toFixed(7)},${e.toFixed(7)}`;
  return [
    '[out:json][timeout:25];',
    '(',
    `  way["building"](${b});`,
    `  way["highway"](${b});`,
    `  way["railway"~"^(rail|light_rail|subway|tram|narrow_gauge)$"](${b});`,
    `  way["natural"="water"](${b});`,
    `  way["waterway"](${b});`,
    `  way["landuse"="reservoir"](${b});`,
    ');',
    'out geom tags;',
  ].join('\n');
}

const MINOR_HIGHWAYS = new Set(['footway', 'path', 'track', 'cycleway', 'steps', 'pedestrian', 'bridleway', 'corridor']);

export function classifyOsm(tags: Record<string, string>): OsmKind | null {
  if (tags.building) return 'building';
  if (tags.highway) return MINOR_HIGHWAYS.has(tags.highway) ? 'path' : 'road';
  if (tags.railway) return 'rail';
  if (tags.natural === 'water' || tags.landuse === 'reservoir') return 'water';
  if (tags.waterway) return 'waterway';
  return null;
}

/** lon/lat (WGS84) → grid pixel-centre (col, row) through the scene's CRS and affine. */
export function lonLatToGrid(g: Georef): ((lon: number, lat: number) => [number, number] | null) | null {
  if (g.epsg === 4326) return (lon, lat) => mapToPixel(g.transform, lon, lat);
  if (!g.proj4) return null;
  const conv = proj4('EPSG:4326', g.proj4);
  return (lon, lat) => {
    try {
      const [x, y] = conv.forward([lon, lat]);
      return Number.isFinite(x) && Number.isFinite(y) ? mapToPixel(g.transform, x, y) : null;
    } catch {
      return null;
    }
  };
}

/** Overpass `out geom` JSON → features on the grid. Unknown kinds and broken geometry are dropped. */
export function parseOverpass(json: unknown, toGrid: (lon: number, lat: number) => [number, number] | null): OsmFeature[] {
  const elements = (json as { elements?: unknown[] } | null)?.elements;
  if (!Array.isArray(elements)) return [];
  const out: OsmFeature[] = [];
  for (const el of elements) {
    const e = el as { type?: string; id?: number; tags?: Record<string, string>; geometry?: Array<{ lat: number; lon: number } | null> };
    if (e.type !== 'way' || !e.tags || !Array.isArray(e.geometry)) continue;
    const kind = classifyOsm(e.tags);
    if (!kind) continue;
    const pts: Array<[number, number]> = [];
    for (const p of e.geometry) {
      if (!p || !Number.isFinite(p.lat) || !Number.isFinite(p.lon)) continue;
      const g = toGrid(p.lon, p.lat);
      if (g) pts.push(g);
    }
    if (pts.length < 2) continue;
    const [a, z] = [pts[0], pts[pts.length - 1]];
    out.push({ id: e.id ?? 0, kind, name: e.tags.name ?? null, tags: e.tags, pts, closed: pts.length > 3 && a[0] === z[0] && a[1] === z[1] });
  }
  return out;
}

/** Liang–Barsky: the part of segment p→q inside [x0, x1] × [y0, y1], or null. */
export function clipSegment(p: [number, number], q: [number, number], x0: number, y0: number, x1: number, y1: number): [[number, number], [number, number]] | null {
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  let t0 = 0;
  let t1 = 1;
  const edges: Array<[number, number]> = [
    [-dx, p[0] - x0],
    [dx, x1 - p[0]],
    [-dy, p[1] - y0],
    [dy, y1 - p[1]],
  ];
  for (const [pp, qq] of edges) {
    if (pp === 0) {
      if (qq < 0) return null;
    } else {
      const t = qq / pp;
      if (pp < 0) t0 = Math.max(t0, t);
      else t1 = Math.min(t1, t);
      if (t0 > t1) return null;
    }
  }
  return [
    [p[0] + t0 * dx, p[1] + t0 * dy],
    [p[0] + t1 * dx, p[1] + t1 * dy],
  ];
}

/** Features clipped to the grid and subdivided every `stepPx`, as flat segment pairs [c0, r0, c1, r1, …] per kind —
 *  short enough that a line draped on the terrain follows it instead of cutting through hills. */
export function osmSegments(features: OsmFeature[], width: number, height: number, stepPx: number): Map<OsmKind, number[]> {
  const out = new Map<OsmKind, number[]>();
  const step = Math.max(0.5, stepPx);
  for (const f of features) {
    let list = out.get(f.kind);
    if (!list) out.set(f.kind, (list = []));
    for (let i = 0; i + 1 < f.pts.length; i++) {
      const seg = clipSegment(f.pts[i], f.pts[i + 1], -0.5, -0.5, width - 0.5, height - 0.5);
      if (!seg) continue;
      const [[a0, a1], [b0, b1]] = seg;
      const n = Math.max(1, Math.ceil(Math.hypot(b0 - a0, b1 - a1) / step));
      for (let k = 0; k < n; k++) {
        const s = k / n;
        const t = (k + 1) / n;
        list.push(a0 + (b0 - a0) * s, a1 + (b1 - a1) * s, a0 + (b0 - a0) * t, a1 + (b1 - a1) * t);
      }
    }
  }
  return out;
}

/** Public Overpass instances are often overloaded: they hang for a minute or answer 504/429, and which one is
 *  fast changes from minute to minute. So: answers are cached in the browser, a mirror is asked in parallel
 *  when the current one is slow, and a failed round is retried once. */
const HEDGE_MS = 6_000; // no answer yet → also ask the next mirror
const ATTEMPT_TIMEOUT_MS = 60_000; // one mirror gets this long (the query itself allows the server 25 s)
const RETRY_DELAY_MS = 2_500;
const CACHE_NAME = 'dw-osm-v1';
const CACHE_TTL_MS = 7 * 24 * 3600 * 1000;
const SAVED_AT = 'x-dw-saved-at';

const cacheKey = (query: string) => `https://osm-cache.depthwizard.invalid/?q=${encodeURIComponent(query)}`;

async function readCache(query: string): Promise<unknown | undefined> {
  try {
    if (typeof caches === 'undefined') return undefined;
    const hit = await (await caches.open(CACHE_NAME)).match(cacheKey(query));
    if (!hit || Date.now() - Number(hit.headers.get(SAVED_AT)) > CACHE_TTL_MS) return undefined;
    return await hit.json();
  } catch {
    return undefined;
  }
}

async function writeCache(query: string, json: unknown) {
  try {
    if (typeof caches === 'undefined') return;
    const res = new Response(JSON.stringify(json), { headers: { 'content-type': 'application/json', [SAVED_AT]: String(Date.now()) } });
    await (await caches.open(CACHE_NAME)).put(cacheKey(query), res);
  } catch {
    // quota or private mode: the overlay still works, it just refetches next time
  }
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => (clearTimeout(t), reject(signal.reason)), { once: true });
  });

/** One round over the mirrors: start the first, add the next after HEDGE_MS or as soon as one fails; first answer wins. */
function fetchFromMirrors(query: string, signal?: AbortSignal): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const attempts: AbortController[] = [];
    const errors: string[] = [];
    let next = 0;
    let pending = 0;
    let done = false;
    let hedge: ReturnType<typeof setTimeout> | undefined;
    const finish = (settle: () => void) => {
      if (done) return;
      done = true;
      clearTimeout(hedge);
      signal?.removeEventListener('abort', onAbort);
      attempts.forEach((a) => a.abort());
      settle();
    };
    const onAbort = () => finish(() => reject(signal!.reason));
    const launch = () => {
      clearTimeout(hedge);
      if (done || next >= OSM_ENDPOINTS.length) return;
      const { label, url } = OSM_ENDPOINTS[next++];
      const ctl = new AbortController();
      attempts.push(ctl);
      pending++;
      const timeout = setTimeout(() => ctl.abort(), ATTEMPT_TIMEOUT_MS);
      fetch(url, { method: 'POST', body: new URLSearchParams({ data: query }), signal: ctl.signal })
        .then(async (r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const json = (await r.json()) as { remark?: unknown };
          // an overloaded server can answer 200 with partial or no data and a "runtime error" remark
          if (typeof json?.remark === 'string' && /runtime error/i.test(json.remark)) throw new Error('query timed out on the server');
          return json;
        })
        .then(
          (json) => finish(() => resolve(json)),
          (e) => {
            pending--;
            errors.push(`${label}: ${ctl.signal.aborted ? 'no answer' : e instanceof Error ? e.message : String(e)}`);
            if (next < OSM_ENDPOINTS.length) launch();
            else if (pending === 0) finish(() => reject(new Error(`OpenStreetMap (Overpass) is busy — ${errors.join('; ')}`)));
          },
        )
        .finally(() => clearTimeout(timeout));
      if (next < OSM_ENDPOINTS.length) hedge = setTimeout(launch, HEDGE_MS);
    };
    if (signal?.aborted) return reject(signal.reason);
    signal?.addEventListener('abort', onAbort, { once: true });
    launch();
  });
}

/** Overpass answer for the query: from the browser cache, else the fastest mirror (one retry round). */
export async function fetchOverpass(query: string, signal?: AbortSignal): Promise<unknown> {
  const cached = await readCache(query);
  if (cached !== undefined) return cached;
  let json: unknown;
  try {
    json = await fetchFromMirrors(query, signal);
  } catch (e) {
    if (signal?.aborted) throw e;
    await sleep(RETRY_DELAY_MS, signal);
    json = await fetchFromMirrors(query, signal);
  }
  void writeCache(query, json);
  return json;
}
