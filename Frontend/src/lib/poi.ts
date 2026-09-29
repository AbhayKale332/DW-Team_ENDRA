/** Facilities from OpenStreetMap (emergency services, education, shelters & civic, transport hubs) for a
 *  georeferenced scene and its surroundings, via the same Overpass client as the OSM overlay (lib/osm.ts).
 *
 *  Data © OpenStreetMap contributors, ODbL — PoiLegend shows the attribution while the layer is on.
 *  Positions are grid pixel-centre coordinates (col, row) and may lie outside the grid. */

export type PoiCategory = 'emergency' | 'education' | 'civic' | 'transport';

export type PoiKind =
  | 'hospital'
  | 'clinic'
  | 'fire_station'
  | 'police'
  | 'ambulance_station'
  | 'school'
  | 'university'
  | 'kindergarten'
  | 'shelter'
  | 'townhall'
  | 'community_centre'
  | 'pharmacy'
  | 'rail_station'
  | 'bus_station'
  | 'helipad'
  | 'aerodrome';

export interface Poi {
  id: string;
  kind: PoiKind;
  name: string | null;
  tags: Record<string, string>;
  col: number;
  row: number;
}

export const POI_CATEGORIES: PoiCategory[] = ['emergency', 'education', 'civic', 'transport'];

export const POI_CATEGORY_LABELS: Record<PoiCategory, string> = {
  emergency: 'Emergency services',
  education: 'Education',
  civic: 'Shelters & civic',
  transport: 'Transport hubs',
};

export const POI_CATEGORY_COLORS: Record<PoiCategory, string> = {
  emergency: '#e03131',
  education: '#7048e8',
  civic: '#2f9e44',
  transport: '#1971c2',
};

/** Declaration order is also the drawing priority: emergency facilities are kept first when the layer is capped. */
export const POI_KIND_META: Record<PoiKind, { label: string; category: PoiCategory; color: string }> = {
  hospital: { label: 'Hospital', category: 'emergency', color: '#e03131' },
  ambulance_station: { label: 'Ambulance station', category: 'emergency', color: '#c92a2a' },
  fire_station: { label: 'Fire station', category: 'emergency', color: '#f76707' },
  police: { label: 'Police', category: 'emergency', color: '#1c7ed6' },
  clinic: { label: 'Clinic', category: 'emergency', color: '#f06595' },
  school: { label: 'School', category: 'education', color: '#7048e8' },
  university: { label: 'College / university', category: 'education', color: '#5f3dc4' },
  kindergarten: { label: 'Kindergarten', category: 'education', color: '#9775fa' },
  shelter: { label: 'Shelter / assembly point', category: 'civic', color: '#2f9e44' },
  townhall: { label: 'Town hall', category: 'civic', color: '#495057' },
  community_centre: { label: 'Community centre', category: 'civic', color: '#0c8599' },
  pharmacy: { label: 'Pharmacy', category: 'civic', color: '#37b24d' },
  rail_station: { label: 'Railway station', category: 'transport', color: '#1971c2' },
  bus_station: { label: 'Bus station', category: 'transport', color: '#1098ad' },
  helipad: { label: 'Helipad', category: 'transport', color: '#e8590c' },
  aerodrome: { label: 'Airport / airstrip', category: 'transport', color: '#364fc7' },
};

export const POI_KINDS = Object.keys(POI_KIND_META) as PoiKind[];

export function poiQuery([s, w, n, e]: [number, number, number, number]): string {
  const b = `${s.toFixed(7)},${w.toFixed(7)},${n.toFixed(7)},${e.toFixed(7)}`;
  return [
    '[out:json][timeout:25];',
    '(',
    `  nwr["amenity"~"^(hospital|clinic|doctors|fire_station|police|school|college|university|kindergarten|shelter|townhall|community_centre|pharmacy|bus_station)$"](${b});`,
    `  nwr["emergency"~"^(ambulance_station|assembly_point)$"](${b});`,
    `  nwr["healthcare"~"^(hospital|clinic)$"](${b});`,
    `  nwr["social_facility"="shelter"](${b});`,
    `  nwr["railway"~"^(station|halt)$"](${b});`,
    `  nwr["aeroway"~"^(aerodrome|heliport|helipad)$"](${b});`,
    ');',
    'out center tags;',
  ].join('\n');
}

/** Emergency first: a hospital's helipad or a school used as a shelter keeps its most critical role. */
export function classifyPoi(t: Record<string, string>): PoiKind | null {
  const a = t.amenity;
  if (a === 'hospital' || t.healthcare === 'hospital') return 'hospital';
  if (t.emergency === 'ambulance_station') return 'ambulance_station';
  if (a === 'fire_station') return 'fire_station';
  if (a === 'police') return 'police';
  if (a === 'clinic' || a === 'doctors' || t.healthcare === 'clinic') return 'clinic';
  if (a === 'school') return 'school';
  if (a === 'college' || a === 'university') return 'university';
  if (a === 'kindergarten') return 'kindergarten';
  if (a === 'shelter' || t.emergency === 'assembly_point' || t.social_facility === 'shelter') return 'shelter';
  if (a === 'townhall') return 'townhall';
  if (a === 'community_centre') return 'community_centre';
  if (a === 'pharmacy') return 'pharmacy';
  if (t.railway === 'station' || t.railway === 'halt') return 'rail_station';
  if (a === 'bus_station') return 'bus_station';
  if (t.aeroway === 'helipad' || t.aeroway === 'heliport') return 'helipad';
  if (t.aeroway === 'aerodrome') return 'aerodrome';
  return null;
}

/** Same-kind facilities closer than this are one place mapped twice (e.g. a station node and its area). */
const DEDUPE_M = 30;

/** Overpass `out center` JSON → facilities on the grid. `gsd` (m / px) is used to merge duplicates. */
export function parsePois(json: unknown, toGrid: (lon: number, lat: number) => [number, number] | null, gsd: number): Poi[] {
  const elements = (json as { elements?: unknown[] } | null)?.elements;
  if (!Array.isArray(elements)) return [];
  const out: Poi[] = [];
  const r2 = (DEDUPE_M / Math.max(gsd, 1e-6)) ** 2;
  for (const el of elements) {
    const e = el as { type?: string; id?: number; lat?: number; lon?: number; center?: { lat?: number; lon?: number }; tags?: Record<string, string> };
    if (!e.tags) continue;
    const kind = classifyPoi(e.tags);
    if (!kind) continue;
    const lat = e.type === 'node' ? e.lat : e.center?.lat;
    const lon = e.type === 'node' ? e.lon : e.center?.lon;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const g = toGrid(lon!, lat!);
    if (!g || !Number.isFinite(g[0]) || !Number.isFinite(g[1])) continue;
    const dup = out.find((p) => p.kind === kind && (p.col - g[0]) ** 2 + (p.row - g[1]) ** 2 < r2);
    if (dup) {
      // keep the better described copy
      if (!dup.name && e.tags.name) Object.assign(dup, { name: e.tags.name, tags: e.tags });
      continue;
    }
    out.push({ id: `${e.type ?? 'x'}/${e.id ?? out.length}`, kind, name: e.tags.name ?? null, tags: e.tags, col: g[0], row: g[1] });
  }
  return out;
}

/** Facilities of the enabled categories, most critical kinds first, at most `max`. */
export function visiblePois(pois: Poi[], categories: Record<PoiCategory, boolean>, max: number): Poi[] {
  const rank = new Map(POI_KINDS.map((k, i) => [k, i]));
  return pois
    .filter((p) => categories[POI_KIND_META[p.kind].category])
    .sort((a, b) => rank.get(a.kind)! - rank.get(b.kind)!)
    .slice(0, max);
}
