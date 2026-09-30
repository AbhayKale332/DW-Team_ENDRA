// Fetch the OpenStreetMap data for a bundled sample once, so its OSM overlay and facilities layer open
// instantly and offline instead of waiting on the public Overpass service.
//
//   node scripts/cache-osm.mjs [public/samples/buildings_large_campus]
//
// Mirrors src/lib/osm.ts (sceneBBox, contextBBox, overpassQuery) and src/lib/poi.ts (poiQuery). Writes osm.json
// and pois.json (raw Overpass answers; the app parses them as it would a live answer).
// Data © OpenStreetMap contributors, ODbL.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import proj4 from 'proj4';

const dir = process.argv[2] ?? 'public/samples/buildings_large_campus';
const ENDPOINT = 'https://overpass-api.de/api/interpreter';
const MAX_BBOX_DEG2 = 0.1;
const CONTEXT_MARGIN = 2;

const meta = JSON.parse(await readFile(join(dir, 'meta.json'), 'utf8'));
const { transform, crs_epsg: epsg, width: W, height: H } = meta.scene;
if (!transform || !epsg) throw new Error('meta.json has no georeferencing');
const src = epsg === 4326 ? 'EPSG:4326' : epsg >= 32601 && epsg <= 32660 ? `+proj=utm +zone=${epsg - 32600} +datum=WGS84 +units=m +no_defs` : epsg >= 32701 && epsg <= 32760 ? `+proj=utm +zone=${epsg - 32700} +south +datum=WGS84 +units=m +no_defs` : null;
if (!src) throw new Error(`EPSG:${epsg} unsupported by this script`);
const [a, b, c, d, e, f] = transform;
const toLonLat = (col, row) => {
  // col,row are pixel-centre indices; pixelToMap adds 0.5
  const x = a * (col + 0.5) + b * (row + 0.5) + c;
  const y = d * (col + 0.5) + e * (row + 0.5) + f;
  return proj4(src, 'EPSG:4326', [x, y]);
};

/** [south, west, north, east], grown by `margin` scene sizes on every side. */
function sceneBBox(margin) {
  const mx = margin * W;
  const my = margin * H;
  const lls = [[-0.5 - mx, -0.5 - my], [W - 0.5 + mx, -0.5 - my], [-0.5 - mx, H - 0.5 + my], [W - 0.5 + mx, H - 0.5 + my]].map(([cc, r]) => toLonLat(cc, r));
  const lons = lls.map((p) => p[0]);
  const lats = lls.map((p) => p[1]);
  return [Math.min(...lats), Math.min(...lons), Math.max(...lats), Math.max(...lons)];
}

function contextBBox() {
  for (const m of [CONTEXT_MARGIN, 1, 0.5, 0]) {
    const [s, w, n, e2] = sceneBBox(m);
    if ((n - s) * (e2 - w) <= MAX_BBOX_DEG2) return [s, w, n, e2];
  }
  throw new Error('scene too large for Overpass');
}

const fmt = ([s, w, n, e2]) => `${s.toFixed(7)},${w.toFixed(7)},${n.toFixed(7)},${e2.toFixed(7)}`;

const osmQuery = (b) =>
  [
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

const poiQuery = (b) =>
  [
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

async function overpass(query) {
  for (let attempt = 1; ; attempt++) {
    const r = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'content-type': 'application/x-www-form-urlencoded; charset=UTF-8', 'user-agent': 'DepthWizard/1.0 (sample cache)' },
      body: new URLSearchParams({ data: query }),
      signal: AbortSignal.timeout(60_000),
    }).catch((err) => ({ ok: false, status: String(err) }));
    if (r.ok) {
      const json = await r.json();
      if (!(typeof json.remark === 'string' && /runtime error/i.test(json.remark))) return json;
    }
    if (attempt >= 8) throw new Error(`Overpass failed: ${r.status}`);
    console.log(`  busy (${r.status}), retrying…`);
    await new Promise((res) => setTimeout(res, 3000 * attempt));
  }
}

for (const [file, query] of [
  ['osm.json', osmQuery(fmt(sceneBBox(0)))],
  ['pois.json', poiQuery(fmt(contextBBox()))],
]) {
  const json = await overpass(query);
  // keep the answer, drop the per-request timestamps so a re-run only changes the file when the data did
  const { osm3s, ...rest } = json;
  await writeFile(join(dir, file), JSON.stringify({ ...rest, osm3s: { copyright: osm3s?.copyright } }));
  console.log(`${file}: ${json.elements?.length ?? 0} elements`);
}
