/** XYZ (slippy-map, Web-Mercator) tiles for the basemap shown around a georeferenced scene.
 *  Display only: the tiles are never sent to the model. Each provider's attribution must be shown while its
 *  tiles are on screen (BasemapAttribution). */

export type BasemapId = 'satellite' | 'street';

export interface BasemapProvider {
  label: string;
  url: (z: number, x: number, y: number) => string;
  maxZoom: number;
  attribution: string;
  attributionUrl: string;
}

/** Both servers answer with CORS headers, so their tiles can be used as WebGL textures. */
export const BASEMAPS: Record<BasemapId, BasemapProvider> = {
  satellite: {
    label: 'Satellite',
    url: (z, x, y) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`,
    maxZoom: 19,
    attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
    attributionUrl: 'https://www.esri.com/en-us/legal/terms/full-master-agreement',
  },
  street: {
    label: 'Street map',
    url: (z, x, y) => `https://tile.openstreetmap.org/${z}/${x}/${y}.png`,
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors',
    attributionUrl: 'https://www.openstreetmap.org/copyright',
  },
};

export type BBox = [south: number, west: number, north: number, east: number];

export interface TileId {
  z: number;
  x: number;
  y: number;
  /** 'inner' tiles are sharper and cover the near surroundings; 'outer' ones fill the rest of the window. */
  ring: 'inner' | 'outer';
}

const MAX_LAT = 85.0511287798;
/** Metres per tile pixel at zoom 0 on the equator (256-px tiles). */
const M_PER_PX_Z0 = 156_543.033_928;

/** Fractional tile coordinates of a lon/lat. */
export function lonLatToTile(lon: number, lat: number, z: number): [number, number] {
  const n = 2 ** z;
  const phi = (Math.max(-MAX_LAT, Math.min(MAX_LAT, lat)) * Math.PI) / 180;
  const x = ((lon + 180) / 360) * n;
  const y = ((1 - Math.log(Math.tan(phi) + 1 / Math.cos(phi)) / Math.PI) / 2) * n;
  return [x, y];
}

/** lon/lat of fractional tile coordinates (x, y at zoom z); integer values are tile corners. */
export function tileToLonLat(x: number, y: number, z: number): [number, number] {
  const n = 2 ** z;
  const lon = (x / n) * 360 - 180;
  const lat = (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) * 180) / Math.PI;
  return [lon, lat];
}

/** Inclusive tile index range covering a box. */
export function tileRange([s, w, n, e]: BBox, z: number): { x0: number; x1: number; y0: number; y1: number } {
  const last = 2 ** z - 1;
  const [ax, ay] = lonLatToTile(w, n, z);
  const [bx, by] = lonLatToTile(e, s, z);
  const clampI = (v: number) => Math.max(0, Math.min(last, Math.floor(v)));
  return { x0: clampI(ax), x1: clampI(bx), y0: clampI(ay), y1: clampI(by) };
}

/** Zoom whose tile pixels are about twice the scene's ground sampling distance: sharp enough next to it
 *  without downloading more than the surroundings are worth. */
export function tileZoomFor(gsd: number, lat: number, maxZoom: number): number {
  const mpp = Math.max(gsd, 0.05) * 2;
  const z = Math.round(Math.log2((M_PER_PX_Z0 * Math.cos((lat * Math.PI) / 180)) / mpp));
  return Math.max(1, Math.min(maxZoom, z));
}

/** Most tiles one scene loads (textures are 256² each). */
export const MAX_TILES = 100;
/** The outer ring is this many zoom levels coarser than the inner one. */
const OUTER_STEP = 2;

/** Tiles for the surroundings: `inner` box at the zoom matching the scene, the rest of the `outer` box two
 *  levels coarser (skipping coarse tiles the inner ring already covers). Zooms drop until it fits MAX_TILES. */
export function planTiles(inner: BBox, outer: BBox, zoom: number, maxTiles = MAX_TILES): TileId[] {
  for (let zi = zoom; zi >= 1; zi--) {
    const zo = Math.max(0, zi - OUTER_STEP);
    const k = 2 ** (zi - zo);
    const ri = tileRange(inner, zi);
    const ro = tileRange(outer, zo);
    const tiles: TileId[] = [];
    for (let y = ri.y0; y <= ri.y1; y++) for (let x = ri.x0; x <= ri.x1; x++) tiles.push({ z: zi, x, y, ring: 'inner' });
    for (let y = ro.y0; y <= ro.y1; y++) {
      for (let x = ro.x0; x <= ro.x1; x++) {
        const covered = x * k >= ri.x0 && (x + 1) * k - 1 <= ri.x1 && y * k >= ri.y0 && (y + 1) * k - 1 <= ri.y1;
        if (!covered) tiles.push({ z: zo, x, y, ring: 'outer' });
      }
    }
    if (tiles.length <= maxTiles) return tiles;
  }
  return [];
}
