/** DEM sources behind one interface: a function from lon/lat to elevation (NaN = no data).
 *
 *  - Terrain Tiles: AWS "Terrarium" PNG tiles (SRTM-based multi-source mosaic; CORS-enabled, no key).
 *  - Local file: any DEM GeoTIFF the user supplies (SRTM, Copernicus GLO-30, CartoDEM, LiDAR DTM ...).
 *
 *  No geoid conversion is applied anywhere: a DEM's heights are reported in the datum it was published in. */
import { get as idbGet, set as idbSet } from 'idb-keyval';
import proj4 from 'proj4';
import type { VerticalDatum } from '@/domain/types';
import { decodeTiffBand } from '../geotiff';
import { mapToPixel } from '../georef';
import { decodePng8 } from '../png';
import { lonLatToTile, tileRange, type BBox } from '../tiles';

export interface DemSampler {
  sourceId: 'terrain-tiles' | 'local-file';
  label: string;
  datum: VerticalDatum;
  /** Native resolution in metres, for the report. */
  resolutionM: number;
  tileZoom?: number;
  notes: string[];
  sample: (lon: number, lat: number) => number;
}

export class DemError extends Error {
  constructor(
    message: string,
    readonly kind: 'network' | 'coverage' | 'format' | 'unsupported' | 'cancelled' = 'network',
  ) {
    super(message);
  }
}

// ---------------------------------------------------------------------------------------------
// Terrain Tiles (Terrarium encoding)

export const TERRARIUM_URL = (z: number, x: number, y: number) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
/** z = 12 is ~30-38 m per pixel at Indian latitudes, i.e. SRTM 1 arc-second; finer zooms only interpolate. */
export const TERRAIN_ZOOM = 12;
export const MAX_DEM_TILES = 16;
const TILE = 256;

/** Terrarium: elevation = (R * 256 + G + B / 256) - 32768 metres. */
export const decodeTerrarium = (r: number, g: number, b: number) => r * 256 + g + b / 256 - 32768;

export function decodeTerrariumPixels(px: Uint8ClampedArray | Uint8Array, channels = 4): Float32Array {
  const out = new Float32Array(px.length / channels);
  for (let i = 0; i < out.length; i++) out[i] = decodeTerrarium(px[i * channels], px[i * channels + 1], px[i * channels + 2]);
  return out;
}

async function fetchTile(z: number, x: number, y: number, signal?: AbortSignal): Promise<Float32Array> {
  // v2: tiles are decoded from the PNG bytes; v1 went through a canvas, whose readback some browsers perturb
  const key = `dw.dem.terrarium.v2.${z}.${x}.${y}`;
  try {
    const hit = (await idbGet(key)) as Float32Array | undefined;
    if (hit && hit.length === TILE * TILE) return hit;
  } catch {
    /* IndexedDB unavailable (private mode): fetch every time */
  }
  let res: Response;
  try {
    res = await fetch(TERRARIUM_URL(z, x, y), { signal });
  } catch {
    if (signal?.aborted) throw new DemError('Cancelled', 'cancelled');
    throw new DemError('Could not reach the elevation tile server. Check the connection, or load a DEM file instead.', 'network');
  }
  if (!res.ok) throw new DemError(`Elevation tile ${z}/${x}/${y} unavailable (HTTP ${res.status}).`, 'coverage');
  // Decoded from the bytes, never through a canvas: anti-fingerprinting browsers (Brave, Firefox strict / private)
  // flip low bits of canvas readback, and one flip of the red channel is a 256 m pit in the terrain.
  let png: ReturnType<typeof decodePng8>;
  try {
    png = decodePng8(await res.arrayBuffer(), [2, 6]);
  } catch (e) {
    if (signal?.aborted) throw new DemError('Cancelled', 'cancelled');
    throw new DemError(`Elevation tile ${z}/${x}/${y} could not be decoded (${(e as Error).message}).`, 'format');
  }
  if (png.width !== TILE || png.height !== TILE) throw new DemError(`Elevation tile ${z}/${x}/${y} is ${png.width} x ${png.height}, not ${TILE} x ${TILE}.`, 'format');
  const h = decodeTerrariumPixels(png.data, png.channels);
  try {
    await idbSet(key, h);
  } catch {
    /* cache is best effort */
  }
  return h;
}

/** Bilinear lookup in a mosaic of Terrarium tiles covering tile columns x0..x1 and rows y0..y1 at zoom z. */
export function mosaicSampler(z: number, x0: number, y0: number, cols: number, rows: number, data: Float32Array): (lon: number, lat: number) => number {
  const W = cols * TILE;
  const H = rows * TILE;
  return (lon, lat) => {
    const [tx, ty] = lonLatToTile(lon, lat, z);
    const px = (tx - x0) * TILE - 0.5;
    const py = (ty - y0) * TILE - 0.5;
    if (px < -0.5 || py < -0.5 || px > W - 0.5 || py > H - 0.5) return NaN;
    const cx = Math.min(Math.max(px, 0), W - 1);
    const cy = Math.min(Math.max(py, 0), H - 1);
    const ix = Math.min(Math.floor(cx), W - 2);
    const iy = Math.min(Math.floor(cy), H - 2);
    const fx = cx - ix;
    const fy = cy - iy;
    const a = data[iy * W + ix];
    const b = data[iy * W + ix + 1];
    const c = data[(iy + 1) * W + ix];
    const d = data[(iy + 1) * W + ix + 1];
    return (1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * c + fx * d);
  };
}

/** Load the Terrain Tiles covering `bbox` ([south, west, north, east], degrees). */
export async function loadTerrainTiles(bbox: BBox, signal?: AbortSignal, z = TERRAIN_ZOOM): Promise<DemSampler> {
  const pad = 0.0005;
  const box: BBox = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad];
  const r = tileRange(box, z);
  const cols = r.x1 - r.x0 + 1;
  const rows = r.y1 - r.y0 + 1;
  if (cols * rows > MAX_DEM_TILES) throw new DemError(`The scene needs ${cols * rows} elevation tiles (limit ${MAX_DEM_TILES}); it is too large to anchor automatically.`, 'unsupported');
  const mosaic = new Float32Array(cols * TILE * rows * TILE);
  const W = cols * TILE;
  await Promise.all(
    Array.from({ length: cols * rows }, async (_, i) => {
      const tx = i % cols;
      const ty = Math.floor(i / cols);
      const tile = await fetchTile(z, r.x0 + tx, r.y0 + ty, signal);
      for (let y = 0; y < TILE; y++) mosaic.set(tile.subarray(y * TILE, (y + 1) * TILE), (ty * TILE + y) * W + tx * TILE);
    }),
  );
  return {
    sourceId: 'terrain-tiles',
    label: 'AWS Terrain Tiles (SRTM-based mosaic)',
    datum: 'EGM96',
    resolutionM: 30,
    tileZoom: z,
    notes: ['Multi-source mosaic (SRTM, NED, ETOPO ...), Web-Mercator resampled; not native SRTM GL1.', 'Orthometric heights (EGM96-like); no geoid conversion applied.'],
    sample: mosaicSampler(z, r.x0, r.y0, cols, rows, mosaic),
  };
}

// ---------------------------------------------------------------------------------------------
// Local DEM GeoTIFF

/** Largest DEM raster read whole (values); bigger files should be clipped to the area first. */
export const MAX_LOCAL_DEM_PIXELS = 80_000_000;

export async function loadLocalDem(file: File, datum: VerticalDatum): Promise<DemSampler> {
  let band: Awaited<ReturnType<typeof decodeTiffBand>>;
  try {
    band = await decodeTiffBand(await file.arrayBuffer());
  } catch {
    throw new DemError(`${file.name} could not be read as a GeoTIFF.`, 'format');
  }
  if (band.width * band.height > MAX_LOCAL_DEM_PIXELS) throw new DemError('That DEM raster is too large to load in the browser. Clip it to the scene area first.', 'unsupported');
  const g = band.georef;
  if (!g) throw new DemError('That DEM has no georeferencing, so it cannot be aligned with the scene.', 'format');
  const conv = g.epsg === 4326 ? null : g.proj4 ? proj4('EPSG:4326', g.proj4) : null;
  if (g.epsg !== 4326 && !conv) throw new DemError(`The DEM's coordinate system (EPSG:${g.epsg ?? '?'}) is not supported. Reproject it to WGS84 or UTM.`, 'unsupported');
  const { data, width: W, height: H } = band;
  // clamp to finite values: DEMs often carry -32768 / -9999 voids that GDAL nodata may not flag
  const bad = (v: number) => !Number.isFinite(v) || v < -1000;
  const res = Math.hypot(g.transform[0], g.transform[3]);
  const resolutionM = g.epsg === 4326 ? res * 111_320 : res;
  return {
    sourceId: 'local-file',
    label: `Local DEM · ${file.name}`,
    datum,
    resolutionM,
    notes: [datum === 'unknown' ? 'Vertical datum not declared; heights are used as published.' : `Vertical datum declared as ${datum}; no geoid conversion applied.`],
    sample: (lon, lat) => {
      const [x, y] = conv ? conv.forward([lon, lat]) : [lon, lat];
      const [c, r] = mapToPixel(g.transform, x, y);
      if (c < -0.5 || r < -0.5 || c > W - 0.5 || r > H - 0.5) return NaN;
      const cx = Math.min(Math.max(c, 0), W - 1);
      const cy = Math.min(Math.max(r, 0), H - 1);
      const ix = Math.min(Math.floor(cx), Math.max(W - 2, 0));
      const iy = Math.min(Math.floor(cy), Math.max(H - 2, 0));
      const fx = cx - ix;
      const fy = cy - iy;
      const at = (i: number, j: number) => data[Math.min(j, H - 1) * W + Math.min(i, W - 1)];
      const a = at(ix, iy);
      const b = at(ix + 1, iy);
      const cc = at(ix, iy + 1);
      const d = at(ix + 1, iy + 1);
      if (bad(a) || bad(b) || bad(cc) || bad(d)) return NaN;
      return (1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * cc + fx * d);
    },
  };
}
