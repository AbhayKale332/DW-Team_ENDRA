/** Anchor a georeferenced scene to a DEM: align, fuse, and rebuild the scene as an absolute DSM. */
import type { AnchoringInfo, Georef, Scene, VerticalDatum } from '@/domain/types';
import { computeStats } from '../heights';
import { lonLatAt } from '../georef';
import { sceneBBox, canGeolocate } from '../osm';
import { anchorCellPx, anchorToDem, cellCount, type CellGrid } from './anchor';
import { DemError, loadLocalDem, loadTerrainTiles, type DemSampler } from './sources';

export { DemError } from './sources';
export type { DemSampler } from './sources';

/** DEM native resolution the anchor cells are matched to. */
export const ANCHOR_CELL_M = 30;
const MIN_COVERAGE = 0.8;
/** The DEM is treated as bare terrain unless told otherwise: it gives smooth ground and buildings that stand on it. */
export const DEFAULT_STRUCTURE_SHARE = 0;

/** Pre-sampled DEM cells shipped with a sample so it can be anchored offline. */
export interface DemCellsCache {
  version: 1;
  rows: number;
  cols: number;
  cellPx: number;
  cells: Array<number | null>;
  source: string;
  datum: VerticalDatum;
  tileZoom?: number;
  fetchedAt: string;
  notes: string[];
}

export type AnchorRequest =
  | { kind: 'terrain-tiles' }
  | { kind: 'local-file'; file: File; datum: VerticalDatum }
  | { kind: 'cells'; cache: DemCellsCache };

/** Why a scene cannot be anchored, or null when it can. Never invents a location. */
export function anchorBlocker(scene: Scene): string | null {
  if (!scene.georef) return 'This image has no coordinate system, so it cannot be placed on the Earth. It stays a relative surface model (rDSM).';
  if (!canGeolocate(scene.georef)) return `The image's coordinate system (EPSG:${(scene.georef as Georef).epsg ?? 'unknown'}) is not supported for DEM lookup. Reproject it to WGS84 or UTM.`;
  if (scene.product === 'DSM' && !scene.ndsm) return 'This scene is already an absolute DSM.';
  return null;
}

/** Sample the DEM at the centre (3 x 3 sub-points) of each anchor cell of the scene grid. */
export function sampleDemCells(georef: Georef, W: number, H: number, k: number, dem: (lon: number, lat: number) => number): CellGrid {
  const cols = cellCount(W, k);
  const rows = cellCount(H, k);
  const data = new Float64Array(rows * cols);
  const fr = [1 / 6, 0.5, 5 / 6];
  for (let r = 0; r < rows; r++) {
    const y0 = r * k;
    const y1 = Math.min(H, y0 + k);
    for (let c = 0; c < cols; c++) {
      const x0 = c * k;
      const x1 = Math.min(W, x0 + k);
      let s = 0;
      let n = 0;
      for (const fy of fr) {
        for (const fx of fr) {
          const ll = lonLatAt(georef, x0 + fx * (x1 - x0) - 0.5, y0 + fy * (y1 - y0) - 0.5);
          if (!ll) continue;
          const v = dem(ll[0], ll[1]);
          if (Number.isFinite(v)) {
            s += v;
            n++;
          }
        }
      }
      data[r * cols + c] = n >= 3 ? s / n : NaN;
    }
  }
  return { data, rows, cols };
}

export function cellsToCache(g: CellGrid, k: number, sampler: Pick<DemSampler, 'label' | 'datum' | 'tileZoom' | 'notes'>): DemCellsCache {
  return {
    version: 1,
    rows: g.rows,
    cols: g.cols,
    cellPx: k,
    cells: Array.from(g.data, (v) => (Number.isFinite(v) ? Math.round(v * 100) / 100 : null)),
    source: sampler.label,
    datum: sampler.datum,
    tileZoom: sampler.tileZoom,
    fetchedAt: new Date().toISOString(),
    notes: sampler.notes,
  };
}

export interface AnchorProgress {
  (message: string): void;
}

/** Fetch/sample the DEM for `scene` and return the same scene rebuilt as an absolute DSM.
 *  `scene.id` is kept, so analysis state tied to the scene survives. */
export async function anchorScene(scene: Scene, req: AnchorRequest, opts: { signal?: AbortSignal; onProgress?: AnchorProgress; structureShare?: number } = {}): Promise<Scene> {
  const blocker = anchorBlocker(scene);
  if (blocker) throw new DemError(blocker, 'unsupported');
  const georef = scene.georef as Georef;
  const nd = scene.ndsm ?? scene.heights;
  const { width: W, height: H } = nd;
  const k = anchorCellPx(scene.gsd, ANCHOR_CELL_M);
  const cellM = k * scene.gsd;

  let cells: CellGrid;
  let source: string;
  let sourceId: AnchoringInfo['sourceId'];
  let datum: VerticalDatum;
  let tileZoom: number | undefined;
  let fetchedAt = new Date().toISOString();
  let notes: string[];
  if (req.kind === 'cells') {
    const c = req.cache;
    if (c.cellPx !== k || c.rows !== cellCount(H, k) || c.cols !== cellCount(W, k)) throw new DemError('The bundled DEM cache does not match this scene grid.', 'coverage');
    cells = { rows: c.rows, cols: c.cols, data: Float64Array.from(c.cells, (v) => (v === null ? NaN : v)) };
    source = `${c.source} (bundled cache)`;
    sourceId = 'bundled-cache';
    datum = c.datum;
    tileZoom = c.tileZoom;
    fetchedAt = c.fetchedAt;
    notes = c.notes;
  } else {
    let sampler: DemSampler;
    if (req.kind === 'terrain-tiles') {
      const bbox = sceneBBox(georef, W, H, 0);
      if (!bbox) throw new DemError('The scene corners could not be converted to latitude/longitude.', 'unsupported');
      opts.onProgress?.('Fetching elevation tiles');
      sampler = await loadTerrainTiles(bbox, opts.signal);
    } else {
      opts.onProgress?.('Reading DEM file');
      sampler = await loadLocalDem(req.file, req.datum);
    }
    opts.onProgress?.('Aligning DEM with the scene');
    cells = sampleDemCells(georef, W, H, k, sampler.sample);
    source = sampler.label;
    sourceId = sampler.sourceId;
    datum = sampler.datum;
    tileZoom = sampler.tileZoom;
    notes = sampler.notes;
  }

  const finite = cells.data.reduce((n, v) => n + (Number.isFinite(v) ? 1 : 0), 0);
  if (finite / cells.data.length < MIN_COVERAGE) throw new DemError(`The DEM covers only ${Math.round((100 * finite) / cells.data.length)} % of the scene.`, 'coverage');

  opts.onProgress?.('Fusing DEM and predicted heights');
  const gain = 1;
  const share = opts.structureShare ?? DEFAULT_STRUCTURE_SHARE;
  const r = anchorToDem(nd.data, W, H, k, cells, { detailGain: gain, structureShare: share });
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of cells.data) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const ndStats = computeStats(nd.data);
  const stats = { ...computeStats(r.dsm), fracBelow1m: ndStats.fracBelow1m };
  const anchoring: AnchoringInfo = {
    source,
    sourceId,
    datum,
    cellM,
    cellPx: k,
    demMinM: lo,
    demMaxM: hi,
    cellMeanRmseM: r.cellMeanRmseM ?? 0,
    detailGain: gain,
    structureShare: share,
    meanOffsetM: r.meanOffsetM ?? 0,
    demCells: { rows: cells.rows, cols: cells.cols, data: cells.data },
    fetchedAt,
    tileZoom,
    notes,
  };
  return {
    ...scene,
    product: 'DSM',
    heights: { data: r.dsm, width: W, height: H },
    ndsm: nd,
    terrain: { data: r.dtm, width: W, height: H },
    anchoring,
    stats,
    warnings: scene.warnings.filter((w) => w.id !== 'anchor-failed'),
  };
}

/** Swap `heights` between the absolute DSM and the above-ground nDSM of an anchored scene. With detail gain 1
 *  (the only value the UI uses) the DSM is exactly `terrain + nDSM`, so nothing extra is kept in memory. */
export function withHeightReference(scene: Scene, ref: 'dsm' | 'ndsm'): Scene {
  if (!scene.anchoring || !scene.ndsm || !scene.terrain) return scene;
  const wantDsm = ref === 'dsm';
  if ((scene.product === 'DSM') === wantDsm) return scene;
  if (!wantDsm) return { ...scene, product: scene.georef ? 'nDSM' : 'rDSM', heights: scene.ndsm, stats: computeStats(scene.ndsm.data) };
  const t = scene.terrain.data;
  const n = scene.ndsm.data;
  const dsm = new Float32Array(t.length);
  for (let i = 0; i < dsm.length; i++) dsm[i] = t[i] + n[i];
  return { ...scene, product: 'DSM', heights: { ...scene.ndsm, data: dsm }, stats: { ...computeStats(dsm), fracBelow1m: computeStats(n).fracBelow1m } };
}

/** Re-fuse an anchored scene with another structure share, from the DEM cells it kept (no network). */
export function refuseScene(scene: Scene, structureShare: number): Scene {
  const a = scene.anchoring;
  const nd = scene.ndsm;
  if (!a?.demCells || !nd) return scene;
  const cells: CellGrid = { rows: a.demCells.rows, cols: a.demCells.cols, data: a.demCells.data };
  const r = anchorToDem(nd.data, nd.width, nd.height, a.cellPx, cells, { detailGain: a.detailGain, structureShare });
  const ndStats = computeStats(nd.data);
  const dsm = { ...nd, data: r.dsm };
  const anchoring: AnchoringInfo = { ...a, structureShare, cellMeanRmseM: r.cellMeanRmseM ?? 0, meanOffsetM: r.meanOffsetM ?? 0 };
  const base = { ...scene, anchoring, terrain: { ...nd, data: r.dtm } };
  if (scene.product !== 'DSM') return base; // showing the nDSM view: only the terrain layer changes
  return { ...base, heights: dsm, stats: { ...computeStats(r.dsm), fracBelow1m: ndStats.fracBelow1m } };
}
