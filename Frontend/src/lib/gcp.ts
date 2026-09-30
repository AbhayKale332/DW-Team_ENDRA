/** Ground control points (GCPs) for scenes without their own georeferencing.
 *
 *  Elevation GCPs make the heights absolute (rDSM/nDSM -> DSM): at each point the ground lies at
 *  `elev - nDSM(pixel)`; 1-2 points give a level ground, 3+ a least-squares plane. DSM = ground + nDSM, so the DSM
 *  passes through the given elevations. Lat/lon GCPs (3+) fit an affine pixel -> UTM georeference, which unlocks the
 *  map, compass, OpenStreetMap layers and DEM anchoring. */
import proj4 from 'proj4';
import type { Affine, AnchoringInfo, Georef, GroundControlPoint, HeightGrid, Scene } from '@/domain/types';
import { gsdFromTransform, proj4ForEpsg } from './georef';
import { computeStats } from './heights';

/** Least squares z = a*u + b*v + c. Null when the points are (nearly) on one line. */
function fitPlane(u: number[], v: number[], z: number[]): [number, number, number] | null {
  const n = z.length;
  if (n < 3) return null;
  const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / n;
  const [mu, mv, mz] = [mean(u), mean(v), mean(z)];
  let suu = 0, svv = 0, suv = 0, suz = 0, svz = 0;
  for (let i = 0; i < n; i++) {
    const du = u[i] - mu, dv = v[i] - mv, dz = z[i] - mz;
    suu += du * du;
    svv += dv * dv;
    suv += du * dv;
    suz += du * dz;
    svz += dv * dz;
  }
  const det = suu * svv - suv * suv;
  if (!(det > 1e-6 * suu * svv)) return null;
  const a = (suz * svv - svz * suv) / det;
  const b = (svz * suu - suz * suv) / det;
  return [a, b, mz - a * mu - b * mv];
}

const rms = (r: number[]) => Math.sqrt(r.reduce((s, x) => s + x * x, 0) / Math.max(1, r.length));

export const hasLatLon = (p: GroundControlPoint) => Number.isFinite(p.lat) && Number.isFinite(p.lon);
export const hasElev = (p: GroundControlPoint) => Number.isFinite(p.elev);

export interface GeorefFit {
  georef: Georef;
  /** Horizontal residual, metres. */
  rmsM: number;
  /** Ground resolution implied by the points, m/px. */
  gsd: number;
}

/** Affine pixel -> UTM (zone of the points' centre) from 3+ lat/lon GCPs. */
export function fitGeoref(points: GroundControlPoint[]): GeorefFit | null {
  const pts = points.filter(hasLatLon);
  if (pts.length < 3) return null;
  const lon0 = pts.reduce((s, p) => s + p.lon!, 0) / pts.length;
  const lat0 = pts.reduce((s, p) => s + p.lat!, 0) / pts.length;
  const epsg = (lat0 >= 0 ? 32600 : 32700) + Math.min(60, Math.floor((lon0 + 180) / 6) + 1);
  const def = proj4ForEpsg(epsg)!;
  const xy = pts.map((p) => proj4('EPSG:4326', def, [p.lon!, p.lat!]));
  // transforms address pixel corners: a pixel centre (col, row) sits at (col + 0.5, row + 0.5)
  const u = pts.map((p) => p.col + 0.5);
  const v = pts.map((p) => p.row + 0.5);
  const fx = fitPlane(u, v, xy.map((q) => q[0]));
  const fy = fitPlane(u, v, xy.map((q) => q[1]));
  if (!fx || !fy) throw new Error('The lat/lon points lie on one line; spread them across the scene.');
  const t: Affine = [fx[0], fx[1], fx[2], fy[0], fy[1], fy[2]];
  const res = pts.map((_, i) => Math.hypot(t[0] * u[i] + t[1] * v[i] + t[2] - xy[i][0], t[3] * u[i] + t[4] * v[i] + t[5] - xy[i][1]));
  return { georef: { epsg, proj4: def, transform: t }, rmsM: rms(res), gsd: gsdFromTransform(t, epsg) };
}

export interface GroundFit {
  /** Ground elevation z = a*col + b*row + c (pixel centres). */
  plane: [number, number, number];
  kind: 'level' | 'plane';
  /** Residual of the DSM at the points, metres. */
  rmsM: number;
}

/** Ground surface from elevation GCPs: level for 1-2 points (or points on a line), a plane for 3+. */
export function fitGround(points: GroundControlPoint[], ndsm: HeightGrid): GroundFit | null {
  const pts = points.filter(hasElev);
  if (!pts.length) return null;
  const at = (p: GroundControlPoint) => {
    const c = Math.min(ndsm.width - 1, Math.max(0, Math.round(p.col)));
    const r = Math.min(ndsm.height - 1, Math.max(0, Math.round(p.row)));
    const h = ndsm.data[r * ndsm.width + c];
    return Number.isFinite(h) ? h : 0;
  };
  const z = pts.map((p) => p.elev! - at(p));
  const fit = fitPlane(pts.map((p) => p.col), pts.map((p) => p.row), z);
  const plane: [number, number, number] = fit ?? [0, 0, z.reduce((s, x) => s + x, 0) / z.length];
  const res = pts.map((p, i) => plane[0] * p.col + plane[1] * p.row + plane[2] - z[i]);
  return { plane, kind: fit ? 'plane' : 'level', rmsM: rms(res) };
}

export interface GcpResult {
  scene: Scene;
  geo: GeorefFit | null;
  ground: GroundFit | null;
}

/** Rebuild a scene from its above-ground heights with the GCPs applied (same id, so analysis state survives). */
export function applyGcpsToScene(scene: Scene, points: GroundControlPoint[]): GcpResult {
  const nd = scene.ndsm ?? scene.heights;
  const geo = fitGeoref(points);
  const ground = fitGround(points, nd);
  const ndStats = computeStats(nd.data);
  // a georeference fixes the horizontal scale: the mesh and objects follow it so they line up with the map
  const gsd = geo?.gsd ?? scene.gsd;
  let next: Scene = {
    ...scene,
    heights: nd,
    ndsm: undefined,
    terrain: undefined,
    anchoring: undefined,
    georef: geo?.georef ?? null,
    product: geo ? 'nDSM' : 'rDSM',
    stats: ndStats,
    gsd,
    gsdSource: geo ? 'user' : scene.gsdSource,
    objects: scene.objects ? { ...scene.objects, gsd } : scene.objects,
    warnings: geo ? scene.warnings.filter((w) => w.id !== 'gsd-assumed') : scene.warnings,
    gcps: points,
  };
  if (!ground) return { scene: next, geo, ground };

  const { width: W, height: H } = nd;
  const [a, b, c] = ground.plane;
  const dtm = new Float32Array(W * H);
  const dsm = new Float32Array(W * H);
  let lo = Infinity;
  let hi = -Infinity;
  for (let r = 0; r < H; r++) {
    for (let q = 0; q < W; q++) {
      const i = r * W + q;
      const h = nd.data[i];
      if (!Number.isFinite(h)) {
        dtm[i] = dsm[i] = NaN;
        continue;
      }
      const g = a * q + b * r + c;
      dtm[i] = g;
      dsm[i] = g + h;
      if (g < lo) lo = g;
      if (g > hi) hi = g;
    }
  }
  const n = points.filter(hasElev).length;
  const anchoring: AnchoringInfo = {
    source: `Ground control points (${n})`,
    sourceId: 'gcp',
    datum: 'unknown',
    cellM: gsd,
    cellPx: 1,
    demMinM: lo,
    demMaxM: hi,
    cellMeanRmseM: ground.rmsM,
    detailGain: 1,
    structureShare: 0,
    meanOffsetM: 0,
    fetchedAt: new Date().toISOString(),
    notes: [],
  };
  next = {
    ...next,
    product: 'DSM',
    heights: { ...nd, data: dsm },
    ndsm: nd,
    terrain: { ...nd, data: dtm },
    anchoring,
    stats: { ...computeStats(dsm), fracBelow1m: ndStats.fracBelow1m },
  };
  return { scene: next, geo, ground };
}
