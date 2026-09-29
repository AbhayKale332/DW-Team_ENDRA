import proj4 from 'proj4';
import type { Affine, Georef } from '@/domain/types';

/** proj4 definition for EPSG codes we can build without a network lookup. */
export function proj4ForEpsg(epsg: number | null | undefined): string | null {
  if (!epsg) return null;
  if (epsg === 4326) return '+proj=longlat +datum=WGS84 +no_defs';
  if (epsg === 3857) return '+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +no_defs';
  if (epsg >= 32601 && epsg <= 32660) return `+proj=utm +zone=${epsg - 32600} +datum=WGS84 +units=m +no_defs`;
  if (epsg >= 32701 && epsg <= 32760) return `+proj=utm +zone=${epsg - 32700} +south +datum=WGS84 +units=m +no_defs`;
  // Other CRSs (e.g. Everest-based Indian zones) need a definition from the GeoTIFF's own keys.
  return null;
}

/** Scale a transform for a grid resampled from (srcW, srcH) to (dstW, dstH) over the same extent. */
export function rescaleTransform(t: Affine, srcW: number, srcH: number, dstW: number, dstH: number): Affine {
  const sx = srcW / dstW;
  const sy = srcH / dstH;
  return [t[0] * sx, t[1] * sy, t[2], t[3] * sx, t[4] * sy, t[5]];
}

/** Map coordinates of a pixel centre. */
export function pixelToMap(t: Affine, col: number, row: number): [number, number] {
  const c = col + 0.5;
  const r = row + 0.5;
  return [t[0] * c + t[1] * r + t[2], t[3] * c + t[4] * r + t[5]];
}

/** Inverse of pixelToMap for north-up or rotated transforms. Returns fractional (col, row) of pixel centres. */
export function mapToPixel(t: Affine, x: number, y: number): [number, number] {
  const det = t[0] * t[4] - t[1] * t[3];
  const dx = x - t[2];
  const dy = y - t[5];
  const c = (t[4] * dx - t[1] * dy) / det;
  const r = (-t[3] * dx + t[0] * dy) / det;
  return [c - 0.5, r - 0.5];
}

export function lonLatAt(g: Georef, col: number, row: number): [number, number] | null {
  const [x, y] = pixelToMap(g.transform, col, row);
  if (g.epsg === 4326) return [x, y];
  if (!g.proj4) return null;
  try {
    const [lon, lat] = proj4(g.proj4, 'EPSG:4326', [x, y]);
    return Number.isFinite(lon) && Number.isFinite(lat) ? [lon, lat] : null;
  } catch {
    return null;
  }
}

/** Metres per pixel from a transform. Geographic CRSs are converted at the scene latitude. */
export function gsdFromTransform(t: Affine, epsg: number | null, centreLat?: number): number {
  const px = Math.hypot(t[0], t[3]);
  const py = Math.hypot(t[1], t[4]);
  let g = (px + py) / 2;
  if (epsg === 4326) {
    const lat = ((centreLat ?? 0) * Math.PI) / 180;
    const mPerDegLat = 111_132.92 - 559.82 * Math.cos(2 * lat);
    const mPerDegLon = 111_412.84 * Math.cos(lat);
    g = (px * mPerDegLon + py * mPerDegLat) / 2;
  }
  return g;
}

export function formatLonLat(ll: [number, number] | null) {
  if (!ll) return '—';
  const [lon, lat] = ll;
  return `${Math.abs(lat).toFixed(6)}°${lat >= 0 ? 'N' : 'S'} ${Math.abs(lon).toFixed(6)}°${lon >= 0 ? 'E' : 'W'}`;
}

/** Transform another grid's pixel to this grid's pixel when both share (or can be reprojected to) a CRS. */
export function makePixelMapper(from: Georef, to: Georef): ((col: number, row: number) => [number, number]) | null {
  const same = from.epsg && to.epsg && from.epsg === to.epsg;
  if (same) {
    return (col, row) => {
      const [x, y] = pixelToMap(from.transform, col, row);
      return mapToPixel(to.transform, x, y);
    };
  }
  if (!from.proj4 || !to.proj4) return null;
  const conv = proj4(from.proj4, to.proj4);
  return (col, row) => {
    const [x, y] = pixelToMap(from.transform, col, row);
    const [x2, y2] = conv.forward([x, y]);
    return mapToPixel(to.transform, x2, y2);
  };
}
