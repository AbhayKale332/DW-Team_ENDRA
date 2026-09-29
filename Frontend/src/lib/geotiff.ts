import { fromArrayBuffer } from 'geotiff';
import type { Affine, Georef } from '@/domain/types';
import { gsdFromTransform, proj4ForEpsg } from './georef';

export interface DecodedTiff {
  width: number;
  height: number;
  bands: number;
  bitsPerSample: number;
  /** 8-bit RGBA, possibly downsampled to `maxSide`. */
  rgba: Uint8ClampedArray;
  rgbaWidth: number;
  rgbaHeight: number;
  georef: Georef | null;
  gsd: number | null;
  nodata: number | null;
}

type Band = ArrayLike<number>;

/** Per-band 2–98 % stretch to 8-bit — mirrors the backend's radiometric stretch for non-8-bit data. */
function stretchBand(src: Band, n: number, is8bit: boolean): Uint8ClampedArray {
  const out = new Uint8ClampedArray(n);
  if (is8bit) {
    for (let i = 0; i < n; i++) out[i] = src[i];
    return out;
  }
  const stride = Math.max(1, Math.floor(n / 100_000));
  const sample: number[] = [];
  for (let i = 0; i < n; i += stride) if (Number.isFinite(src[i])) sample.push(src[i]);
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.02)] ?? 0;
  const hi = sample[Math.floor(sample.length * 0.98)] ?? 1;
  const span = hi - lo || 1;
  for (let i = 0; i < n; i++) out[i] = ((src[i] - lo) / span) * 255;
  return out;
}

function readGeoref(image: Awaited<ReturnType<Awaited<ReturnType<typeof fromArrayBuffer>>['getImage']>>): Georef | null {
  const keys = image.getGeoKeys() ?? {};
  const fd = image.getFileDirectory() as unknown as { getValue?: (k: string) => unknown } & Record<string, unknown>;
  const get = (k: string) => (typeof fd.getValue === 'function' ? fd.getValue(k) : fd[k]);
  let transform: Affine | null = null;
  const mt = get('ModelTransformation') as number[] | undefined;
  if (mt && mt.length >= 16) {
    transform = [mt[0], mt[1], mt[3], mt[4], mt[5], mt[7]];
  } else {
    try {
      const [ox, oy] = image.getOrigin();
      const [rx, ry] = image.getResolution();
      transform = [rx, 0, ox, 0, ry, oy];
    } catch {
      transform = null;
    }
  }
  if (!transform) return null;
  const k = keys as Record<string, unknown>;
  const projected = Number(k.ProjectedCSTypeGeoKey) || null;
  const geographic = Number(k.GeographicTypeGeoKey) || null;
  const modelType = Number(k.GTModelTypeGeoKey) || null;
  let epsg: number | null = projected && projected !== 32767 ? projected : null;
  if (!epsg && modelType === 2) epsg = geographic && geographic !== 32767 ? geographic : 4326;
  if (!epsg && geographic && geographic !== 32767 && modelType !== 1) epsg = geographic;
  return { epsg, proj4: proj4ForEpsg(epsg), transform };
}

export async function decodeTiff(buffer: ArrayBuffer, maxSide = 8192): Promise<DecodedTiff> {
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const bands = image.getSamplesPerPixel();
  const bitsPerSample = image.getBitsPerSample(0);
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const rw = Math.max(1, Math.round(width * scale));
  const rh = Math.max(1, Math.round(height * scale));
  const samples = bands >= 3 ? [0, 1, 2] : [0];
  const rasters = (await image.readRasters({ samples, width: rw, height: rh, resampleMethod: 'bilinear' })) as unknown as Band[];
  const n = rw * rh;
  const is8 = bitsPerSample === 8;
  const chans = rasters.map((b) => stretchBand(b, n, is8));
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const r = chans[0][i];
    rgba[i * 4] = r;
    rgba[i * 4 + 1] = chans.length > 1 ? chans[1][i] : r;
    rgba[i * 4 + 2] = chans.length > 2 ? chans[2][i] : r;
    rgba[i * 4 + 3] = 255;
  }
  const georef = readGeoref(image);
  let gsd: number | null = null;
  if (georef) {
    let lat: number | undefined;
    if (georef.epsg === 4326) lat = georef.transform[5] + georef.transform[4] * (height / 2);
    gsd = gsdFromTransform(georef.transform, georef.epsg, lat);
    if (!(gsd > 0 && gsd < 1000)) gsd = null;
  }
  return { width, height, bands, bitsPerSample, rgba, rgbaWidth: rw, rgbaHeight: rh, georef, gsd, nodata: image.getGDALNoData() };
}

/** Read a single-band raster (reference DSM) as float32. */
export async function decodeTiffBand(buffer: ArrayBuffer): Promise<{ data: Float32Array; width: number; height: number; georef: Georef | null; nodata: number | null }> {
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const r = (await image.readRasters({ samples: [0] })) as unknown as Band[];
  const nodata = image.getGDALNoData();
  const data = new Float32Array(width * height);
  const src = r[0];
  for (let i = 0; i < data.length; i++) {
    const v = src[i];
    data[i] = nodata !== null && v === nodata ? NaN : v;
  }
  return { data, width, height, georef: readGeoref(image), nodata };
}

export function isTiffName(name: string) {
  return /\.tiff?$/i.test(name);
}
