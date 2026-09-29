import { writeArrayBuffer } from 'geotiff';
import type { Scene } from '@/domain/types';
import { lut, colorAt, type ColormapId } from '@/theme/colormaps';
import { hillshade } from '../heights';
import { quantise16 } from '../rg16';
import { encodeGray16Png } from '../png';
import { writeNpyF32 } from '../npy';

/** Float32 GeoTIFF of the height grid. Georeferenced scenes keep their CRS and (grid-rescaled) transform;
 *  others are written in a user-defined local projected CRS at the true GSD so GIS tools measure correctly. */
export type GeoTiffLayer = 'primary' | 'ndsm' | 'terrain';

const VERTICAL_CS: Record<string, number> = { EGM96: 5773, EGM2008: 3855 };

/** What a raster layer of this scene is, in words a GIS user can trust (also written into the file). */
export function layerDescription(scene: Scene, layer: GeoTiffLayer): { product: string; units: string; citation: string } {
  const a = scene.anchoring;
  if (layer === 'terrain' && a) return { product: 'DTM', units: 'metres', citation: `Bare-earth terrain elevation, DEM-anchored (${a.source}); datum ${a.datum}` };
  if (layer === 'ndsm' || scene.product !== 'DSM') {
    if (scene.product === 'rDSM' && layer !== 'ndsm') return { product: 'rDSM', units: 'metres (relative)', citation: 'DepthWizard local grid (not georeferenced); relative heights above local ground, metres' };
    return { product: 'nDSM', units: 'metres above ground', citation: 'Height above ground, metres; no vertical datum' };
  }
  return { product: 'DSM', units: `metres above ${a?.datum ?? 'DEM datum'}`, citation: `Absolute DSM, DEM-anchored (${a?.source ?? 'DEM'}); heights above ${a?.datum ?? 'the DEM datum'}` };
}

export function exportGeoTiff(scene: Scene, layer: GeoTiffLayer = 'primary'): Blob {
  const grid = layer === 'ndsm' ? (scene.ndsm ?? scene.heights) : layer === 'terrain' ? (scene.terrain ?? scene.heights) : scene.heights;
  const { data, width, height } = grid;
  const desc = layerDescription(scene, layer);
  const meta: Record<string, unknown> = { width, height, GTRasterTypeGeoKey: 1 };
  const g = scene.georef;
  if (g) {
    const [a, b, c, d, e, f] = g.transform;
    if (Math.abs(b) < 1e-12 && Math.abs(d) < 1e-12) {
      meta.ModelPixelScale = [a, -e, 0];
      meta.ModelTiepoint = [0, 0, 0, c, f, 0];
    } else {
      meta.ModelTransformation = [a, b, 0, c, d, e, 0, f, 0, 0, 0, 0, 0, 0, 0, 1];
    }
    if (g.epsg === 4326) {
      meta.GTModelTypeGeoKey = 2;
      meta.GeographicTypeGeoKey = 4326;
    } else {
      meta.GTModelTypeGeoKey = 1;
      meta.ProjectedCSTypeGeoKey = g.epsg ?? 32767;
    }
  } else {
    meta.GTModelTypeGeoKey = 1;
    meta.ProjectedCSTypeGeoKey = 32767;
    meta.GTCitationGeoKey = 'DepthWizard local grid (not georeferenced); relative heights, metres';
    meta.ModelPixelScale = [scene.gsd, scene.gsd, 0];
    meta.ModelTiepoint = [0, 0, 0, 0, 0, 0];
  }
  const values = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) values[i] = Number.isFinite(data[i]) ? data[i] : -9999;
  meta.GDAL_NODATA = '-9999';
  if (g && scene.anchoring && desc.product !== 'nDSM') {
    const vcs = VERTICAL_CS[scene.anchoring.datum];
    if (vcs) meta.VerticalCSTypeGeoKey = vcs;
  }
  if (g) meta.GTCitationGeoKey = desc.citation;
  const buf = writeArrayBuffer(values, meta as never);
  return new Blob([buf], { type: 'image/tiff' });
}

export function exportNpy(scene: Scene): Blob {
  return new Blob([writeNpyF32(scene.heights.data, [scene.heights.height, scene.heights.width]) as BlobPart], { type: 'application/octet-stream' });
}

export function export16BitPng(scene: Scene): Blob {
  const { min, max } = scene.stats;
  const q = quantise16(scene.heights.data, min, max);
  return encodeGray16Png(q, scene.heights.width, scene.heights.height, {
    Software: 'DepthWizard',
    Description: `height_m = ${min} + png16 / 65535 * (${max} - ${min})`,
    'DepthWizard:gsd_m': String(scene.gsd),
    'DepthWizard:product': scene.product,
  });
}

export interface HeatmapOptions {
  colormap: ColormapId;
  range: [number, number];
  hillshade: number;
  colorbar: boolean;
  sunAzimuth: number;
  sunElevation: number;
}

/** Heatmap at native grid resolution (+ optional colorbar panel), as PNG or JPEG. */
export async function exportHeatmap(scene: Scene, type: 'image/png' | 'image/jpeg', o: HeatmapOptions): Promise<Blob> {
  const { data, width: W, height: H } = scene.heights;
  const table = lut(o.colormap);
  const hs = o.hillshade > 0 ? hillshade(data, W, H, scene.gsd, o.sunAzimuth, o.sunElevation) : null;
  const panel = o.colorbar ? Math.max(90, Math.round(W * 0.09)) : 0;
  const canvas = new OffscreenCanvas(W + panel, H);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  const img = ctx.createImageData(W, H);
  const [lo, hi] = o.range;
  const span = hi - lo || 1;
  for (let i = 0; i < W * H; i++) {
    const t = Math.min(1, Math.max(0, (data[i] - lo) / span));
    const k = Math.round(t * 255) * 4;
    const shade = hs ? 1 - o.hillshade + o.hillshade * (0.35 + (0.9 * hs[i]) / 255) : 1;
    img.data[i * 4] = Math.min(255, table[k] * shade);
    img.data[i * 4 + 1] = Math.min(255, table[k + 1] * shade);
    img.data[i * 4 + 2] = Math.min(255, table[k + 2] * shade);
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  if (panel) {
    ctx.fillStyle = '#0d1013';
    ctx.fillRect(W, 0, panel, H);
    const barX = W + panel * 0.18;
    const barW = panel * 0.22;
    const top = H * 0.08;
    const bot = H * 0.92;
    for (let y = Math.floor(top); y <= bot; y++) {
      const [r, g, b] = colorAt(o.colormap, 1 - (y - top) / (bot - top));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(barX, y, barW, 1);
    }
    ctx.fillStyle = '#e8ecf1';
    ctx.font = `${Math.max(11, Math.round(panel * 0.13))}px Inter, sans-serif`;
    ctx.textBaseline = 'middle';
    const tx = barX + barW + panel * 0.06;
    ctx.fillText(`${hi.toFixed(1)} m`, tx, top);
    ctx.fillText(`${((lo + hi) / 2).toFixed(1)} m`, tx, (top + bot) / 2);
    ctx.fillText(`${lo.toFixed(1)} m`, tx, bot);
  }
  return canvas.convertToBlob({ type, quality: 0.92 });
}
