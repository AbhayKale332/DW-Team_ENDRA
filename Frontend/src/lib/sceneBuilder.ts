import type { Artefact, ClassMap, Georef, GridPoint, GsdSource, HeightGrid, Product, Provenance, Scene, SceneCloud, SceneMeta, SceneObjects, SceneWarning } from '@/domain/types';
import { computeStats } from './heights';
import { proj4ForEpsg, rescaleTransform } from './georef';
import { resampleMask } from './cloud';
import { withConfidentCut } from './uncertainty';

export interface BuildSceneArgs {
  name: string;
  image: Blob;
  heights: HeightGrid;
  /** Per-pixel σ of `heights` (`ndsm_std_m.npy`); dropped if the size does not match. */
  uncertainty?: HeightGrid | null;
  /** Object classes on the height grid; dropped if the size does not match. */
  classes?: ClassMap | null;
  /** 3D objects on the height grid; dropped if the grid does not match. */
  objects?: SceneObjects | null;
  /** Clouds masked out of the input; `heights` must already be filled under them. */
  cloud?: SceneCloud | null;
  meta: SceneMeta;
  gsd?: number | null;
  gsdSource?: GsdSource;
  /** Georeferencing of the *input image* at its original size. */
  inputGeoref?: Georef | null;
  inputSize?: { width: number; height: number };
  artefacts?: Artefact[];
  statusLines?: string[];
  warning?: string | null;
  resampleNote?: string;
  provenance: Provenance;
}

/** Objects whose position falls under the cloud are the model reading the fill colour: drop them. */
function objectsOutsideCloud(objects: SceneObjects, under: Uint8Array): SceneObjects {
  const { width, height } = objects;
  const at = (x: number, y: number) => under[Math.min(height - 1, Math.max(0, Math.floor(y))) * width + Math.min(width - 1, Math.max(0, Math.floor(x)))] >= 128;
  const centre = (poly: GridPoint[]) => {
    let x = 0;
    let y = 0;
    for (const [px, py] of poly) {
      x += px;
      y += py;
    }
    return poly.length ? at(x / poly.length, y / poly.length) : false;
  };
  return {
    ...objects,
    trees: objects.trees.filter((t) => !at(t.x, t.y)),
    buildings: objects.buildings.filter((b) => !centre(b.poly)),
    water: objects.water.filter((w) => !centre(w.poly)),
  };
}

async function imageSize(blob: Blob) {
  const bmp = await createImageBitmap(blob);
  const s = { width: bmp.width, height: bmp.height };
  bmp.close();
  return s;
}

export async function buildScene(a: BuildSceneArgs): Promise<Scene> {
  const { heights, meta } = a;
  const cloud = a.cloud ?? null;
  // the fill under the cloud is invented: keep it out of the statistics
  const under = cloud ? resampleMask(cloud.mask, cloud.width, cloud.height, heights.width, heights.height) : null;
  let statsData = heights.data;
  if (under) {
    statsData = heights.data.slice();
    for (let i = 0; i < under.length; i++) if (under[i] >= 128) statsData[i] = NaN;
  }
  const stats = computeStats(statsData);
  const size = a.inputSize ?? (await imageSize(a.image));

  // The Space omits the affine (it may resample); restore it from the input GeoTIFF, scaled to the grid.
  let georef: Georef | null = null;
  if (a.inputGeoref) {
    georef = {
      ...a.inputGeoref,
      transform: rescaleTransform(a.inputGeoref.transform, size.width, size.height, heights.width, heights.height),
    };
  } else if (meta.scene?.transform && meta.scene.transform.length === 6 && meta.scene.crs_epsg) {
    const t = meta.scene.transform as [number, number, number, number, number, number];
    // Backends following rasterio order: (a, b, c, d, e, f) with c/f the origin.
    georef = { epsg: meta.scene.crs_epsg, proj4: proj4ForEpsg(meta.scene.crs_epsg), transform: t };
  }

  const gsd = a.gsd ?? meta.scene?.gsd_m ?? 0.5;
  const gsdSource: GsdSource = a.gsdSource ?? meta.scene?.gsd_source ?? 'assumed';
  let product: Product = meta.product === 'DSM' ? 'DSM' : 'rDSM';
  if (product !== 'DSM' && georef) product = 'nDSM';

  const warnings: SceneWarning[] = [];
  if (a.warning || stats.fracBelow1m < 0.15) {
    warnings.push({
      id: 'flat-ground',
      level: 'warning',
      title: 'Check the declared GSD',
      message:
        a.warning ??
        `Only ${(stats.fracBelow1m * 100).toFixed(1)} % of the scene is below 1 m. The model may be reading texture as terrain — double-check the resolution (m/pixel).`,
    });
  }
  if (cloud)
    warnings.push({
      id: 'clouds-masked',
      level: 'info',
      title: 'Clouds masked',
      message: `Clouds covered ${(cloud.coverage * 100).toFixed(1)} % of the image. The model saw those areas filled with the surrounding colours, and their heights were filled from the surrounding ground — they are hatched in the view and are not measurements.`,
    });
  if (a.resampleNote) warnings.push({ id: 'resampled', level: 'info', title: 'Scene resampled', message: a.resampleNote });
  if (gsdSource === 'assumed')
    warnings.push({
      id: 'gsd-assumed',
      level: 'info',
      title: 'Resolution assumed',
      message: `No resolution was declared, so ${gsd.toFixed(2)} m/px was assumed. Heights scale with this value.`,
    });

  return {
    id: crypto.randomUUID(),
    name: a.name,
    image: a.image,
    imageWidth: size.width,
    imageHeight: size.height,
    heights,
    // no σ under the cloud: the heights there are filled, not measured
    uncertainty: a.uncertainty && a.uncertainty.width === heights.width && a.uncertainty.height === heights.height ? withConfidentCut(a.uncertainty, meta, under) : null,
    classes: a.classes && a.classes.width === heights.width && a.classes.height === heights.height ? a.classes : null,
    // Positions are grid pixels; metres come from the scene's effective GSD, so carry that one.
    objects:
      a.objects && a.objects.width === heights.width && a.objects.height === heights.height
        ? { ...(under ? objectsOutsideCloud(a.objects, under) : a.objects), gsd }
        : null,
    cloud,
    gsd,
    gsdSource,
    product,
    stats,
    georef,
    meta,
    artefacts: a.artefacts ?? [],
    warnings,
    statusLines: a.statusLines ?? [],
    provenance: a.provenance,
  };
}

export const PRODUCT_LABELS: Record<Product, { short: string; long: string }> = {
  rDSM: { short: 'rDSM', long: 'Relative surface model — heights above local ground at the stated resolution; no coordinate system, so no absolute elevation' },
  nDSM: { short: 'nDSM', long: 'Height above ground, georeferenced — not an elevation until it is anchored to a DEM' },
  DSM: { short: 'DSM', long: 'Absolute surface model — elevations above the vertical datum of the DEM it was anchored to' },
};
