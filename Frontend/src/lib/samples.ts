import type { ReferenceSurface, Scene, SceneMeta } from '@/domain/types';
import { parseNpy } from './npy';
import { buildScene } from './sceneBuilder';
import { classMapFromPng, fetchOptional } from './classMap';
import { fetchObjects } from './objects';
import type { DemCellsCache } from './dem';

export interface SampleDef {
  id: string;
  name: string;
  description: string;
  base: string;
  hasReference: boolean;
}

/** Real results from the hosted model, made with scripts/sample-from-space.mjs.
 *  The first one also backs the offline demo provider (src/api/mock/MockProvider.ts). */
export const SAMPLES: SampleDef[] = [
  {
    id: 'buildings-large-campus',
    name: 'Large campus (GeoTIFF, 0.6 m)',
    description: 'A georeferenced 0.6 m scene run through the DepthWizard model — buildings, trees, water, compass and OpenStreetMap.',
    base: './samples/buildings_large_campus',
    hasReference: false,
  },
];

async function fetchOk(url: string) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (HTTP ${r.status})`);
  return r;
}

export async function loadSample(def: SampleDef): Promise<{ scene: Scene; reference: ReferenceSurface | null; demCache: DemCellsCache | null }> {
  const [npy, meta, rgb, seg] = await Promise.all([
    fetchOk(`${def.base}/ndsm_m.npy`).then((r) => r.arrayBuffer()),
    fetchOk(`${def.base}/meta.json`).then((r) => r.json() as Promise<SceneMeta>),
    fetchOk(`${def.base}/rgb.png`).then((r) => r.blob()),
    fetchOptional(`${def.base}/seg.png`),
  ]);
  const arr = parseNpy(npy);
  const grid = { width: arr.shape[1], height: arr.shape[0], gsd: meta.scene?.gsd_m };
  const scene = await buildScene({
    name: def.name,
    image: rgb,
    heights: { data: arr.data, width: grid.width, height: grid.height },
    classes: classMapFromPng(seg, meta, grid.width, grid.height),
    objects: await fetchObjects(`${def.base}/objects.json`, grid),
    meta,
    provenance: { provider: 'bundled sample', source: 'sample', createdAt: new Date().toISOString() },
    statusLines: meta.note ? [meta.note] : [],
  });
  let reference: ReferenceSurface | null = null;
  if (def.hasReference) {
    const gt = parseNpy(await fetchOk(`${def.base}/gt_ndsm_m.npy`).then((r) => r.arrayBuffer()));
    reference = {
      name: 'gt_ndsm_m.npy (exact synthetic heights)',
      kind: 'nDSM',
      data: gt.data,
      alignment: 'same-extent',
      notes: ['Same pixel grid as the prediction.'],
    };
  }
  // DEM cells cached at build time (scripts/cache-dem.mjs) so the sample anchors without a network
  const demCache = (await fetchOk(`${def.base}/dem_cells.json`).then((r) => r.json() as Promise<DemCellsCache>).catch(() => null)) ?? null;
  return { scene, reference, demCache };
}
