import { create } from 'zustand';

/** A sample scene: a .dwproj in a folder under public/samples, listed by samples/index.json (server/samples.mjs). */
export interface SampleDef {
  id: string;
  name: string;
  /** Path of the .dwproj relative to samples/. */
  file: string;
}

export const sampleUrl = (s: SampleDef) => `./samples/${s.file.split('/').map(encodeURIComponent).join('/')}`;

export const useSamples = create<{ samples: SampleDef[] }>()(() => ({ samples: [] }));

<<<<<<< HEAD
let pending: Promise<SampleDef[]> | null = null;

/** Fetch the sample list (again): folders added since the last call show up. */
export function loadSamples(): Promise<SampleDef[]> {
  pending ??= fetch('./samples/index.json', { cache: 'no-store' })
    .then(async (r) => (r.ok ? ((await r.json()) as { samples?: unknown }) : null))
    .then((j) => (Array.isArray(j?.samples) ? (j.samples as SampleDef[]).filter((s) => s && typeof s.id === 'string' && typeof s.file === 'string') : []))
    .catch(() => useSamples.getState().samples)
    .then((samples) => {
      useSamples.setState({ samples });
      return samples;
    })
    .finally(() => {
      pending = null;
    });
  return pending;
=======
export interface SampleResult {
  scene: Scene;
  reference: ReferenceSurface | null;
  demCache: DemCellsCache | null;
  /** Overpass answers cached at build time (scripts/cache-osm.mjs): the OSM overlay and the facilities layer. */
  osm: unknown | null;
  pois: unknown | null;
}

const optionalJson = (url: string) => fetchOk(url).then((r) => r.json() as Promise<unknown>).catch(() => null);

export async function loadSample(def: SampleDef): Promise<SampleResult> {
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
  const [demCache, osm, pois] = await Promise.all([
    optionalJson(`${def.base}/dem_cells.json`) as Promise<DemCellsCache | null>,
    optionalJson(`${def.base}/osm.json`),
    optionalJson(`${def.base}/pois.json`),
  ]);
  return { scene, reference, demCache, osm, pois };
>>>>>>> 486e375 (Fix: OSM - Data loading speed optimized)
}
