import { strFromU8, strToU8, unzipSync, zipSync, type Zippable } from 'fflate';
import { notifications } from '@mantine/notifications';
import type { Artefact, SceneMeta } from '@/domain/types';
import { buf, json, mimeOf, readProject, type Manifest } from '@/lib/dwproj';
import type { DemCellsCache } from '@/lib/dem';
import { parseNpy, writeNpyF32 } from '@/lib/npy';
import { serializeObjects } from '@/lib/objects';
import type { OsmFeature } from '@/lib/osm';
import type { Poi } from '@/lib/poi';
import { encodeGray8Png } from '@/lib/png';
import { buildScene } from '@/lib/sceneBuilder';
import { download, safeStem } from '@/lib/download';
import { getRecentBlob, putRecent } from '@/lib/recent';
import { prepareInput, stemOf } from '@/lib/input';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { useCamera, viewportApi } from '@/store/camera';
import { useTool, type GridPoint } from '@/store/tool';
import { useUi } from '@/store/ui';
import { useAnchor } from '@/store/anchor';
import { pickUseCases, snapshotUseCases } from '@/store/usecases';
import { osmFeaturesFor, useOsm } from '@/features/osm/osmStore';
import { poisFor, usePoi } from '@/features/poi/poiStore';
import { restoreUseCases } from '@/features/usecases/actions';
import { reportError } from './openFile';

async function thumbnail(): Promise<string | null> {
  const shot = await viewportApi.screenshot();
  if (!shot) return null;
  const bmp = await createImageBitmap(shot);
  const w = 240;
  const h = Math.round((bmp.height / bmp.width) * w);
  const c = new OffscreenCanvas(w, h);
  c.getContext('2d')!.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const blob = await c.convertToBlob({ type: 'image/jpeg', quality: 0.8 });
  return new Promise((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result as string);
    r.readAsDataURL(blob);
  });
}

/** Model output files fetched once per URL (the Space's file links expire, so a project keeps the bytes). */
const outputCache = new Map<string, Blob>();

async function outputBlob(a: Artefact, fetchMissing: boolean): Promise<Blob | null> {
  if (a.blob) return a.blob;
  if (!a.url) return null;
  const hit = outputCache.get(a.url);
  if (hit || !fetchMissing) return hit ?? null;
  try {
    const r = await fetch(a.url);
    if (!r.ok) return null;
    const b = await r.blob();
    outputCache.set(a.url, b);
    return b;
  } catch {
    return null;
  }
}

/** Already-compressed formats are stored, not deflated again. */
const STORED = /\.(png|jpe?g|glb|zip|dwproj|tiff?)$/i;
const safeName = (n: string) => n.replace(/[\\/]+/g, '_');

/** Serialise the whole workspace into a .dwproj (zip) blob: everything needed to reopen it on another machine,
 *  offline. `fetchOutputs` downloads model output files not fetched yet (Recent skips that to stay cheap). */
export async function buildProjectBlob({ fetchOutputs = true }: { fetchOutputs?: boolean } = {}): Promise<{ blob: Blob; name: string } | null> {
  const st = useScene.getState();
  const { scene, reference, input } = st;
  if (!scene) return null;
  const { set: _s, reset: _r, ...view } = useView.getState();
  const tool = useTool.getState();
  const ext = scene.image.type === 'image/jpeg' ? 'jpg' : 'png';
  const files: Zippable = {};
  const put = (name: string, data: Uint8Array) => void (files[name] = STORED.test(name) ? [data, { level: 0 }] : data);
  const bytes = async (b: Blob) => new Uint8Array(await b.arrayBuffer());

  const imageName = `input.${ext}`;
  put(imageName, await bytes(scene.image));
  // always the above-ground heights: an anchored scene re-anchors from dem_cells.json when reopened
  put('ndsm_m.npy', writeNpyF32((scene.ndsm ?? scene.heights).data, [scene.heights.height, scene.heights.width]));
  if (reference) put('reference_m.npy', writeNpyF32(reference.data, [scene.heights.height, scene.heights.width]));
  if (scene.classes) put('seg.png', await bytes(encodeGray8Png(scene.classes.data, scene.classes.width, scene.classes.height)));
  // Same schema the Space publishes, so a project's objects.json is readable by anything that reads the Space's.
  if (scene.objects) put('objects.json', strToU8(JSON.stringify(serializeObjects(scene.objects))));

  const a = scene.anchoring;
  if (a?.demCells) {
    const cache: DemCellsCache = {
      version: 1,
      rows: a.demCells.rows,
      cols: a.demCells.cols,
      cellPx: a.cellPx,
      cells: Array.from(a.demCells.data, (v) => (Number.isFinite(v) ? v : null)),
      source: a.source.replace(/ \(bundled cache\)$/, ''),
      datum: a.datum,
      tileZoom: a.tileZoom,
      fetchedAt: a.fetchedAt,
      notes: a.notes,
    };
    put('dem_cells.json', strToU8(JSON.stringify(cache)));
  }
  const osm = osmFeaturesFor(useOsm.getState(), scene);
  if (osm) put('osm.json', strToU8(JSON.stringify(osm)));
  const pois = poisFor(usePoi.getState(), scene);
  if (pois) put('pois.json', strToU8(JSON.stringify(pois)));

  let source: Manifest['source'] = null;
  if (input) {
    source = { name: input.name, file: `source/${safeName(input.name)}` };
    put(source.file, await bytes(input.file));
  }
  const outputs: NonNullable<Manifest['outputs']> = [];
  for (const art of scene.artefacts) {
    // the model's height map is the project's own ndsm_m.npy: do not store it twice
    if (art.name === 'ndsm_m.npy') {
      outputs.push({ name: art.name, file: 'ndsm_m.npy' });
      continue;
    }
    const b = await outputBlob(art, fetchOutputs);
    if (!b) continue;
    const file = `outputs/${safeName(art.name)}`;
    put(file, await bytes(b));
    outputs.push({ name: art.name, file });
  }

  const manifest: Manifest = {
    format: 'dwproj',
    version: 2,
    name: scene.name,
    imageName,
    gsd: scene.gsd,
    gsdSource: scene.gsdSource,
    georef: scene.georef,
    meta: scene.meta,
    statusLines: scene.statusLines,
    provenance: scene.provenance,
    view,
    bookmarks: useCamera.getState().bookmarks,
    tools: { probe: tool.probe, measure: tool.measure, profile: tool.profile },
    reference: reference ? { name: reference.name, kind: reference.kind, alignment: reference.alignment, notes: reference.notes } : null,
    classes: scene.classes ? { names: Array.from(scene.classes.names, (n) => n ?? '') } : null,
    anchoring: a ? { source: a.source, sourceId: a.sourceId, structureShare: a.structureShare, heightRef: scene.product === 'DSM' ? 'dsm' : 'ndsm' } : null,
    source,
    outputs,
    params: st.params,
    usecases: snapshotUseCases(),
    removeOffset: st.removeOffset,
    dismissed: st.dismissed,
  };
  files['manifest.json'] = strToU8(JSON.stringify(manifest, null, 2));
  const zip = zipSync(files, { level: 6 });
  return { blob: new Blob([zip as BlobPart], { type: 'application/zip' }), name: `${safeStem(scene.name)}.dwproj` };
}

/** Save (and Export → Entire Project): the complete project as one .dwproj file. */
export async function saveProject() {
  if (!useScene.getState().scene) return;
  const id = notifications.show({ loading: true, title: 'Saving project', message: 'Packing all project files', autoClose: false, withCloseButton: false });
  try {
    const built = await buildProjectBlob();
    if (!built) return void notifications.hide(id);
    download(built.blob, built.name);
    await rememberRecent(built.blob);
    useScene.getState().set({ dirty: false });
    notifications.update({ id, loading: false, color: 'teal', title: 'Project saved', message: `${built.name} · ${(built.blob.size / 1e6).toFixed(1)} MB`, autoClose: 3500, withCloseButton: true });
  } catch (e) {
    notifications.hide(id);
    reportError(e, 'Could not save project');
  }
}

/** Keep the current workspace in Recent (IndexedDB). */
export async function rememberRecent(blob?: Blob) {
  const scene = useScene.getState().scene;
  if (!scene) return;
  const b = blob ?? (await buildProjectBlob({ fetchOutputs: false }))?.blob;
  if (!b) return;
  await putRecent({ id: scene.id, name: scene.name, savedAt: new Date().toISOString(), thumbnail: await thumbnail(), size: b.size }, b);
}

const isPoint = (p: unknown): p is GridPoint => !!p && typeof (p as GridPoint).col === 'number' && typeof (p as GridPoint).row === 'number';
const points = (v: unknown): GridPoint[] => (Array.isArray(v) ? v.filter(isPoint) : []);

/** Make a project the current workspace. `sample` marks a bundled sample scene. */
async function adoptProject(bytes: Uint8Array, opts: { sample?: boolean } = {}) {
  const { manifest, files, image, heights, classes, objects } = readProject(bytes);
  const outputs: Artefact[] = (manifest.outputs ?? [])
    .filter((o) => files[o.file])
    .map((o) => {
      const blob = new Blob([files[o.file] as BlobPart], { type: mimeOf(o.name) });
      return { name: o.name, blob, size: blob.size };
    });
  let scene = await buildScene({
    name: manifest.name,
    image,
    heights,
    classes,
    objects,
    meta: manifest.meta,
    gsd: manifest.gsd,
    gsdSource: manifest.gsdSource,
    inputGeoref: manifest.georef,
    inputSize: { width: heights.width, height: heights.height },
    artefacts: outputs,
    statusLines: manifest.statusLines,
    provenance: { ...manifest.provenance, source: opts.sample ? 'sample' : 'project' },
  });

  // Anchor from the DEM cells saved with the project: same result as when it was saved, and no network needed.
  const demCache = json<DemCellsCache>(files['dem_cells.json']);
  let anchored = false;
  if (demCache) {
    try {
      const dem = await import('@/lib/dem');
      const saved = manifest.anchoring;
      scene = await dem.anchorScene(scene, { kind: 'cells', cache: demCache }, { structureShare: saved?.structureShare });
      if (saved && scene.anchoring) scene = { ...scene, anchoring: { ...scene.anchoring, source: saved.source, sourceId: saved.sourceId } };
      if (saved?.heightRef === 'ndsm') scene = dem.withHeightReference(scene, 'ndsm');
      anchored = true;
    } catch (e) {
      console.warn('Project DEM cells unusable; anchoring again', e);
    }
  }

  useTool.getState().clear();
  useScene.getState().setScene(scene);
  const anchoring = await import('@/features/anchoring/runAnchoring');
  if (demCache) anchoring.registerBundledDem(scene.id, demCache);
  if (anchored) useAnchor.getState().set({ status: 'done', message: null, sceneId: scene.id });
  else anchoring.autoAnchor();

  if (manifest.reference && files['reference_m.npy']) {
    const ref = parseNpy(buf(files['reference_m.npy']));
    useScene.getState().setReference({ ...manifest.reference, data: ref.data });
  }
  const st = useScene.getState();
  st.set({ removeOffset: !!manifest.removeOffset });
  for (const id of manifest.dismissed ?? []) st.dismiss(id);
  if (manifest.params) st.setParams(manifest.params);

  // OpenStreetMap layers as they were fetched: shown without asking the Overpass API again
  const osm = json<OsmFeature[]>(files['osm.json']);
  if (Array.isArray(osm)) useOsm.setState({ sceneId: scene.id, status: 'ready', features: osm, error: null });
  const pois = json<Poi[]>(files['pois.json']);
  if (Array.isArray(pois)) usePoi.setState({ sceneId: scene.id, status: 'ready', pois, error: null });

  useView.getState().reset();
  useView.getState().set(manifest.view as never);
  useCamera.getState().set({ mode: 'orbit', bookmarks: manifest.bookmarks ?? [] });
  const probe = manifest.tools?.probe;
  useTool.getState().set({ probe: isPoint(probe) ? probe : null, measure: points(manifest.tools?.measure).slice(0, 2), profile: points(manifest.tools?.profile) });
  restoreUseCases(scene.id, pickUseCases(manifest.usecases));
  useUi.getState().set({ projectOpen: false });

  // The original image, staged again so the model can be re-run from this project.
  const src = manifest.source && files[manifest.source.file];
  if (manifest.source && src) {
    const file = new File([src as BlobPart], manifest.source.name, { type: mimeOf(manifest.source.name) });
    void prepareInput(file)
      .then((input) => useScene.getState().setInput(input))
      .catch((e) => console.warn('Project source image unreadable', e));
  } else {
    useScene.getState().setInput(null);
  }
}

export async function openProject(file: Blob & { name?: string }, opts: { sample?: boolean } = {}) {
  try {
    await adoptProject(new Uint8Array(await file.arrayBuffer()), opts);
    if (!opts.sample) notifications.show({ title: 'Project opened', message: file.name ?? 'Project', color: 'teal' });
    return true;
  } catch (e) {
    reportError(e, opts.sample ? 'Could not load sample' : 'Could not open project');
    return false;
  }
}

export async function openRecent(id: string) {
  const blob = await getRecentBlob(id);
  if (!blob) {
    notifications.show({ color: 'red', title: 'Recent project unavailable', message: 'It may have been cleared from browser storage.' });
    return;
  }
  await openProject(blob);
}

/** Open a result bundle (the Space's *_result.zip, or loose ndsm_m.npy + meta.json + rgb.png) without inference. */
export async function openResultBundle(input: File[]) {
  try {
    const files: Record<string, Uint8Array> = {};
    for (const f of input) {
      if (f.name.toLowerCase().endsWith('.zip')) {
        const z = unzipSync(new Uint8Array(await f.arrayBuffer()));
        for (const [k, v] of Object.entries(z)) files[k.split('/').pop()!.toLowerCase()] = v;
      } else files[f.name.toLowerCase()] = new Uint8Array(await f.arrayBuffer());
    }
    const npyKey = ['ndsm_m.npy', 'pred_ndsm_m.npy', 'dsm_m.npy'].find((k) => files[k]) ?? Object.keys(files).find((k) => k.endsWith('.npy') && !k.startsWith('gt_'));
    if (!npyKey) throw new Error('The bundle has no height map (ndsm_m.npy).');
    const arr = parseNpy(files[npyKey].slice().buffer);
    const [h, w] = arr.shape;
    const meta: SceneMeta = files['meta.json'] ? JSON.parse(strFromU8(files['meta.json'])) : {};
    const imgKey = ['rgb.png', 'rgb.jpg', 'texture.png'].find((k) => files[k]);
    let image: Blob;
    if (imgKey) image = new Blob([files[imgKey] as BlobPart], { type: imgKey.endsWith('jpg') ? 'image/jpeg' : 'image/png' });
    else {
      // No optical image: use a neutral grey drape so the geometry is still viewable.
      const c = new OffscreenCanvas(w, h);
      const ctx = c.getContext('2d')!;
      ctx.fillStyle = '#9aa3ad';
      ctx.fillRect(0, 0, w, h);
      image = await c.convertToBlob({ type: 'image/png' });
    }
    const name = stemOf(input[0].name).replace(/_result$/, '');
    const scene = await buildScene({
      name,
      image,
      heights: { data: arr.data, width: w, height: h },
      meta,
      provenance: { provider: 'result bundle', source: 'bundle', createdAt: new Date().toISOString() },
      statusLines: [`Opened from ${input.map((f) => f.name).join(', ')}`],
    });
    useTool.getState().clear();
    useScene.getState().setScene(scene);
    void import('@/features/anchoring/runAnchoring').then((m) => m.autoAnchor());
    const gtKey = ['gt_ndsm_m.npy', 'gt_dsm_m.npy'].find((k) => files[k]);
    if (gtKey) {
      const gt = parseNpy(files[gtKey].slice().buffer);
      if (gt.shape[0] === h && gt.shape[1] === w)
        useScene.getState().setReference({ name: gtKey, kind: gtKey.includes('dsm_m') && !gtKey.includes('ndsm') ? 'DSM' : 'nDSM', data: gt.data, alignment: 'same-extent', notes: ['Loaded with the bundle.'] });
    }
    useUi.getState().set({ projectOpen: false });
    notifications.show({ title: 'Result opened', message: name, color: 'teal' });
  } catch (e) {
    reportError(e, 'Could not open result bundle');
  }
}
