import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { notifications } from '@mantine/notifications';
import type { CameraBookmark, ClassMap, Georef, Provenance, ReferenceSurface, SceneMeta, SceneObjects } from '@/domain/types';
import { parseNpy, writeNpyF32 } from '@/lib/npy';
import { parseObjects, serializeObjects } from '@/lib/objects';
import { decodeGray8Png, encodeGray8Png } from '@/lib/png';
import { buildScene } from '@/lib/sceneBuilder';
import { download, safeStem } from '@/lib/download';
import { getRecentBlob, putRecent } from '@/lib/recent';
import { stemOf } from '@/lib/input';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { useCamera, viewportApi } from '@/store/camera';
import { useTool } from '@/store/tool';
import { useUi } from '@/store/ui';
import { reportError } from './openFile';

interface Manifest {
  format: 'dwproj';
  version: 1;
  name: string;
  imageName: string;
  gsd: number;
  gsdSource: 'user' | 'geotiff' | 'assumed';
  georef: Georef | null;
  meta: SceneMeta;
  statusLines: string[];
  provenance: Provenance;
  view: Record<string, unknown>;
  bookmarks: CameraBookmark[];
  tools: { probe: unknown; measure: unknown; profile: unknown };
  reference: { name: string; kind: ReferenceSurface['kind']; alignment: ReferenceSurface['alignment']; notes: string[] } | null;
  /** Present when the project carries `seg.png` (object classes). Absent in older projects. */
  classes?: { names: string[] } | null;
}

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

/** Serialise the current workspace into a .dwproj (zip) blob. */
export async function buildProjectBlob(): Promise<{ blob: Blob; name: string } | null> {
  const { scene, reference } = useScene.getState();
  if (!scene) return null;
  const { set: _s, reset: _r, ...view } = useView.getState();
  const tool = useTool.getState();
  const ext = scene.image.type === 'image/jpeg' ? 'jpg' : 'png';
  const manifest: Manifest = {
    format: 'dwproj',
    version: 1,
    name: scene.name,
    imageName: `input.${ext}`,
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
  };
  const files: Record<string, Uint8Array> = {
    'manifest.json': strToU8(JSON.stringify(manifest, null, 2)),
    [manifest.imageName]: new Uint8Array(await scene.image.arrayBuffer()),
    'ndsm_m.npy': writeNpyF32(scene.heights.data, [scene.heights.height, scene.heights.width]),
  };
  if (reference) files['reference_m.npy'] = writeNpyF32(reference.data, [scene.heights.height, scene.heights.width]);
  if (scene.classes) files['seg.png'] = new Uint8Array(await encodeGray8Png(scene.classes.data, scene.classes.width, scene.classes.height).arrayBuffer());
  // Same schema the Space publishes, so a project's objects.json is readable by anything that reads the Space's.
  if (scene.objects) files['objects.json'] = strToU8(JSON.stringify(serializeObjects(scene.objects)));
  const zip = zipSync(files, { level: 6 });
  return { blob: new Blob([zip as BlobPart], { type: 'application/zip' }), name: `${safeStem(scene.name)}.dwproj` };
}

export async function saveProject() {
  try {
    const built = await buildProjectBlob();
    if (!built) return;
    download(built.blob, built.name);
    await rememberRecent(built.blob);
    useScene.getState().set({ dirty: false });
    notifications.show({ title: 'Project saved', message: built.name, color: 'teal' });
  } catch (e) {
    reportError(e, 'Could not save project');
  }
}

/** Keep the current workspace in Recent (IndexedDB). */
export async function rememberRecent(blob?: Blob) {
  const scene = useScene.getState().scene;
  if (!scene) return;
  const b = blob ?? (await buildProjectBlob())?.blob;
  if (!b) return;
  await putRecent({ id: scene.id, name: scene.name, savedAt: new Date().toISOString(), thumbnail: await thumbnail(), size: b.size }, b);
}

async function adoptProject(bytes: Uint8Array) {
  const files = unzipSync(bytes);
  const manifest = JSON.parse(strFromU8(files['manifest.json'])) as Manifest;
  if (manifest.format !== 'dwproj') throw new Error('Not a DepthWizard project');
  const arr = parseNpy(files['ndsm_m.npy'].slice().buffer);
  const [h, w] = arr.shape;
  const img = files[manifest.imageName];
  const image = new Blob([img as BlobPart], { type: manifest.imageName.endsWith('jpg') ? 'image/jpeg' : 'image/png' });
  let classes: ClassMap | null = null;
  if (manifest.classes && files['seg.png']) {
    try {
      classes = { ...decodeGray8Png(files['seg.png'].slice().buffer), names: manifest.classes.names };
    } catch (e) {
      console.warn('Project seg.png unreadable; class layer disabled', e);
    }
  }
  let objects: SceneObjects | null = null;
  if (files['objects.json']) {
    try {
      objects = parseObjects(JSON.parse(strFromU8(files['objects.json'])), { width: w, height: h, gsd: manifest.gsd });
    } catch (e) {
      console.warn('Project objects.json unreadable; 3D objects disabled', e);
    }
  }
  const scene = await buildScene({
    name: manifest.name,
    image,
    heights: { data: arr.data, width: w, height: h },
    classes,
    objects,
    meta: manifest.meta,
    gsd: manifest.gsd,
    gsdSource: manifest.gsdSource,
    inputGeoref: manifest.georef,
    inputSize: { width: w, height: h },
    statusLines: manifest.statusLines,
    provenance: { ...manifest.provenance, source: 'project' },
  });
  useScene.getState().setScene(scene);
  if (manifest.reference && files['reference_m.npy']) {
    const ref = parseNpy(files['reference_m.npy'].slice().buffer);
    useScene.getState().setReference({ ...manifest.reference, data: ref.data });
  }
  useView.getState().set(manifest.view as never);
  useCamera.getState().set({ mode: 'orbit', bookmarks: manifest.bookmarks ?? [] });
  useTool.getState().clear();
  useUi.getState().set({ projectOpen: false });
}

export async function openProject(file: Blob & { name?: string }) {
  try {
    await adoptProject(new Uint8Array(await file.arrayBuffer()));
    notifications.show({ title: 'Project opened', message: file.name ?? 'Project', color: 'teal' });
  } catch (e) {
    reportError(e, 'Could not open project');
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
