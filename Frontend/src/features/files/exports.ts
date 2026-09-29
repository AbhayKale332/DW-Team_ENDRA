import { notifications } from '@mantine/notifications';
import { useScene } from '@/store/scene';
import { activeObjectKinds, anyObjectKind, useView } from '@/store/view';
import { viewportApi } from '@/store/camera';
import { download, safeStem } from '@/lib/download';
import { displayRange } from '@/features/viewport/terrainState';
import { reportError } from './openFile';

export type ExportKind = 'glb' | 'obj' | 'ply' | 'stl' | 'png' | 'jpg' | 'geotiff' | 'npy' | 'png16' | 'screenshot' | 'server-glb';

export const EXPORT_LABELS: Record<ExportKind, string> = {
  glb: 'GLB (glTF binary, textured)',
  obj: 'OBJ + MTL + texture (.zip)',
  ply: 'PLY (vertex-coloured)',
  stl: 'STL (3D printing)',
  png: 'Heatmap PNG',
  jpg: 'Heatmap JPG',
  geotiff: 'GeoTIFF (Float32 heights)',
  npy: 'NumPy array (.npy, metres)',
  png16: '16-bit PNG (encoded heights)',
  screenshot: 'Viewport screenshot (PNG)',
  'server-glb': 'Model output mesh (terrain.glb)',
};

/** Run one export with progress/success/failure notifications. */
export async function runExport(kind: ExportKind) {
  const scene = useScene.getState().scene;
  if (!scene) return;
  const v = useView.getState();
  const stem = safeStem(scene.name);
  const id = notifications.show({ loading: true, title: 'Exporting', message: EXPORT_LABELS[kind], autoClose: false, withCloseButton: false });
  try {
    const range = displayRange(scene.stats, v.rangeMode, v.customRange);
    const meshOpts = { budget: 600_000, wallThreshold: v.walls ? v.wallThreshold : 0, exaggeration: v.exaggeration, colormap: v.colormap, range };
    let blob: Blob | null = null;
    let name = '';
    switch (kind) {
      case 'glb': {
        const { exportGlb } = await import('@/lib/exporters/mesh');
        // WYSIWYG: the GLB carries the object classes that are switched on
        const withObjects = anyObjectKind(activeObjectKinds(v.objectKinds, scene.objects));
        blob = await exportGlb(scene, { ...meshOpts, objects: v.objectKinds });
        name = withObjects ? `${stem}_scene.glb` : `${stem}_terrain.glb`;
        break;
      }
      case 'obj': {
        const { exportObjZip } = await import('@/lib/exporters/mesh');
        blob = await exportObjZip(scene, meshOpts);
        name = `${stem}_terrain_obj.zip`;
        break;
      }
      case 'ply': {
        const { exportPly } = await import('@/lib/exporters/mesh');
        blob = await exportPly(scene, meshOpts);
        name = `${stem}_terrain.ply`;
        break;
      }
      case 'stl': {
        const { exportStl } = await import('@/lib/exporters/mesh');
        blob = await exportStl(scene, { ...meshOpts, exaggeration: 1 });
        name = `${stem}_terrain.stl`;
        break;
      }
      case 'png':
      case 'jpg': {
        const { exportHeatmap } = await import('@/lib/exporters/raster');
        blob = await exportHeatmap(scene, kind === 'png' ? 'image/png' : 'image/jpeg', {
          colormap: v.colormap,
          range,
          hillshade: v.hillshadeStrength,
          colorbar: true,
          sunAzimuth: v.sunAzimuth,
          sunElevation: v.sunElevation,
        });
        name = `${stem}_heatmap.${kind}`;
        break;
      }
      case 'geotiff': {
        const { exportGeoTiff } = await import('@/lib/exporters/raster');
        blob = exportGeoTiff(scene);
        name = `${stem}_${scene.product.toLowerCase()}_m.tif`;
        break;
      }
      case 'npy': {
        const { exportNpy } = await import('@/lib/exporters/raster');
        blob = exportNpy(scene);
        name = `${stem}_ndsm_m.npy`;
        break;
      }
      case 'png16': {
        const { export16BitPng } = await import('@/lib/exporters/raster');
        blob = export16BitPng(scene);
        name = `${stem}_height16.png`;
        break;
      }
      case 'screenshot': {
        blob = await viewportApi.screenshot();
        name = `${stem}_view.png`;
        break;
      }
      case 'server-glb': {
        const a = scene.artefacts.find((x) => x.name === 'terrain.glb');
        if (!a?.url) throw new Error('This result has no server mesh.');
        const r = await fetch(a.url);
        blob = await r.blob();
        name = `${stem}_model_terrain.glb`;
        break;
      }
    }
    if (!blob) throw new Error('Nothing was produced.');
    download(blob, name);
    notifications.update({ id, loading: false, color: 'teal', title: 'Export ready', message: `${name} · ${(blob.size / 1e6).toFixed(1)} MB`, autoClose: 3500, withCloseButton: true });
  } catch (e) {
    notifications.hide(id);
    reportError(e, 'Export failed');
  }
}
