import { notifications } from '@mantine/notifications';
import { getProvider } from '@/api/registry';
import { DepthWizardError, toDepthWizardError } from '@/api/errors';
import { buildScene } from '@/lib/sceneBuilder';
import { stemOf } from '@/lib/input';
import { useScene } from '@/store/scene';
import { useSettings } from '@/store/settings';
import { useUi } from '@/store/ui';
import { useTool } from '@/store/tool';

let controller: AbortController | null = null;

export function resolveGsd(): { gsd: number | null; source: 'user' | 'geotiff' | 'assumed' } {
  const { input, params } = useScene.getState();
  if (params.gsdMode === 'auto') {
    if (input?.fileGsd) return { gsd: input.fileGsd, source: 'geotiff' };
    return { gsd: null, source: 'assumed' };
  }
  return { gsd: params.gsd, source: 'user' };
}

/** Upload the prepared input, stream progress into the store and adopt the result as the new scene. */
export async function runPrediction() {
  const s = useScene.getState();
  const input = s.input;
  if (!input || s.run.status === 'running') return;
  const settings = useSettings.getState();
  const provider = getProvider({ provider: settings.provider, spaceId: settings.spaceId });

  controller?.abort();
  controller = new AbortController();
  const signal = controller.signal;
  const { gsd, source } = resolveGsd();
  // A TIFF may have been downsampled for the browser; the GSD sent must describe the uploaded pixels.
  let uploadGsd = gsd;
  if (gsd && input.isTiff) {
    const bmp = await createImageBitmap(input.upload);
    uploadGsd = gsd * (input.width / bmp.width);
    bmp.close();
  }

  s.setRun({ status: 'running', stage: 'connecting', event: { stage: 'connecting' }, error: null, startedAt: performance.now() });
  s.set({ inputPreview: true });
  try {
    const res = await provider.predict(
      { upload: input.upload, uploadName: input.uploadName, gsd: uploadGsd, tta: s.params.tta },
      {
        signal,
        onProgress: (ev) => useScene.getState().setRun({ stage: ev.stage, event: ev }),
      },
    );
    useScene.getState().setRun({ stage: 'building', event: { stage: 'building' } });
    const effGsd = res.status.gsd ?? res.meta.scene?.gsd_m ?? uploadGsd ?? 0.5;
    const effSource = source === 'geotiff' ? 'geotiff' : (res.status.gsdSource ?? res.meta.scene?.gsd_source ?? source);
    const scene = await buildScene({
      name: stemOf(input.name),
      image: input.display,
      heights: res.heights,
      classes: res.classes,
      objects: res.objects,
      meta: res.meta,
      gsd: effGsd,
      gsdSource: effSource,
      inputGeoref: input.georef,
      inputSize: { width: input.width, height: input.height },
      artefacts: res.artefacts,
      statusLines: res.status.lines,
      warning: res.warning,
      resampleNote: res.status.resampleNote,
      provenance: {
        provider: provider.label,
        source: 'inference',
        createdAt: new Date().toISOString(),
        params: { gsd: uploadGsd, tta: s.params.tta },
        modelVersion: typeof res.meta.preproc?.version === 'string' ? `v${String(res.meta.preproc.version).replace(/^v/, '')}` : undefined,
      },
    });
    useTool.getState().clear();
    useScene.getState().setScene(scene);
    // the input image stays up until the user chooses to explore the result
    useScene.getState().set({ dirty: true, inputPreview: true });
    // georeferenced results become an absolute DSM once a DEM is fetched (rDSM stays relative)
    void import('@/features/anchoring/runAnchoring').then((m) => m.autoAnchor());
    // a re-run of the same image keeps its ground control points
    void import('@/features/gcp/gcpStore').then((m) => m.reapplyGcps());
    // Result replaces the empty state: collapse the project panel so the viewport matches the product view.
    useUi.getState().set({ projectOpen: window.innerWidth >= 1440 });
    // 'done' is set once the mesh is built (see Terrain); keep the stepper on "Building 3D mesh".
    // Keep every fresh result in File → Recent (after the mesh has rendered, for the thumbnail).
    setTimeout(() => void import('@/features/files/project').then((m) => m.rememberRecent()), 4000);
  } catch (e) {
    const err = toDepthWizardError(e);
    if (err.kind === 'cancelled') {
      useScene.getState().setRun({ status: 'idle', stage: null, event: null, error: null });
      useScene.getState().set({ inputPreview: false });
      notifications.show({ title: 'Run cancelled', message: 'The previous result is still shown.', color: 'gray' });
    } else {
      useScene.getState().setRun({ status: 'error', error: err });
    }
  } finally {
    controller = null;
  }
}

export function cancelPrediction() {
  controller?.abort(new DepthWizardError('cancelled', 'Run cancelled'));
}

export function markRunDone() {
  const run = useScene.getState().run;
  if (run.status === 'running') useScene.getState().setRun({ status: 'done', stage: 'done', event: { stage: 'done' } });
}
