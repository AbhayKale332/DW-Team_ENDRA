import { notifications } from '@mantine/notifications';
import { getProvider } from '@/api/registry';
import { DepthWizardError, toDepthWizardError } from '@/api/errors';
import { buildScene } from '@/lib/sceneBuilder';
import { stemOf, type PreparedInput } from '@/lib/input';
import { MIN_COVERAGE, type CloudMask } from '@/lib/cloud';
import type { SceneCloud } from '@/domain/types';
import { analysisWorker } from '@/workers/clients';
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

/** The clouds to mask for this run: none when the switch is off. Waits for (or redoes) a pending detection. */
async function cloudsToMask(input: PreparedInput): Promise<CloudMask | null> {
  const s = useScene.getState();
  if (!s.params.cloudMask) return null;
  if (s.inputCloud?.status === 'done') return s.inputCloud.mask;
  const found = await analysisWorker().cloudDetect(input.upload);
  return found.coverage >= MIN_COVERAGE ? found : null;
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
    // Clouds: the model gets the image with them painted over (same size, so the GSD above still holds).
    const clouds = await cloudsToMask(input);
    const cloudFree = clouds ? await analysisWorker().cloudFill(input.upload, clouds) : null;
    if (signal.aborted) throw new DepthWizardError('cancelled', 'Run cancelled');
    const res = await provider.predict(
      {
        upload: cloudFree ?? input.upload,
        uploadName: cloudFree ? `${stemOf(input.uploadName)}.png` : input.uploadName,
        gsd: uploadGsd,
        tta: s.params.tta,
      },
      {
        signal,
        onProgress: (ev) => useScene.getState().setRun({ stage: ev.stage, event: ev }),
      },
    );
    useScene.getState().setRun({ stage: 'building', event: { stage: 'building' } });
    let heights = res.heights;
    let cloud: SceneCloud | null = null;
    if (clouds && cloudFree) {
      const filled = await analysisWorker().cloudHeights(heights.data, heights.width, heights.height, clouds);
      heights = { ...heights, data: filled };
      cloud = { ...clouds, image: cloudFree };
    }
    const effGsd = res.status.gsd ?? res.meta.scene?.gsd_m ?? uploadGsd ?? 0.5;
    const effSource = source === 'geotiff' ? 'geotiff' : (res.status.gsdSource ?? res.meta.scene?.gsd_source ?? source);
    const scene = await buildScene({
      name: stemOf(input.name),
      image: input.display,
      heights,
      uncertainty: res.uncertainty,
      classes: res.classes,
      objects: res.objects,
      cloud,
      meta: { ...res.meta, source_image_name: input.name },
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
        params: { gsd: uploadGsd, tta: s.params.tta, cloudMask: !!cloud },
        modelVersion: typeof res.meta.preproc?.version === 'string' ? `v${String(res.meta.preproc.version).replace(/^v/, '')}` : undefined,
      },
    });
    useTool.getState().clear();
    useScene.getState().setScene(scene);
    // straight to the 3D result (the mesh builds in view)
    useScene.getState().set({ dirty: true, inputPreview: false });
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
