import { notifications } from '@mantine/notifications';
import { prepareInput } from '@/lib/input';
import { useScene } from '@/store/scene';
import { useUi } from '@/store/ui';
import { DepthWizardError, toDepthWizardError } from '@/api/errors';
import { downloadSample, sampleUrl, useSampleLoad, type SampleDef, type SampleLoad } from '@/lib/samples';
import { hasProjectCore, projectStream } from '@/lib/dwproj';
import { useTool } from '@/store/tool';
import { useCamera } from '@/store/camera';
import { useView } from '@/store/view';
import { registerBundledOsm } from '@/features/osm/osmStore';
import { registerBundledPois } from '@/features/poi/poiStore';

/** Show the native file picker. Resolves with the chosen files (empty if cancelled). */
export function pickFiles(accept: string, multiple = false): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = accept;
    input.multiple = multiple;
    input.style.display = 'none';
    input.addEventListener('change', () => {
      resolve(Array.from(input.files ?? []));
      input.remove();
    });
    input.addEventListener('cancel', () => {
      resolve([]);
      input.remove();
    });
    document.body.appendChild(input);
    input.click();
  });
}

export const IMAGE_ACCEPT = '.png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff';
export const OPEN_ACCEPT = `${IMAGE_ACCEPT},.dwproj,.zip,.npy,.json`;

export function reportError(e: unknown, title?: string) {
  const err = toDepthWizardError(e);
  notifications.show({ color: 'red', title: title ?? err.message, message: err.detail ?? err.message, autoClose: 8000 });
}

/** Stage an image as the input for a new analysis (does not run inference). */
export async function stageImage(file: File) {
  try {
    const id = notifications.show({ loading: true, title: 'Reading image', message: file.name, autoClose: false, withCloseButton: false });
    const input = await prepareInput(file);
    notifications.hide(id);
    useScene.getState().setInput(input);
    // GeoTIFFs default to their own resolution; plain images keep the mode the user last chose.
    if (input.fileGsd) useScene.getState().setParams({ gsdMode: 'auto' });
    useUi.getState().set({ projectOpen: true });
    void import('@/features/input/cloudPrompt').then((m) => m.detectInputClouds(input, { ask: true }));
  } catch (e) {
    notifications.clean();
    reportError(e, e instanceof DepthWizardError ? e.message : 'Could not open image');
  }
}

/** Route any opened file(s) to the right loader: images, projects or result bundles. */
export async function openFiles(files: File[]) {
  if (!files.length) return;
  const first = files[0];
  const lower = first.name.toLowerCase();
  if (lower.endsWith('.dwproj')) {
    const { openProject } = await import('./project');
    return openProject(first);
  }
  if (lower.endsWith('.zip') || files.some((f) => f.name.toLowerCase().endsWith('.npy'))) {
    const { openResultBundle } = await import('./project');
    return openResultBundle(files);
  }
  return stageImage(first);
}

/** Open a sample scene: its .dwproj, like any project, unzipped as it downloads. The scene (image, height map,
 *  classes, objects) opens as soon as its files are in; the source image and model outputs follow in the background.
 *  Progress shows in the viewport (SampleLoading). */
export async function openSample(def: SampleDef) {
  useSampleLoad.getState().controller?.abort();
  const controller = new AbortController();
  const current = () => useSampleLoad.getState().controller === controller;
  const stage = (stage: SampleLoad['stage']) => useSampleLoad.setState((s) => (s.load ? { load: { ...s.load, stage } } : s));
  useSampleLoad.setState({ controller, load: { name: def.name, stage: 'download', loaded: 0, total: 0 } });
  // the scene opened early, which the background download completes (assigned in callbacks, hence the casts)
  let sceneId = null as string | null;
  try {
    const url = sampleUrl(def);
    // Overpass answers cached at build time (scripts/cache-osm.mjs) sit beside the .dwproj
    const dir = url.slice(0, url.lastIndexOf('/') + 1);
    const optionalJson = (name: string) =>
      fetch(dir + name, { signal: controller.signal })
        .then((x) => (x.ok ? (x.json() as Promise<unknown>) : null))
        .catch(() => null);
    const bundled = Promise.all([optionalJson('osm.json'), optionalJson('pois.json')]);
    const project = await import('./project');
    const open = async (files: Record<string, Uint8Array>) => {
      stage('open');
      if (!current() || !(await project.openSampleFiles(files))) return null;
      const id = useScene.getState().scene?.id ?? null;
      const [osm, pois] = await bundled;
      if (id && osm) registerBundledOsm(id, osm);
      if (id && pois) registerBundledPois(id, pois);
      return id;
    };

    let early = null as Promise<string | null> | null;
    const stream = projectStream(() => {
      // projects with the manifest stored last cannot show early: they open when complete
      if (!hasProjectCore(stream.files)) return;
      early = open({ ...stream.files }).then((id) => {
        sceneId = id;
        if (current()) {
          if (id) stage('extras');
          else controller.abort(); // the error is shown; the rest would not help
        }
        return id;
      });
    });
    await downloadSample(url, def.file, controller.signal, (chunk) => {
      stream.push(chunk);
      // another scene replaced the sample: the rest of it is not needed any more
      if (sceneId && useScene.getState().scene?.id !== sceneId) controller.abort();
    });
    stream.push(new Uint8Array(0), true);
    if (!current()) return;
    if (!early) await open(stream.files);
    else if ((await early) && current() && project.attachProjectExtras(sceneId!, stream.files))
      notifications.show({ color: 'teal', title: 'Full scene downloaded', message: def.name, autoClose: 4000 });
  } catch (e) {
    if (!controller.signal.aborted) reportError(e, sceneId ? 'Could not download the full sample' : 'Could not load sample');
  } finally {
    if (current()) useSampleLoad.setState({ load: null, controller: null });
  }
}

export function newProject() {
  const { dirty, scene } = useScene.getState();
  const go = () => {
    useScene.getState().clearAll();
    useTool.getState().clear();
    useCamera.getState().set({ mode: 'orbit', bookmarks: [] });
    useView.getState().reset();
    useUi.getState().set({ projectOpen: true, inspectorOpen: false });
  };
  if (scene && dirty) {
    void import('@mantine/modals').then(({ modals }) =>
      modals.openConfirmModal({
        title: 'Start a new project?',
        children: 'The current result has not been saved. Starting a new project discards it.',
        labels: { confirm: 'Discard and start new', cancel: 'Cancel' },
        confirmProps: { color: 'red' },
        onConfirm: go,
      }),
    );
  } else go();
}
