import { notifications } from '@mantine/notifications';
import { prepareInput } from '@/lib/input';
import { useScene } from '@/store/scene';
import { useUi } from '@/store/ui';
import { DepthWizardError, toDepthWizardError } from '@/api/errors';
import { downloadSample, sampleUrl, useSampleLoad, type SampleDef } from '@/lib/samples';
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

/** Open a sample scene: its .dwproj, like any project. Progress shows in the viewport (SampleLoading). */
export async function openSample(def: SampleDef) {
  useSampleLoad.getState().controller?.abort();
  const controller = new AbortController();
  const current = () => useSampleLoad.getState().controller === controller;
  useSampleLoad.setState({ controller, load: { name: def.name, stage: 'download', loaded: 0, total: 0 } });
  try {
    const url = sampleUrl(def);
    // Overpass answers cached at build time (scripts/cache-osm.mjs) sit beside the .dwproj
    const dir = url.slice(0, url.lastIndexOf('/') + 1);
    const optionalJson = (name: string) =>
      fetch(dir + name, { signal: controller.signal })
        .then((x) => (x.ok ? (x.json() as Promise<unknown>) : null))
        .catch(() => null);
    const [blob, osm, pois] = await Promise.all([downloadSample(url, def.file, controller.signal), optionalJson('osm.json'), optionalJson('pois.json')]);
    if (!current()) return;
    useSampleLoad.setState((s) => ({ load: s.load && { ...s.load, stage: 'open' } }));
    const { openProject } = await import('./project');
    if (current() && (await openProject(blob, { sample: true }))) {
      const id = useScene.getState().scene?.id;
      if (id && osm) registerBundledOsm(id, osm);
      if (id && pois) registerBundledPois(id, pois);
    }
  } catch (e) {
    if (!controller.signal.aborted) reportError(e, 'Could not load sample');
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
