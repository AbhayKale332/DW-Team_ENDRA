import { notifications } from '@mantine/notifications';
import { anchorBlocker, anchorScene, DemError, type AnchorRequest, type DemCellsCache } from '@/lib/dem';
import { useAnchor } from '@/store/anchor';
import { useScene } from '@/store/scene';
import { useSettings } from '@/store/settings';

let controller: AbortController | null = null;
/** DEM cells shipped with a bundled sample, keyed by scene id, so it can be anchored offline. */
const bundled = new Map<string, DemCellsCache>();

export function registerBundledDem(sceneId: string, cache: DemCellsCache) {
  bundled.set(sceneId, cache);
}

/** Anchor the current scene to a DEM. Failure never removes the result: the scene simply stays an nDSM. */
export async function runAnchoring(req: AnchorRequest, opts: { silent?: boolean } = {}) {
  const scene = useScene.getState().scene;
  if (!scene) return;
  const blocker = anchorBlocker(scene);
  if (blocker) {
    useAnchor.getState().set({ status: 'blocked', message: blocker, sceneId: scene.id });
    return;
  }
  controller?.abort();
  const ctl = new AbortController();
  controller = ctl;
  useAnchor.getState().set({ status: 'running', message: 'Starting', sceneId: scene.id });
  try {
    const next = await anchorScene(scene, req, {
      signal: ctl.signal,
      onProgress: (message) => useAnchor.getState().set({ message, sceneId: scene.id }),
    });
    if (ctl.signal.aborted || useScene.getState().scene?.id !== scene.id) return;
    // rebase on the latest scene object: the user may have toggled a layer meanwhile, but heights are unchanged
    useScene.getState().updateScene({ ...next, warnings: useScene.getState().scene?.warnings ?? next.warnings });
    useAnchor.getState().set({ status: 'done', message: null, sceneId: scene.id });
    if (!opts.silent) notifications.show({ title: 'Absolute DSM ready', message: `Anchored to ${next.anchoring?.source ?? 'the DEM'}.`, color: 'green' });
  } catch (e) {
    if (ctl.signal.aborted || (e instanceof DemError && e.kind === 'cancelled')) return;
    const message = e instanceof Error ? e.message : String(e);
    useAnchor.getState().set({ status: 'error', message, sceneId: scene.id });
    if (!opts.silent) notifications.show({ title: 'Could not anchor to a DEM', message: `${message} The result stays a height-above-ground model (nDSM).`, color: 'orange' });
  } finally {
    if (controller === ctl) controller = null;
  }
}

/** After a result is adopted: anchor georeferenced scenes automatically (if enabled), note why others are not. */
export function autoAnchor() {
  const scene = useScene.getState().scene;
  if (!scene) return;
  useAnchor.getState().set({ status: 'idle', message: null, sceneId: scene.id });
  const blocker = anchorBlocker(scene);
  if (blocker) {
    useAnchor.getState().set({ status: 'blocked', message: blocker, sceneId: scene.id });
    return;
  }
  const cache = bundled.get(scene.id);
  if (cache) return void runAnchoring({ kind: 'cells', cache }, { silent: true });
  if (!useSettings.getState().autoAnchor) return;
  void runAnchoring({ kind: 'terrain-tiles' }, { silent: true });
}
