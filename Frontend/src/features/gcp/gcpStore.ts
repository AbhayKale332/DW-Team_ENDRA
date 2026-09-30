import { create } from 'zustand';
import { notifications } from '@mantine/notifications';
import type { GroundControlPoint, Scene } from '@/domain/types';
import { applyGcpsToScene, hasElev, hasLatLon } from '@/lib/gcp';
import { useAnchor } from '@/store/anchor';
import { useScene } from '@/store/scene';
import { reportError } from '@/features/files/openFile';

interface GcpState {
  /** Scene the draft belongs to. */
  sceneId: string | null;
  /** Image + grid the draft was made on: a re-run of the same image keeps it. */
  key: string | null;
  /** Points being edited (applied ones live on scene.gcps). */
  points: GroundControlPoint[];
  /** The next click on the terrain adds a point. */
  picking: boolean;
}

export const useGcp = create<GcpState>()(() => ({ sceneId: null, key: null, points: [], picking: false }));

const keyOf = (s: Scene) => {
  const g = s.ndsm ?? s.heights;
  return `${s.name}:${g.width}x${g.height}`;
};

/** GCPs can be applied: an elevation makes heights absolute, three lat/lon points georeference the scene. */
export const gcpsReady = (points: GroundControlPoint[]) => points.some(hasElev) || points.filter(hasLatLon).length >= 3;

// Follow the current scene: its applied points, else the draft of an earlier run of the same image.
useScene.subscribe((s, prev) => {
  const scene = s.scene;
  if (scene === prev.scene) return;
  const st = useGcp.getState();
  if (!scene) return useGcp.setState({ sceneId: null, points: [], picking: false });
  if (st.sceneId === scene.id) return;
  const key = keyOf(scene);
  useGcp.setState({ sceneId: scene.id, key, picking: false, points: scene.gcps ?? (st.key === key ? st.points : []) });
});

const uid = () => `g${Date.now().toString(36)}${Math.floor(Math.random() * 1e4).toString(36)}`;

export function addGcp(col: number, row: number) {
  const st = useGcp.getState();
  useGcp.setState({ points: [...st.points, { id: uid(), col, row, lat: null, lon: null, elev: null }], picking: false });
}

export function updateGcp(id: string, patch: Partial<GroundControlPoint>) {
  useGcp.setState({ points: useGcp.getState().points.map((p) => (p.id === id ? { ...p, ...patch } : p)) });
}

export function removeGcp(id: string) {
  useGcp.setState({ points: useGcp.getState().points.filter((p) => p.id !== id) });
}

/** Apply the draft to the current scene (an empty draft removes applied GCPs). */
export async function applyGcps(opts: { silent?: boolean } = {}) {
  const scene = useScene.getState().scene;
  if (!scene) return;
  const points = useGcp.getState().points;
  try {
    const { scene: next } = applyGcpsToScene(scene, points);
    if (!points.length) next.gcps = undefined;
    useScene.getState().updateScene(next);
    useScene.getState().set({ dirty: true });
    useGcp.setState({ picking: false });
    if (next.anchoring) useAnchor.getState().set({ status: 'done', message: null, sceneId: next.id });
    // georeferenced without elevations: a DEM can make it absolute
    else (await import('@/features/anchoring/runAnchoring')).autoAnchor();
    if (!opts.silent && points.length) notifications.show({ title: 'Ground control points applied', message: `${points.length} point${points.length > 1 ? 's' : ''}`, color: 'teal' });
  } catch (e) {
    reportError(e, 'Could not apply ground control points');
  }
}

export function clearGcps() {
  useGcp.setState({ points: [], picking: false });
  void applyGcps({ silent: true });
}

/** After a re-run of the same image: apply the points kept from the previous result. */
export function reapplyGcps() {
  const scene = useScene.getState().scene;
  const st = useGcp.getState();
  if (scene && !scene.georef && !scene.gcps && st.sceneId === scene.id && gcpsReady(st.points)) void applyGcps({ silent: true });
}
