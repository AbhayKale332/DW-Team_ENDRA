import { create } from 'zustand';
import type { HeightStats, Scene } from '@/domain/types';
import type { TerrainFrame } from '@/lib/pick';
import { objectSurfaceIfReady } from './objectSurfaceClient';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';

interface TerrainInfo {
  vertices: number;
  step: number;
  wallCells: number;
}

interface TerrainState {
  info: TerrainInfo | null;
  fps: number;
  set: (p: Partial<Omit<TerrainState, 'set'>>) => void;
}

export const useTerrainInfo = create<TerrainState>()((set) => ({ info: null, fps: 0, set: (p) => set(p) }));

/** Vertical scale used to flatten the mesh in the 2D views (keeps normals valid). */
export const FLAT_SCALE = 1e-4;

export function sceneBase(scene: Scene) {
  return scene.stats.min;
}

export function sceneExtent(scene: Scene) {
  return {
    x: (scene.heights.width - 1) * scene.gsd,
    z: (scene.heights.height - 1) * scene.gsd,
    h: scene.stats.max - scene.stats.min,
  };
}

const groundCache = new WeakMap<Scene, number>();

/** Height (above sceneBase, metres) of the ground just outside the grid: the 10th percentile of its border
 *  pixels, low enough to skip buildings and trees cut by the edge. The basemap and off-grid facilities sit here. */
export function groundLevel(scene: Scene): number {
  const hit = groundCache.get(scene);
  if (hit !== undefined) return hit;
  const { data, width: W, height: H } = scene.heights;
  const vals: number[] = [];
  const push = (i: number) => {
    if (Number.isFinite(data[i])) vals.push(data[i]);
  };
  for (let c = 0; c < W; c++) {
    push(c);
    push((H - 1) * W + c);
  }
  for (let r = 1; r < H - 1; r++) {
    push(r * W);
    push(r * W + W - 1);
  }
  vals.sort((a, b) => a - b);
  const g = vals.length ? vals[Math.floor(vals.length * 0.1)] - sceneBase(scene) : 0;
  groundCache.set(scene, g);
  return g;
}

export function verticalScale() {
  const v = useView.getState();
  return v.mode === 'dsm3d' ? v.exaggeration : FLAT_SCALE;
}

/** The model's heights — what probe, profile and measurement read, whatever the mesh shows. */
export function currentFrame(): TerrainFrame | null {
  const scene = useScene.getState().scene;
  if (!scene) return null;
  return { grid: scene.heights, gsd: scene.gsd, base: sceneBase(scene), scaleY: verticalScale() };
}

/** The surface as meshed — what walk and flight cameras must follow. With 3D objects on this is the ground
 *  under the models rather than their canopy or roof bumps, which would otherwise leave you walking on treetops. */
export function visibleFrame(): TerrainFrame | null {
  const scene = useScene.getState().scene;
  if (!scene) return null;
  // until the worker has the flattened surface, follow the raw one (only for the first moments of a scene)
  const flat = objectSurfaceIfReady(scene, useView.getState().objectKinds);
  const grid = flat ? { ...scene.heights, data: flat } : scene.heights;
  return { grid, gsd: scene.gsd, base: sceneBase(scene), scaleY: verticalScale() };
}

/** Colour range for height layers, in absolute metres. */
export function displayRange(stats: HeightStats, mode = useView.getState().rangeMode, custom = useView.getState().customRange): [number, number] {
  if (mode === 'custom') return custom[1] > custom[0] ? custom : [custom[0], custom[0] + 1];
  if (mode === 'full') return [stats.min, Math.max(stats.max, stats.min + 0.5)];
  return [stats.p2, Math.max(stats.p98, stats.p2 + 0.5)];
}
