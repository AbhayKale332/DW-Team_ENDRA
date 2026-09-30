import type { Scene } from '@/domain/types';
import { assessBuildings, type FloodModel } from '@/lib/usecases/flood';
import { buildAnalysisGrid, type AnalysisGrid } from '@/lib/usecases/grid';
import { DEFAULT_TOWER, type Tower } from '@/lib/usecases/telecom';
import { usePoi, poisFor } from '@/features/poi/poiStore';
import { useScene } from '@/store/scene';
import { useUseCases, type UseCaseSnapshot } from '@/store/usecases';
import { analysisWorker } from '@/workers/clients';

/** The analysis grid of a scene, built once per scene object and mirrored into the analysis worker. */
const cache = new WeakMap<Scene, { grid: AnalysisGrid; id: string; ready: Promise<void> }>();
let counter = 0;

export function analysisGridFor(scene: Scene) {
  let hit = cache.get(scene);
  if (!hit) {
    const grid = buildAnalysisGrid(scene);
    const id = `${scene.id}:${++counter}`;
    hit = { grid, id, ready: Promise.resolve(analysisWorker().setGrid(id, grid)) };
    cache.set(scene, hit);
  }
  return hit;
}

let seq = { coverage: 0, suggest: 0, flood: 0 };
let timer: ReturnType<typeof setTimeout> | undefined;
/** Water level of a reopened project: applied once its flood model has been rebuilt (a new scene resets the level). */
let pendingRise: { sceneId: string; rise: number } | null = null;

const uid = () => `t${Date.now().toString(36)}${Math.floor(Math.random() * 1e4).toString(36)}`;

/** Recompute coverage for the current towers (debounced: dragging a slider fires many changes). */
export function refreshCoverage(delay = 150) {
  clearTimeout(timer);
  timer = setTimeout(() => void runCoverage(), delay);
}

export async function runCoverage() {
  const scene = useScene.getState().scene;
  const u = useUseCases.getState();
  if (!scene) return;
  if (!u.towers.length) {
    u.set({ coverage: null, coverageStatus: 'idle', suggestions: [] });
    return;
  }
  const my = ++seq.coverage;
  u.set({ coverageStatus: 'running' });
  try {
    const { id, ready } = analysisGridFor(scene);
    await ready;
    const res = await analysisWorker().coverage(id, u.towers, u.params);
    if (my !== seq.coverage) return;
    useUseCases.getState().set({ coverage: res, coverageStatus: 'done', suggestions: [] });
  } catch (e) {
    if (my !== seq.coverage) return;
    console.error('Coverage failed', e);
    useUseCases.getState().set({ coverageStatus: 'error' });
  }
}

export function addTower(col: number, row: number, extra: Partial<Tower> = {}): string {
  const t: Tower = { id: uid(), col, row, ...DEFAULT_TOWER, ...extra };
  const u = useUseCases.getState();
  u.set({ towers: [...u.towers, t], selectedTower: t.id, placing: false });
  refreshCoverage(0);
  return t.id;
}

export function updateTower(id: string, patch: Partial<Tower>) {
  const u = useUseCases.getState();
  u.set({ towers: u.towers.map((t) => (t.id === id ? { ...t, ...patch } : t)) });
  refreshCoverage();
}

export function removeTower(id: string) {
  const u = useUseCases.getState();
  u.set({ towers: u.towers.filter((t) => t.id !== id), selectedTower: u.selectedTower === id ? null : u.selectedTower });
  refreshCoverage(0);
}

export async function suggestTowers() {
  const scene = useScene.getState().scene;
  const u = useUseCases.getState();
  if (!scene || !u.towers.length) return;
  const my = ++seq.suggest;
  u.set({ suggesting: true });
  try {
    const { id, ready } = analysisGridFor(scene);
    await ready;
    const base = u.towers[u.towers.length - 1];
    const picks = await analysisWorker().suggest(id, u.towers, u.params, 3, { heightM: base.heightM, eirpDbm: base.eirpDbm });
    if (my === seq.suggest) useUseCases.getState().set({ suggestions: picks, suggesting: false });
  } catch (e) {
    console.error('Site suggestion failed', e);
    if (my === seq.suggest) useUseCases.getState().set({ suggesting: false });
  }
}

/** Build the flood model for the chosen source, then rank the scene's buildings against it. */
export async function runFlood() {
  const scene = useScene.getState().scene;
  const u = useUseCases.getState();
  if (!scene) return;
  const my = ++seq.flood;
  u.set({ floodStatus: 'running', floodError: null });
  try {
    const { id, grid, ready } = analysisGridFor(scene);
    await ready;
    const source = u.floodSource === 'point' ? (u.floodPoint ? { kind: 'point' as const, ...u.floodPoint } : null) : { kind: u.floodSource };
    if (!source) {
      useUseCases.getState().set({ floodStatus: 'idle', floodError: 'Click the terrain to choose where the water comes from.', flood: null, risks: [], pickingSource: true });
      return;
    }
    const model = await analysisWorker().flood(id, source);
    if (my !== seq.flood) return;
    if ('error' in model) {
      useUseCases.getState().set({ floodStatus: 'error', floodError: model.error, flood: null, risks: [] });
      return;
    }
    const restored = pendingRise?.sceneId === scene.id ? pendingRise.rise : null;
    pendingRise = null;
    useUseCases.getState().set({ flood: model, floodStatus: 'done', rise: Math.min(restored ?? useUseCases.getState().rise, model.maxRise), risks: risksFor(scene, grid, model) });
  } catch (e) {
    if (my !== seq.flood) return;
    useUseCases.getState().set({ floodStatus: 'error', floodError: e instanceof Error ? e.message : String(e), flood: null, risks: [] });
  }
}

function risksFor(scene: Scene, grid: AnalysisGrid, model: FloodModel) {
  const b = scene.objects?.buildings;
  if (!b?.length) return [];
  return assessBuildings(b, grid, model, poisFor(usePoi.getState(), scene), scene.gsd);
}

/** Facilities arrived after the model was built: re-rank with them. */
export function rerankWithFacilities() {
  const scene = useScene.getState().scene;
  const { flood } = useUseCases.getState();
  if (!scene || !flood) return;
  useUseCases.getState().set({ risks: risksFor(scene, analysisGridFor(scene).grid, flood) });
}

/** Scenario inputs of a reopened project; coverage and flood results are recomputed from them. */
export function restoreUseCases(sceneId: string, snap: UseCaseSnapshot) {
  const { rise, ...rest } = snap;
  pendingRise = typeof rise === 'number' && rise > 0 ? { sceneId, rise } : null;
  useUseCases.getState().reset();
  useUseCases.getState().set(rest);
  if (rest.towers?.length) refreshCoverage(0);
}

/** Drop results computed for another scene / terrain. */
export function invalidateUseCases() {
  seq = { coverage: seq.coverage + 1, suggest: seq.suggest + 1, flood: seq.flood + 1 };
  useUseCases.getState().set({ coverage: null, coverageStatus: 'idle', suggestions: [], flood: null, floodStatus: 'idle', floodError: null, risks: [], playing: false, rise: 0, focusBuilding: null });
}
