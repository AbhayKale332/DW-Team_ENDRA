import { wrap, type Remote } from 'comlink';
import type { TerrainWorkerApi } from './terrain.worker';
import type { AnalysisWorkerApi } from './analysis.worker';
import type { FloodSimWorkerApi } from './floodSim.worker';

let terrain: Remote<TerrainWorkerApi> | null = null;
let analysis: Remote<AnalysisWorkerApi> | null = null;
let floodSim: Remote<FloodSimWorkerApi> | null = null;

/** Lazily-created singleton terrain worker. */
export function terrainWorker(): Remote<TerrainWorkerApi> {
  if (!terrain) {
    const w = new Worker(new URL('./terrain.worker.ts', import.meta.url), { type: 'module', name: 'terrain' });
    terrain = wrap<TerrainWorkerApi>(w);
  }
  return terrain;
}

/** Lazily-created singleton worker for the use-case analyses (telecom coverage, flood). */
export function analysisWorker(): Remote<AnalysisWorkerApi> {
  if (!analysis) {
    const w = new Worker(new URL('./analysis.worker.ts', import.meta.url), { type: 'module', name: 'analysis' });
    analysis = wrap<AnalysisWorkerApi>(w);
  }
  return analysis;
}

/** Lazily-created singleton worker stepping the flood hydraulics. */
export function floodSimWorker(): Remote<FloodSimWorkerApi> {
  if (!floodSim) {
    const w = new Worker(new URL('./floodSim.worker.ts', import.meta.url), { type: 'module', name: 'flood-sim' });
    floodSim = wrap<FloodSimWorkerApi>(w);
  }
  return floodSim;
}
