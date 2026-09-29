import { wrap, type Remote } from 'comlink';
import type { TerrainWorkerApi } from './terrain.worker';

let terrain: Remote<TerrainWorkerApi> | null = null;

/** Lazily-created singleton terrain worker. */
export function terrainWorker(): Remote<TerrainWorkerApi> {
  if (!terrain) {
    const w = new Worker(new URL('./terrain.worker.ts', import.meta.url), { type: 'module', name: 'terrain' });
    terrain = wrap<TerrainWorkerApi>(w);
  }
  return terrain;
}
