import { expose } from 'comlink';
import type { AnalysisGrid } from '@/lib/usecases/grid';
import { buildFloodModel, type FloodError, type FloodModel, type FloodSource } from '@/lib/usecases/flood';
import { computeCoverage, suggestSites, type CoverageResult, type SiteSuggestion, type TelecomParams, type Tower } from '@/lib/usecases/telecom';

/** Use-case analyses (radio coverage, flood arrival): seconds of ray casting / flood fill, kept off the UI thread. */
const grids = new Map<string, AnalysisGrid>();

const need = (id: string) => {
  const g = grids.get(id);
  if (!g) throw new Error('Analysis grid not loaded');
  return g;
};

const api = {
  /** Keep the grid worker-side so each analysis only ships its small parameters. Replaces any previous grid. */
  setGrid(id: string, grid: AnalysisGrid): void {
    grids.clear();
    grids.set(id, grid);
  },
  coverage(id: string, towers: Tower[], params: TelecomParams): CoverageResult {
    return computeCoverage(need(id), towers, params);
  },
  suggest(id: string, towers: Tower[], params: TelecomParams, count: number, tower: Pick<Tower, 'heightM' | 'eirpDbm'>): SiteSuggestion[] {
    return suggestSites(need(id), towers, params, count, tower);
  },
  flood(id: string, source: FloodSource): FloodModel | FloodError {
    return buildFloodModel(need(id), source);
  },
};

export type AnalysisWorkerApi = typeof api;
expose(api);
