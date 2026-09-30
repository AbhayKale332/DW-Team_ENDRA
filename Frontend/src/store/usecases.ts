import { create } from 'zustand';
import type { BuildingRisk, FloodModel } from '@/lib/usecases/flood';
import { DEFAULT_PARAMS, type CoverageResult, type SiteSuggestion, type TelecomParams, type Tower } from '@/lib/usecases/telecom';

export type UseCase = 'telecom' | 'flood';
export type RunStatus = 'idle' | 'running' | 'done' | 'error';
export type FloodSourceKind = 'water' | 'point' | 'bathtub';

interface UseCaseState {
  active: UseCase;
  /** The scenario overlay is showing; results are only drawn on the terrain while it is. */
  open: boolean;

  // ---- telecom
  towers: Tower[];
  params: TelecomParams;
  /** The next click on the terrain places a tower. */
  placing: boolean;
  selectedTower: string | null;
  coverage: CoverageResult | null;
  coverageStatus: RunStatus;
  suggestions: SiteSuggestion[];
  suggesting: boolean;
  /** Draw the coverage layer over the terrain. */
  telecomOverlay: boolean;

  // ---- flood
  floodSource: FloodSourceKind;
  floodPoint: { col: number; row: number } | null;
  /** The next click on the terrain sets the flood source point. */
  pickingSource: boolean;
  flood: FloodModel | null;
  floodStatus: RunStatus;
  floodError: string | null;
  /** Water level above the source's normal level, metres. */
  rise: number;
  playing: boolean;
  /** Bumped to start the simulated water again from rest (replaying the rise from the start). */
  floodEpoch: number;
  floodOverlay: boolean;
  /** Shade dry cells by how soon they would flood. */
  vulnerability: boolean;
  risks: BuildingRisk[];
  /** Building highlighted in the table (index into scene.objects.buildings). */
  focusBuilding: number | null;

  set: (p: Partial<Omit<UseCaseState, 'set' | 'reset'>>) => void;
  reset: () => void;
}

const initial = {
  active: 'telecom' as UseCase,
  open: false,
  towers: [] as Tower[],
  params: DEFAULT_PARAMS,
  placing: false,
  selectedTower: null as string | null,
  coverage: null as CoverageResult | null,
  coverageStatus: 'idle' as RunStatus,
  suggestions: [] as SiteSuggestion[],
  suggesting: false,
  telecomOverlay: true,
  floodSource: 'water' as FloodSourceKind,
  floodPoint: null as { col: number; row: number } | null,
  pickingSource: false,
  flood: null as FloodModel | null,
  floodStatus: 'idle' as RunStatus,
  floodError: null as string | null,
  rise: 0,
  playing: false,
  floodEpoch: 0,
  floodOverlay: true,
  vulnerability: true,
  risks: [] as BuildingRisk[],
  focusBuilding: null as number | null,
};

export const useUseCases = create<UseCaseState>()((set) => ({
  ...initial,
  set: (p) => set(p),
  reset: () => set(initial),
}));

/** What a project keeps of the scenarios: the inputs. Results are recomputed when it is reopened. */
const PERSISTED = ['active', 'open', 'towers', 'params', 'telecomOverlay', 'floodSource', 'floodPoint', 'rise', 'floodOverlay', 'vulnerability'] as const;
export type UseCaseSnapshot = Partial<Pick<UseCaseState, (typeof PERSISTED)[number]>>;

export function snapshotUseCases(): UseCaseSnapshot {
  const s = useUseCases.getState();
  return Object.fromEntries(PERSISTED.map((k) => [k, s[k]])) as UseCaseSnapshot;
}

/** Keep only the known keys of a saved snapshot (it comes from a file). */
export function pickUseCases(saved: Record<string, unknown> | null | undefined): UseCaseSnapshot {
  if (!saved) return {};
  return Object.fromEntries(PERSISTED.filter((k) => saved[k] !== undefined).map((k) => [k, saved[k]])) as UseCaseSnapshot;
}
