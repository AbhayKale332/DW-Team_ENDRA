import { create } from 'zustand';
import type { ReferenceSurface, Scene, ValidationResult } from '@/domain/types';
import type { PreparedInput } from '@/lib/input';
import type { ProgressEvent, ProgressStage } from '@/api/provider';
import type { DepthWizardError } from '@/api/errors';

export type RunStatus = 'idle' | 'running' | 'error' | 'done';

export interface RunState {
  status: RunStatus;
  stage: ProgressStage | null;
  event: ProgressEvent | null;
  error: DepthWizardError | null;
  startedAt: number | null;
}

export type GsdMode = 'auto' | 'preset' | 'custom';

export interface RunParams {
  gsdMode: GsdMode;
  gsd: number;
  tta: boolean;
}

interface SceneState {
  input: PreparedInput | null;
  params: RunParams;
  scene: Scene | null;
  /** Object URL for scene.image, owned by the store. */
  imageUrl: string | null;
  reference: ReferenceSurface | null;
  validation: ValidationResult | null;
  removeOffset: boolean;
  run: RunState;
  /** Mesh build progress 0..1 while the terrain worker runs; null when idle. */
  meshBuilding: boolean;
  dirty: boolean;
  /** The viewport shows the input image (not the scene) from the start of a run until the user explores the result. */
  inputPreview: boolean;
  dismissed: string[];
  setInput: (input: PreparedInput | null) => void;
  setParams: (p: Partial<RunParams>) => void;
  setScene: (scene: Scene | null) => void;
  /** Swap in a derived version of the current scene (same image and grid): keeps reference, validation and the image URL. */
  updateScene: (scene: Scene) => void;
  setReference: (ref: ReferenceSurface | null) => void;
  setValidation: (v: ValidationResult | null) => void;
  setRun: (p: Partial<RunState>) => void;
  set: (p: Partial<Pick<SceneState, 'removeOffset' | 'meshBuilding' | 'dirty' | 'inputPreview'>>) => void;
  dismiss: (id: string) => void;
  clearAll: () => void;
}

const idleRun: RunState = { status: 'idle', stage: null, event: null, error: null, startedAt: null };

export const useScene = create<SceneState>()((set, get) => ({
  input: null,
  params: { gsdMode: 'auto', gsd: 0.5, tta: false },
  scene: null,
  imageUrl: null,
  reference: null,
  validation: null,
  removeOffset: false,
  run: idleRun,
  meshBuilding: false,
  dirty: false,
  inputPreview: false,
  dismissed: [],
  setInput: (input) => {
    const prev = get().input;
    if (prev && prev !== input) URL.revokeObjectURL(prev.previewUrl);
    set({ input, inputPreview: false });
  },
  setParams: (p) => set({ params: { ...get().params, ...p } }),
  setScene: (scene) => {
    const prevUrl = get().imageUrl;
    if (prevUrl) URL.revokeObjectURL(prevUrl);
    set({
      scene,
      imageUrl: scene ? URL.createObjectURL(scene.image) : null,
      reference: null,
      validation: null,
      dismissed: [],
      dirty: false,
      inputPreview: false,
    });
  },
  updateScene: (scene) => set({ scene }),
  setReference: (reference) => set({ reference, validation: null }),
  setValidation: (validation) => set({ validation }),
  setRun: (p) => set({ run: { ...get().run, ...p } }),
  set: (p) => set(p),
  dismiss: (id) => set({ dismissed: [...get().dismissed, id] }),
  clearAll: () => {
    const { imageUrl, input } = get();
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    if (input) URL.revokeObjectURL(input.previewUrl);
    set({ input: null, scene: null, imageUrl: null, reference: null, validation: null, run: idleRun, dirty: false, inputPreview: false, dismissed: [] });
  },
}));

export const isRunning = () => useScene.getState().run.status === 'running';

/** The viewport shows the input image instead of the scene (see InputPreview). */
export const isPreviewing = (s: Pick<SceneState, 'inputPreview' | 'input' | 'run'>) => s.inputPreview && !!s.input && s.run.status !== 'idle';
