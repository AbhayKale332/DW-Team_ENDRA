import type { Artefact, ClassMap, GsdSource, HeightGrid, Product, SceneMeta, SceneObjects } from '@/domain/types';

export type BackendState = 'unknown' | 'connecting' | 'running' | 'sleeping' | 'building' | 'starting' | 'error' | 'paused';

export interface BackendStatus {
  state: BackendState;
  message?: string;
}

export type ProgressStage = 'connecting' | 'uploading' | 'queued' | 'processing' | 'fetching' | 'building' | 'done';

export interface ProgressEvent {
  stage: ProgressStage;
  /** Queue position (0-based) when queued. */
  position?: number;
  /** Server-estimated seconds remaining. */
  eta?: number;
  /** 0..1 when known. */
  progress?: number;
  message?: string;
}

export interface PredictRequest {
  /** File actually sent to the model (TIFF inputs are re-encoded to PNG client-side). */
  upload: Blob;
  uploadName: string;
  /** Metres per pixel of `upload`; null = let the model assume its canonical GSD. */
  gsd: number | null;
  tta: boolean;
}

export interface ParsedStatus {
  lines: string[];
  heightLow?: number;
  heightHigh?: number;
  mean?: number;
  median?: number;
  gsd?: number;
  gsdSource?: GsdSource;
  sceneW?: number;
  sceneH?: number;
  pctBelow1m?: number;
  product?: Product;
  artefactCount?: number;
  resampleNote?: string;
}

export interface PredictionResult {
  heights: HeightGrid;
  /** Object classes (`seg.png`); null when the backend does not provide them. */
  classes?: ClassMap | null;
  /** Trees / buildings / water (`objects.json`); null when the backend does not provide them. */
  objects?: SceneObjects | null;
  meta: SceneMeta;
  status: ParsedStatus;
  warning: string | null;
  artefacts: Artefact[];
}

export interface ProviderCapabilities {
  absoluteDsm: boolean;
  uncertainty: boolean;
  serverValidation: boolean;
  cancel: boolean;
}

export interface ModelInfo {
  name: string;
  summary: string;
  endpoint: string;
  details: Array<[string, string]>;
}

export interface InferenceProvider {
  readonly id: 'gradio-space' | 'depthwizard-serve' | 'mock';
  readonly label: string;
  readonly capabilities: ProviderCapabilities;
  status(onChange?: (s: BackendStatus) => void): Promise<BackendStatus>;
  predict(req: PredictRequest, opts: { onProgress: (p: ProgressEvent) => void; signal: AbortSignal }): Promise<PredictionResult>;
  modelInfo(): ModelInfo;
}
