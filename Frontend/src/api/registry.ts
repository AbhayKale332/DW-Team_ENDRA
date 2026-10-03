import type { InferenceProvider } from './provider';
import { GradioSpaceProvider } from './gradio/GradioSpaceProvider';
import { MockProvider } from './mock/MockProvider';
import { LocalProvider } from './local/LocalProvider';

export type ProviderId = 'gradio-space' | 'depthwizard-serve' | 'mock';
export const IS_DESKTOP = new URLSearchParams(globalThis.location?.search ?? '').has('desktop');

export const DEFAULT_SPACE_ID = (import.meta.env.VITE_SPACE_ID as string | undefined) || 'akashch1512/SingleViewHeigthEstimation';

export const PROVIDER_OPTIONS: Array<{ value: ProviderId; label: string }> = [
  ...(IS_DESKTOP ? [{ value: 'depthwizard-serve' as const, label: 'Local ONNX model (offline)' }] : [{ value: 'gradio-space' as const, label: 'Hugging Face Space (live model)' }]),
  { value: 'mock', label: 'Offline demo (sample result)' },
];

export interface ProviderConfig {
  provider: ProviderId;
  spaceId: string;
}

let cached: { key: string; provider: InferenceProvider } | null = null;

/** One provider instance per configuration, so the Gradio connection is reused across runs. */
export function getProvider(cfg: ProviderConfig): InferenceProvider {
  const forceMock = new URLSearchParams(globalThis.location?.search ?? '').has('mock');
  const id: ProviderId = forceMock ? 'mock' : cfg.provider;
  const key = `${id}|${cfg.spaceId}`;
  if (cached?.key === key) return cached.provider;
  const provider = id === 'mock' ? new MockProvider() : id === 'depthwizard-serve' ? new LocalProvider() : new GradioSpaceProvider({ spaceId: cfg.spaceId });
  cached = { key, provider };
  return provider;
}
