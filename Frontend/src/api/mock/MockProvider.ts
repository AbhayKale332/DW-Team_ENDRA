import type { BackendStatus, InferenceProvider, ModelInfo, PredictionResult, PredictRequest, ProgressEvent } from '../provider';
import { DepthWizardError } from '../errors';
import { parseNpy } from '@/lib/npy';
import { classMapFromPng, fetchOptional } from '@/lib/classMap';
import { fetchObjects } from '@/lib/objects';
import { SAMPLES } from '@/lib/samples';
import type { SceneMeta } from '@/domain/types';

async function fetchOk(url: string, signal: AbortSignal) {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new DepthWizardError('network', 'Sample unavailable', `Could not load ${url} (HTTP ${r.status})`);
  return r;
}

const wait = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => {
      clearTimeout(t);
      reject(new DepthWizardError('cancelled', 'Run cancelled'));
    });
  });

/** Offline provider: walks through the real progress stages and returns the bundled sample result.
 *  Used by the E2E suite and selectable in Settings for demos without network access. */
export class MockProvider implements InferenceProvider {
  readonly id = 'mock' as const;
  readonly label = 'Offline demo (sample result)';
  readonly capabilities = { absoluteDsm: false, uncertainty: false, serverValidation: false, cancel: true };

  constructor(private readonly base = SAMPLES[0].base) {}

  async status(onChange?: (s: BackendStatus) => void): Promise<BackendStatus> {
    const s: BackendStatus = { state: 'running', message: 'Offline demo provider' };
    onChange?.(s);
    return s;
  }

  async predict(req: PredictRequest, { onProgress, signal }: { onProgress: (p: ProgressEvent) => void; signal: AbortSignal }): Promise<PredictionResult> {
    const fail = new URLSearchParams(globalThis.location?.search ?? '').get('mockError');
    onProgress({ stage: 'connecting' });
    await wait(250, signal);
    onProgress({ stage: 'uploading' });
    await wait(300, signal);
    onProgress({ stage: 'queued', position: 1, eta: 2 });
    await wait(400, signal);
    onProgress({ stage: 'processing', eta: 1 });
    await wait(700, signal);
    if (fail === 'quota') throw new DepthWizardError('quota', 'GPU quota used up', 'Your ZeroGPU quota for today is used up.');
    onProgress({ stage: 'fetching' });
    const [buf, meta, segBuf] = await Promise.all([
      fetchOk(`${this.base}/ndsm_m.npy`, signal).then((r) => r.arrayBuffer()),
      fetchOk(`${this.base}/meta.json`, signal).then((r) => r.json() as Promise<SceneMeta>),
      fetchOptional(`${this.base}/seg.png`, { signal }),
    ]);
    const arr = parseNpy(buf);
    const gsd = req.gsd ?? meta.scene?.gsd_m ?? 0.5;
    const grid = { width: arr.shape[1], height: arr.shape[0], gsd };
    const classes = classMapFromPng(segBuf, meta, grid.width, grid.height);
    const objects = await fetchObjects(`${this.base}/objects.json`, grid, { signal });
    return {
      heights: { data: arr.data, width: grid.width, height: grid.height },
      classes,
      objects,
      meta: { ...meta, scene: { ...meta.scene, gsd_m: gsd, gsd_source: req.gsd ? 'user' : 'assumed' } },
      status: { lines: ['Offline demo: returned the bundled sample result.'], gsd, gsdSource: req.gsd ? 'user' : 'assumed' },
      warning: null,
      artefacts: [
        { name: 'ndsm_m.npy', url: `${this.base}/ndsm_m.npy` },
        { name: 'meta.json', url: `${this.base}/meta.json` },
        ...(classes ? [{ name: 'seg.png', url: `${this.base}/seg.png` }] : []),
        ...(objects ? [{ name: 'objects.json', url: `${this.base}/objects.json` }] : []),
      ],
    };
  }

  modelInfo(): ModelInfo {
    return {
      name: 'Offline demo provider',
      summary: 'Returns the bundled sample result instead of calling the model.',
      endpoint: 'local',
      details: [['Output', 'Bundled sample nDSM (metres)']],
    };
  }
}
