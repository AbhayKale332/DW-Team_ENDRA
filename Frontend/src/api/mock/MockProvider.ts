import type { BackendStatus, InferenceProvider, ModelInfo, PredictionResult, PredictRequest, ProgressEvent } from '../provider';
import { DepthWizardError } from '../errors';
import { readProject } from '@/lib/dwproj';
import { loadSamples, sampleUrl, type SampleDef } from '@/lib/samples';
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

  /** Default: the first sample project. */
  constructor(private readonly sample?: SampleDef) {}

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
    const sample = this.sample ?? (await loadSamples())[0];
    if (!sample) throw new DepthWizardError('network', 'Sample unavailable', 'There is no sample project (.dwproj) under samples/.');
    const buf: ArrayBuffer = await (await fetchOk(sampleUrl(sample), signal)).arrayBuffer();
    const p = readProject(new Uint8Array(buf));
    const meta: SceneMeta = p.manifest.meta;
    const gsd = req.gsd ?? p.manifest.gsd;
    const { heights, classes } = p;
    const objects = p.objects ? { ...p.objects, gsd } : null;
    const file = (name: string, data: BlobPart | undefined, type: string) => (data ? [{ name, blob: new Blob([data], { type }) }] : []);
    return {
      heights,
      classes,
      objects,
      meta: { ...meta, scene: { ...meta.scene, gsd_m: gsd, gsd_source: req.gsd ? 'user' : 'assumed' } },
      status: { lines: ['Offline demo: returned the bundled sample result.'], gsd, gsdSource: req.gsd ? 'user' : 'assumed' },
      warning: null,
      artefacts: [
        ...file('ndsm_m.npy', p.files['ndsm_m.npy'] as BlobPart, 'application/octet-stream'),
        ...file('meta.json', JSON.stringify(meta), 'application/json'),
        ...file('seg.png', p.files['seg.png'] as BlobPart | undefined, 'image/png'),
        ...file('objects.json', p.files['objects.json'] as BlobPart | undefined, 'application/json'),
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
