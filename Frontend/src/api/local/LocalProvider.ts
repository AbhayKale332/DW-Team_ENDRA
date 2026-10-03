import type { BackendStatus, InferenceProvider, ModelInfo, PredictionResult, PredictRequest, ProgressEvent } from '../provider';
import { DepthWizardError, toDepthWizardError } from '../errors';
import type { SceneMeta } from '@/domain/types';
import { parseNpy } from '@/lib/npy';
import { classMapFromPng } from '@/lib/classMap';
import { parseObjects } from '@/lib/objects';
import { parseUncertainty } from '@/lib/uncertainty';

const BASE = './local-api';

async function http(path: string, init?: RequestInit) {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({})) as { detail?: string };
    throw new DepthWizardError('inference', 'Local model unavailable', body.detail ?? `HTTP ${res.status}`);
  }
  return res;
}

function wait(signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, 500);
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
  });
}

/** Desktop-owned, loopback-only FastAPI service. No HF token or network needed. */
export class LocalProvider implements InferenceProvider {
  readonly id = 'depthwizard-serve' as const;
  readonly label = 'Local ONNX model';
  // Cancelling stops waiting; the backend completes its current job.
  readonly capabilities = { absoluteDsm: false, uncertainty: true, serverValidation: false, cancel: false };

  async status(onChange?: (s: BackendStatus) => void): Promise<BackendStatus> {
    let s: BackendStatus;
    try {
      const health = await (await http('/api/health')).json() as { ok: boolean };
      s = health.ok ? { state: 'running', message: 'Local model ready (CPU)' } : { state: 'starting', message: 'Loading local model…' };
    } catch (e) {
      s = { state: 'error', message: e instanceof DepthWizardError ? `${e.message}: ${e.detail ?? ''}` : e instanceof Error ? e.message : 'Local model unavailable' };
    }
    onChange?.(s);
    return s;
  }

  async predict(req: PredictRequest, { onProgress, signal }: { onProgress: (p: ProgressEvent) => void; signal: AbortSignal }): Promise<PredictionResult> {
    try {
      onProgress({ stage: 'uploading', message: 'Sending image to local model' });
      const body = new FormData();
      body.append('file', req.upload, req.uploadName);
      body.append('gsd', String(req.gsd ?? 0));
      body.append('tta', String(req.tta));
      const started = await (await http('/api/predict', { method: 'POST', body, signal })).json() as { job: string };
      if (!/^[a-f0-9]{12}$/.test(started.job)) throw new Error('Invalid local job id');
      let job: { stage: string; progress?: number; error?: string; files?: string[] };
      do {
        job = await (await http(`/api/job/${started.job}`, { signal })).json();
        if (job.stage === 'error') throw new DepthWizardError('inference', 'Local prediction failed', job.error);
        onProgress({ stage: job.stage === 'queued' ? 'queued' : 'processing', progress: job.progress, message: job.stage });
        if (job.stage !== 'done') await wait(signal);
      } while (job.stage !== 'done');

      onProgress({ stage: 'fetching' });
      const files = job.files ?? [];
      const url = (name: string) => `/api/result/${started.job}/${encodeURIComponent(name)}`;
      const optional = async (name: string) => files.includes(name) ? (await http(url(name), { signal })).arrayBuffer() : null;
      const [npy, meta, seg, std, objects] = await Promise.all([
        (await http(url('ndsm_m.npy'), { signal })).arrayBuffer(),
        http(url('meta.json'), { signal }).then((r) => r.json() as Promise<SceneMeta>),
        optional('seg.png'), optional('ndsm_std_m.npy'),
        files.includes('objects.json') ? http(url('objects.json'), { signal }).then((r) => r.json() as Promise<unknown>) : null,
      ]);
      const arr = parseNpy(npy);
      if (arr.shape.length !== 2) throw new Error('Expected a two-dimensional local height grid');
      const [height, width] = arr.shape;
      return {
        heights: { data: arr.data, width, height }, meta,
        classes: classMapFromPng(seg, meta, width, height),
        uncertainty: parseUncertainty(std, width, height),
        objects: parseObjects(objects, { width, height, gsd: meta.scene?.gsd_m }),
        status: { lines: ['Local ONNX prediction (CPU)'], gsd: meta.scene?.gsd_m, gsdSource: meta.scene?.gsd_source, product: meta.product },
        warning: null,
        artefacts: files.filter((name) => !name.startsWith('input')).map((name) => ({ name, url: `${BASE}${url(name)}` })),
      };
    } catch (e) {
      if (signal.aborted) throw new DepthWizardError('cancelled', 'Run cancelled', 'Stopped waiting. The local model finishes its current job in the background.');
      throw toDepthWizardError(e);
    }
  }

  modelInfo(): ModelInfo {
    return { name: 'Local DepthWizard ONNX model', summary: 'Runs the installed model on this computer without an internet connection.', endpoint: 'Local CPU inference', details: [['Install model', 'Desktop menu → Model → Install ONNX model…'], ['Output', 'Height above ground in metres'], ['Runtime', 'ONNX Runtime (CPU)']] };
  }
}
