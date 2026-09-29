import type { BackendStatus, InferenceProvider, ModelInfo, PredictionResult, PredictRequest, ProgressEvent } from '../provider';
import { classifyStatusText, DepthWizardError, ERROR_COPY, toDepthWizardError } from '../errors';
import { fileBaseName, findArtefact, parsePredictTuple, type GradioFile } from './parseOutputs';
import { parseStatus, plainText } from './parseStatus';
import { parseNpy } from '@/lib/npy';
import { classMapFromPng } from '@/lib/classMap';
import { parseObjects } from '@/lib/objects';
import type { Artefact, SceneMeta } from '@/domain/types';

/** Same-origin path of the authenticating proxy (Vite dev/preview server or server/serve.mjs). */
export const SPACE_PROXY = './hf-space';

export interface GradioSpaceOptions {
  spaceId: string;
  /** Base URL of the proxy that forwards to the private Space with the server-side HF_TOKEN. */
  base?: string;
}

/** Iterate Server-Sent Events from a fetch Response. */
async function* readSse(res: Response, signal: AbortSignal): AsyncGenerator<{ event: string; data: string }> {
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = '';
  try {
    while (!signal.aborted) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.search(/\r?\n\r?\n/)) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx).replace(/^\r?\n\r?\n/, '');
        let event = 'message';
        const data: string[] = [];
        for (const line of block.split(/\r?\n/)) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
        }
        yield { event, data: data.join('\n') };
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Talks to the private DepthWizard Space through the same-origin proxy using Gradio's REST call API.
 *  The browser never sees the access token: the proxy adds `Authorization: Bearer $HF_TOKEN`. */
export class GradioSpaceProvider implements InferenceProvider {
  readonly id = 'gradio-space' as const;
  readonly label = 'DepthWizard model (Hugging Face Space)';
  readonly capabilities = { absoluteDsm: false, uncertainty: false, serverValidation: false, cancel: true };
  private readonly base: string;

  constructor(private readonly opts: GradioSpaceOptions) {
    this.base = (opts.base ?? SPACE_PROXY).replace(/\/$/, '');
  }

  /** Route any Space URL (absolute hf.space URL or server path) back through the proxy. */
  private viaProxy(f: GradioFile): string | undefined {
    if (f.url) {
      try {
        const u = new URL(f.url, 'https://placeholder.invalid');
        return `${this.base}${u.pathname}${u.search}`;
      } catch {
        /* fall through */
      }
    }
    return f.path ? `${this.base}/gradio_api/file=${f.path}` : undefined;
  }

  private async http(path: string, init: RequestInit = {}) {
    let res: Response;
    try {
      res = await fetch(`${this.base}${path}`, init);
    } catch (e) {
      throw toDepthWizardError(e);
    }
    if (res.status === 401 || res.status === 403) throw new DepthWizardError('auth', ERROR_COPY.auth.title, `HTTP ${res.status} from the model Space — check HF_TOKEN in .env.`);
    if (res.status === 404 && path === '/config') throw new DepthWizardError('auth', ERROR_COPY.auth.title, 'The Space was not found — it is private (token missing/invalid) or the Space id is wrong.');
    if (res.status === 502 || res.status === 503 || res.status === 504) throw new DepthWizardError('unavailable', ERROR_COPY.unavailable.title, `HTTP ${res.status} — the Space may be sleeping or restarting.`);
    if (!res.ok) throw new DepthWizardError('network', ERROR_COPY.network.title, `HTTP ${res.status} ${res.statusText}`);
    return res;
  }

  /** The proxy must answer /config with the Space's Gradio JSON — an HTML page means no proxy is running. */
  private async checkConfig(signal?: AbortSignal) {
    const res = await this.http('/config', { signal, cache: 'no-store' });
    const type = res.headers.get('content-type') ?? '';
    if (!type.includes('json'))
      throw new DepthWizardError(
        'unavailable',
        'Model proxy not configured',
        'The app server is not proxying /hf-space to the model. Create .env with HF_TOKEN (see .env.example) and restart `npm run dev` or `npm run serve`.',
      );
    return res.json() as Promise<{ version?: string }>;
  }

  async status(onChange?: (s: BackendStatus) => void): Promise<BackendStatus> {
    onChange?.({ state: 'connecting' });
    let s: BackendStatus;
    try {
      await this.checkConfig();
      s = { state: 'running', message: `${this.opts.spaceId} (private, via server proxy)` };
    } catch (e) {
      const err = toDepthWizardError(e);
      const waking = err.kind === 'unavailable' && !err.message.includes('proxy');
      s = { state: waking ? 'sleeping' : 'error', message: err.detail ?? err.message };
    }
    onChange?.(s);
    return s;
  }

  async predict(req: PredictRequest, { onProgress, signal }: { onProgress: (p: ProgressEvent) => void; signal: AbortSignal }): Promise<PredictionResult> {
    onProgress({ stage: 'connecting' });
    await this.checkConfig(signal);

    onProgress({ stage: 'uploading' });
    const form = new FormData();
    form.append('files', new File([req.upload], req.uploadName, { type: req.upload.type || 'image/png' }));
    const uploaded = (await (await this.http('/gradio_api/upload', { method: 'POST', body: form, signal })).json()) as string[];
    if (!Array.isArray(uploaded) || !uploaded[0]) throw new DepthWizardError('network', 'Upload failed', 'The model Space did not accept the image.');
    const fileData = { path: uploaded[0], orig_name: req.uploadName, size: req.upload.size, mime_type: req.upload.type || 'image/png', meta: { _type: 'gradio.FileData' } };

    onProgress({ stage: 'queued' });
    const call = (await (
      await this.http('/gradio_api/call/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // gsd 0 tells the Space to use the GeoTIFF / canonical resolution.
        body: JSON.stringify({ data: [fileData, req.gsd ?? 0, req.tta] }),
        signal,
      })
    ).json()) as { event_id?: string };
    if (!call.event_id) throw new DepthWizardError('inference', 'The model did not start', 'No event id was returned.');

    const stream = await this.http(`/gradio_api/call/predict/${call.event_id}`, { signal, headers: { Accept: 'text/event-stream' } });
    let data: unknown[] | null = null;
    let sawHeartbeat = false;
    try {
      for await (const ev of readSse(stream, signal)) {
        if (ev.event === 'heartbeat') {
          if (!sawHeartbeat) onProgress({ stage: 'processing' });
          sawHeartbeat = true;
        } else if (ev.event === 'generating') {
          onProgress({ stage: 'processing' });
        } else if (ev.event === 'complete') {
          data = JSON.parse(ev.data) as unknown[];
          break;
        } else if (ev.event === 'error') {
          let msg = 'The model service reported an error.';
          try {
            const parsed = JSON.parse(ev.data);
            if (parsed) msg = typeof parsed === 'string' ? parsed : (parsed.message ?? msg);
          } catch {
            if (ev.data && ev.data !== 'null') msg = ev.data;
          }
          throw toDepthWizardError(new Error(msg));
        }
      }
    } catch (e) {
      if (signal.aborted) throw new DepthWizardError('cancelled', 'Run cancelled');
      throw toDepthWizardError(e);
    }
    if (signal.aborted) throw new DepthWizardError('cancelled', 'Run cancelled');
    if (!data) throw new DepthWizardError('inference', 'No result returned', 'The model service closed the connection without a result.');

    const parsed = parsePredictTuple(data);
    // Rewrite every artefact URL through the proxy (the Space is private).
    const rawFiles = (Array.isArray(data[3]) ? data[3] : data[3] ? [data[3]] : []) as GradioFile[];
    const artefacts: Artefact[] = rawFiles
      .filter((f) => f && (f.url || f.path))
      .map((f) => ({ name: fileBaseName(f), url: this.viaProxy(f), size: f.size ?? undefined }));
    const npy = findArtefact(artefacts, 'ndsm_m.npy');
    const metaFile = findArtefact(artefacts, 'meta.json');
    const segFile = findArtefact(artefacts, 'seg.png');
    const objectsFile = findArtefact(artefacts, 'objects.json');
    if (!npy?.url) throw classifyStatusText(parsed.status);

    onProgress({ stage: 'fetching' });
    const [npyBuf, meta, segBuf, objectsJson] = await Promise.all([
      this.http(npy.url.slice(this.base.length), { signal }).then((r) => r.arrayBuffer()),
      metaFile?.url
        ? this.http(metaFile.url.slice(this.base.length), { signal })
            .then((r) => r.json() as Promise<SceneMeta>)
            .catch(() => ({}) as SceneMeta)
        : Promise.resolve({} as SceneMeta),
      // Optional: older Space builds do not publish a class map.
      segFile?.url
        ? this.http(segFile.url.slice(this.base.length), { signal })
            .then((r) => r.arrayBuffer())
            .catch(() => null)
        : Promise.resolve(null),
      // Optional, likewise: 3D objects extracted from the class + height maps.
      objectsFile?.url
        ? this.http(objectsFile.url.slice(this.base.length), { signal })
            .then((r) => r.json() as Promise<unknown>)
            .catch(() => null)
        : Promise.resolve(null),
    ]);
    const arr = parseNpy(npyBuf);
    const [h, w] = arr.shape;
    return {
      heights: { data: arr.data, width: w, height: h },
      classes: classMapFromPng(segBuf, meta, w, h),
      objects: parseObjects(objectsJson, { width: w, height: h, gsd: meta.scene?.gsd_m }),
      meta,
      status: parseStatus(parsed.status),
      warning: parsed.warning ? plainText(parsed.warning) : null,
      artefacts,
    };
  }

  modelInfo(): ModelInfo {
    return {
      name: 'DepthWizardNet v3',
      summary:
        'Satellite-native monocular height estimation: a DINOv3 ViT-L/16 encoder pretrained on SAT-493M feeds a DPT decoder with a metric regression head and an adaptive-bin classification head, fused per pixel by a learned gate.',
      endpoint: `huggingface.co/spaces/${this.opts.spaceId} · /predict (private, via server proxy)`,
      details: [
        ['Encoder', 'DINOv3 ViT-L/16 · SAT-493M (frozen)'],
        ['Decoder', 'DPT, layers 6 / 12 / 18 / 24'],
        ['Heads', 'A: metric regression · B: 96 adaptive bins (0–120 m) · gated fusion'],
        ['Output', 'Height above ground (nDSM), metres, float32'],
        ['Canonical GSD', '0.5 m/px (inputs are resampled to it)'],
        ['Tiling', '512 px tiles, 50 % overlap, Hann blending, batched in parallel on the GPU'],
        ['TTA', '8 dihedral transforms in one batch (optional)'],
        ['Max scene side', '8192 px (larger scenes are downsampled)'],
        ['Runtime', 'Hugging Face ZeroGPU'],
        ['Access', 'Private Space — authenticated by the server with HF_TOKEN'],
      ],
    };
  }
}
