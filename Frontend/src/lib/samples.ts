import { create } from 'zustand';

/** A sample scene: a .dwproj in a folder under public/samples, listed by samples/index.json (server/samples.mjs). */
export interface SampleDef {
  id: string;
  name: string;
  /** Path of the .dwproj relative to samples/. */
  file: string;
  /** Input image and height map relative to samples/, shown while the .dwproj downloads (scripts/sample-previews.mjs). */
  image?: string;
  height?: string;
}

/** URL of a file under samples/ (a SampleDef's `file`, `image` or `height`). */
export const sampleUrl = (path: string) => `./samples/${path.split('/').map(encodeURIComponent).join('/')}`;

export const useSamples = create<{ samples: SampleDef[] }>()(() => ({ samples: [] }));

let pending: Promise<SampleDef[]> | null = null;

/** Fetch the sample list (again): folders added since the last call show up. */
export function loadSamples(): Promise<SampleDef[]> {
  pending ??= fetch('./samples/index.json', { cache: 'no-store' })
    .then(async (r) => (r.ok ? ((await r.json()) as { samples?: unknown }) : null))
    .then((j) => (Array.isArray(j?.samples) ? (j.samples as SampleDef[]).filter((s) => s && typeof s.id === 'string' && typeof s.file === 'string') : []))
    .catch(() => useSamples.getState().samples)
    .then((samples) => {
      useSamples.setState({ samples });
      return samples;
    })
    .finally(() => {
      pending = null;
    });
  return pending;
}

/** A sample scene being opened: download progress (total 0 when the server sends no size), unpacking, then the
 *  heavy extras still downloading behind the opened scene. */
export interface SampleLoad {
  name: string;
  stage: 'download' | 'open' | 'extras';
  loaded: number;
  total: number;
  /** Samples with quick looks show them over the viewport while the .dwproj downloads, then wait for the user to
   *  open the 3D view. Image URLs are null while loading, '' when they failed to load. */
  preview?: { image: string | null; height: string | null; ready: boolean };
}

export const useSampleLoad = create<{ load: SampleLoad | null; controller: AbortController | null }>()(() => ({ load: null, controller: null }));

export function cancelSampleLoad() {
  useSampleLoad.getState().controller?.abort();
  useSampleLoad.setState({ load: null, controller: null });
}

/** Download a sample file, handing each chunk to `onChunk` and reporting bytes received to the loading card. */
export async function downloadSample(url: string, name: string, signal: AbortSignal, onChunk: (chunk: Uint8Array) => void): Promise<void> {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`Could not load ${name} (HTTP ${r.status})`);
  // a compressed response reports its encoded length, so only trust the header when nothing re-encoded it
  const total = r.headers.get('content-encoding') ? 0 : Number(r.headers.get('content-length') ?? 0);
  const set = (loaded: number) => useSampleLoad.setState((s) => (s.load ? { load: { ...s.load, loaded, total } } : s));
  if (!r.body) return onChunk(new Uint8Array(await r.arrayBuffer()));
  const reader = r.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    onChunk(value);
    loaded += value.length;
    set(loaded);
  }
}
