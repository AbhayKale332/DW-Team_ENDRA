import { afterEach, describe, expect, it, vi } from 'vitest';
import { writeNpyF32 } from '@/lib/npy';
import { GradioSpaceProvider } from './GradioSpaceProvider';

const H = 2;
const W = 3;
const npy = (data: number[], h = H, w = W) => writeNpyF32(Float32Array.from(data), [h, w]);

/** A Space answering one /predict through the proxy, its result files served from `files` (missing = 404). */
function fakeSpace(files: Record<string, Uint8Array | string | null>) {
  const asked: string[] = [];
  const downloads = Object.keys(files).map((name) => ({ path: `/tmp/gradio/ab/${name}`, url: `https://x.hf.space/gradio_api/file=/tmp/gradio/ab/${name}`, orig_name: name }));
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    asked.push(url);
    const json = (v: unknown) => new Response(JSON.stringify(v), { headers: { 'content-type': 'application/json' } });
    if (url.endsWith('/config')) return json({ version: '5' });
    if (url.endsWith('/gradio_api/upload')) return json(['/tmp/gradio/in.png']);
    if (url.endsWith('/gradio_api/call/predict')) return json({ event_id: 'e1' });
    if (url.endsWith('/gradio_api/call/predict/e1')) {
      const data = [null, null, null, downloads, '**Height** 0.00 – 5.00 m', null];
      return new Response(`event: complete\ndata: ${JSON.stringify(data)}\n\n`, { headers: { 'content-type': 'text/event-stream' } });
    }
    const name = url.split('/').pop()!;
    const body = files[name];
    if (body === null || body === undefined) return new Response('gone', { status: 404 });
    return new Response(body as BodyInit);
  });
  vi.stubGlobal('fetch', fetchMock);
  return asked;
}

async function run() {
  const p = new GradioSpaceProvider({ spaceId: 'a/b' });
  return p.predict({ upload: new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }), uploadName: 'in.png', gsd: 0.5, tta: false }, { onProgress: () => {}, signal: new AbortController().signal });
}

afterEach(() => vi.unstubAllGlobals());

describe('GradioSpaceProvider: optional σ (ndsm_std_m.npy)', () => {
  it('fetches and parses σ when the Space publishes it', async () => {
    const asked = fakeSpace({ 'ndsm_m.npy': npy([0, 1, 2, 3, 4, 5]), 'meta.json': '{}', 'ndsm_std_m.npy': npy([0.5, 0.5, 1, 1, 2, 2]) });
    const res = await run();
    expect(asked.some((u) => u.endsWith('ndsm_std_m.npy'))).toBe(true);
    expect(res.heights.width).toBe(W);
    expect(res.uncertainty).toBeTruthy();
    expect(Array.from(res.uncertainty!.data)).toEqual([0.5, 0.5, 1, 1, 2, 2]);
  });

  it('leaves σ out, without asking for it, when the Space does not publish it (the v3 Space)', async () => {
    const asked = fakeSpace({ 'ndsm_m.npy': npy([0, 1, 2, 3, 4, 5]), 'meta.json': '{}' });
    const res = await run();
    expect(asked.some((u) => u.includes('ndsm_std_m'))).toBe(false);
    expect(res.uncertainty).toBeNull();
    expect(Array.from(res.heights.data)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('still loads the result when σ is listed but unavailable or the wrong size', async () => {
    fakeSpace({ 'ndsm_m.npy': npy([0, 1, 2, 3, 4, 5]), 'ndsm_std_m.npy': null });
    expect((await run()).uncertainty).toBeNull();
    fakeSpace({ 'ndsm_m.npy': npy([0, 1, 2, 3, 4, 5]), 'ndsm_std_m.npy': npy([1, 1, 1, 1], 2, 2) });
    const res = await run();
    expect(res.uncertainty).toBeNull();
    expect(res.heights.height).toBe(H);
  });
});
