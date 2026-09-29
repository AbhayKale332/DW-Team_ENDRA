import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchOverpass, OSM_ENDPOINTS } from './osm';

type Reply = { status?: number; body?: unknown; delay?: number; hang?: boolean };

/** fetch stub: each mirror answers per its script (unscripted ones answer 503); hung requests settle only when aborted. */
function stubMirrors(script: Record<string, Reply | Reply[]>) {
  const calls: string[] = [];
  const fetchMock = vi.fn((url: string, init: RequestInit) => {
    calls.push(url);
    const entry = script[url] ?? { status: 503 };
    const reply = Array.isArray(entry) ? (entry.shift() ?? { hang: true }) : entry;
    // a plain object rather than a Response: its body reader needs real event-loop turns the fake clock never gives
    const status = reply.status ?? 200;
    const res = { ok: status < 400, status, json: async () => reply.body ?? {} } as Response;
    return new Promise<Response>((resolve, reject) => {
      init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
      if (reply.hang) return;
      if (reply.delay) setTimeout(() => resolve(res), reply.delay);
      else resolve(res);
    });
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
}

const [A, B, C, D] = OSM_ENDPOINTS.map((e) => e.url);
const OK = { elements: [{ type: 'way', id: 1 }] };

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('fetchOverpass', () => {
  it('uses the first mirror when it answers', async () => {
    const calls = stubMirrors({ [A]: { body: OK }, [B]: { body: { elements: [] } } });
    await expect(fetchOverpass('q1')).resolves.toEqual(OK);
    expect(calls).toEqual([A]);
  });

  it('moves to the next mirror at once on an error status', async () => {
    const calls = stubMirrors({ [A]: { status: 504 }, [B]: { body: OK } });
    await expect(fetchOverpass('q2')).resolves.toEqual(OK);
    expect(calls).toEqual([A, B]);
  });

  it('also asks the next mirror when the first is slow, and takes whichever answers', async () => {
    vi.useFakeTimers();
    const calls = stubMirrors({ [A]: { hang: true }, [B]: { body: OK } });
    const p = fetchOverpass('q3');
    await vi.advanceTimersByTimeAsync(6_000);
    await expect(p).resolves.toEqual(OK);
    expect(calls).toEqual([A, B]);
  });

  it('treats a 200 carrying a server timeout remark as a failure', async () => {
    const calls = stubMirrors({ [A]: { body: { elements: [], remark: 'runtime error: Query timed out in "query"' } }, [B]: { body: OK } });
    await expect(fetchOverpass('q4')).resolves.toEqual(OK);
    expect(calls).toEqual([A, B]);
  });

  it('retries one round after every mirror failed', async () => {
    vi.useFakeTimers();
    const calls = stubMirrors({ [A]: [{ status: 504 }, { body: OK }], [B]: [{ status: 429 }] });
    const p = fetchOverpass('q5');
    await vi.advanceTimersByTimeAsync(2_500);
    await expect(p).resolves.toEqual(OK);
    expect(calls).toEqual([A, B, C, D, A]);
  });

  it('reports the mirrors when the retry fails too', async () => {
    vi.useFakeTimers();
    stubMirrors({ [A]: [{ status: 504 }, { status: 504 }], [B]: [{ status: 429 }, { status: 429 }] });
    const p = fetchOverpass('q6');
    const settled = expect(p).rejects.toThrow(/busy.*HTTP 504.*HTTP 429/);
    await vi.advanceTimersByTimeAsync(2_500);
    await settled;
  });

  it('names each mirror in the error', async () => {
    vi.useFakeTimers();
    stubMirrors({});
    const p = fetchOverpass('q8');
    const settled = expect(p).rejects.toThrow(/overpass-api\.de: HTTP 503.*overpass\.kumi\.systems: HTTP 503/);
    await vi.advanceTimersByTimeAsync(2_500);
    await settled;
  });

  it('stops everything when the caller aborts', async () => {
    stubMirrors({ [A]: { hang: true }, [B]: { hang: true } });
    const ctl = new AbortController();
    const p = fetchOverpass('q7', ctl.signal);
    ctl.abort();
    await expect(p).rejects.toBeDefined();
  });
});
