// Guards for the /hf-space proxy in server/serve.mjs: which Space requests may pass, which headers are forwarded,
// and a per-client limit on model runs. Kept separate so they can be unit-tested without starting a server.

/** Matches MAX_FILE_BYTES in src/lib/input.ts (the largest image the app accepts), plus multipart overhead. */
export const MAX_UPLOAD_BYTES = 512 * 1024 * 1024 + 1024 * 1024;

/** The Gradio endpoints the app calls (src/api/gradio/GradioSpaceProvider.ts); everything else is refused. */
const ALLOWED = [
  ['GET', /^\/config(\?.*)?$/],
  ['POST', /^\/gradio_api\/upload(\?.*)?$/],
  ['POST', /^\/gradio_api\/call\/predict$/],
  ['GET', /^\/gradio_api\/call\/predict\/[A-Za-z0-9_-]+$/],
  ['GET', /^\/(gradio_api\/)?file=[^?#]+(\?.*)?$/],
];

export function isAllowedSpaceRequest(method, path) {
  const m = method === 'HEAD' ? 'GET' : method;
  if (path.includes('..')) return false;
  return ALLOWED.some(([am, re]) => am === m && re.test(path));
}

const FORWARD = ['accept', 'accept-encoding', 'content-type', 'content-length', 'range', 'last-event-id', 'transfer-encoding'];

/** Only the headers the Space needs; never the app's cookies, origin or client identity. */
export function pickForwardHeaders(headers) {
  const out = {};
  for (const h of FORWARD) if (headers[h] !== undefined) out[h] = headers[h];
  return out;
}

/** Fixed-window counter per key. `hit` returns 0 when allowed, else the milliseconds until the window resets. */
export function createRateLimiter({ max, windowMs, now = () => Date.now() }) {
  const hits = new Map();
  return {
    hit(key) {
      const t = now();
      let e = hits.get(key);
      if (!e || t >= e.reset) {
        e = { count: 0, reset: t + windowMs };
        hits.set(key, e);
      }
      if (hits.size > 10_000) for (const [k, v] of hits) if (t >= v.reset) hits.delete(k);
      if (e.count >= max) return e.reset - t;
      e.count++;
      return 0;
    },
  };
}
