// Several Hugging Face tokens for the /hf-space proxy (serve.mjs, vite.config.ts, the Vercel function): each
// account has its own daily ZeroGPU allowance, so when one fails the next one takes over. The proxy benches a token
// whose request was refused (401/402/403/429, or a 404 on /config — a private Space hides itself from a bad token)
// or whose predict run reported an error (quota or otherwise), and tells the client (x-dw-spare-tokens) whether
// retrying is worth it.

/** HF_TOKENS (comma / whitespace separated), then HF_TOKEN, HF_TOKEN_2, HF_TOKEN_3 … — duplicates dropped. */
export function parseTokens(env) {
  const list = String(env.HF_TOKENS || '').split(/[\s,;]+/);
  list.push(env.HF_TOKEN);
  const numbered = Object.keys(env)
    .map((k) => /^HF_TOKEN_(\d+)$/.exec(k))
    .filter(Boolean)
    .sort((a, b) => Number(a[1]) - Number(b[1]));
  for (const m of numbered) list.push(env[m[0]]);
  return [...new Set(list.map((t) => String(t ?? '').trim()).filter(Boolean))];
}

const CALL = '/gradio_api/call/predict';
const RESULT = /^\/gradio_api\/call\/predict\/([A-Za-z0-9_-]+)$/;

/** Responses the pool reads; the proxy asks for them uncompressed (both are small). */
export const isWatchedPath = (path) => path === CALL || RESULT.test(path);

/** ZeroGPU: "You have exceeded your GPU quota (120s requested vs. 44s left). Try again in 3:12:05". */
export function quotaRetryMs(text) {
  if (!/quota/i.test(text)) return null;
  const m = /try again in\s+(\d+):(\d{2}):(\d{2})/i.exec(text);
  return m ? ((Number(m[1]) * 60 + Number(m[2])) * 60 + Number(m[3])) * 1000 : 0;
}

/** Upstream statuses that mean this token was refused — another token may get through. */
export function isTokenFailure(status, path) {
  return status === 401 || status === 402 || status === 403 || status === 429 || (status === 404 && path === '/config');
}

// a whole error event (up to its blank line), so a quota message split over chunks is read before deciding
const SSE_ERROR = /(^|\n)event:\s*error\r?\n[\s\S]*?\r?\n\r?\n/;
const short = (t) => `…${t.slice(-4)}`;

/** cooldownMs: quota without a reset time and refused tokens; failCooldownMs: other errors in a predict run. */
export function createTokenPool(tokens, { cooldownMs = 60 * 60_000, failCooldownMs = 5 * 60_000, now = () => Date.now(), log = console.warn } = {}) {
  const benchedUntil = new Map();
  const events = new Map(); // predict event id → the token that started it
  const ready = (t) => (benchedUntil.get(t) ?? 0) <= now();

  function current() {
    const free = tokens.find(ready);
    if (free) return free;
    // all benched: use the one that should recover first (the Space answers with the quota message if not)
    return [...tokens].sort((a, b) => benchedUntil.get(a) - benchedUntil.get(b))[0];
  }

  function bench(token, ms, reason) {
    benchedUntil.set(token, now() + ms);
    const left = tokens.filter(ready).length;
    log(`[depthwizard] HF token ${short(token)} failed (${reason}) — ${left} of ${tokens.length} token(s) left.`);
  }

  return {
    size: tokens.length,
    /** Token for a Space request: a result stream uses the token its run was started with. */
    pick(path) {
      const id = RESULT.exec(path)?.[1];
      return (id && events.get(id)) || current();
    },
    /** Other tokens that still have quota — sent to the client so it only reruns when one exists. */
    spare(token) {
      return tokens.filter((t) => t !== token && ready(t)).length;
    },
    /** Watch an upstream response (without consuming it): remember run → token, bench a token that failed. */
    observe(path, token, res) {
      if (!token) return;
      if (isTokenFailure(res.statusCode, path)) {
        const retryAfter = Number(res.headers?.['retry-after']) * 1000;
        if (ready(token)) bench(token, retryAfter > 0 ? retryAfter : cooldownMs, `HTTP ${res.statusCode}`);
        return;
      }
      if (!isWatchedPath(path)) return;
      const isCall = path === CALL;
      const id = RESULT.exec(path)?.[1];
      let text = '';
      res.on('data', (c) => {
        if (isCall) text = (text + c).slice(0, 4096);
        else {
          text = (text + c).slice(-2048); // enough to span a message split over chunks
          const retry = quotaRetryMs(text);
          const failed = retry !== null || SSE_ERROR.test(text);
          if (failed && ready(token)) bench(token, retry !== null ? retry || cooldownMs : failCooldownMs, retry !== null ? 'ZeroGPU quota' : 'run error');
          if (failed) text = '';
        }
      });
      res.on('end', () => {
        if (id) return void events.delete(id);
        const eventId = /"event_id"\s*:\s*"([^"]+)"/.exec(text)?.[1];
        if (!eventId) return;
        if (events.size > 1000) events.delete(events.keys().next().value);
        events.set(eventId, token);
      });
    },
  };
}
