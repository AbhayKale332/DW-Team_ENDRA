// Overpass (OpenStreetMap) relay shared by server/serve.mjs and the Vite dev server.
// Since 2026 overpass-api.de answers 406 to any browser User-Agent (Mozilla/…) and to any request carrying a
// Referer; a browser can change neither. So the app asks /overpass/<mirror> on its own server, which forwards
// with an app User-Agent and no Referer/Origin/cookies.
// Only the listed mirrors and small form posts are relayed: this is not an open proxy.

export const OVERPASS_MIRRORS = {
  'overpass-api.de': 'https://overpass-api.de/api/interpreter',
  'overpass.private.coffee': 'https://overpass.private.coffee/api/interpreter',
  'overpass.kumi.systems': 'https://overpass.kumi.systems/api/interpreter',
};

const MAX_BODY = 16 * 1024; // the app's query is well under 1 KB
const UPSTREAM_TIMEOUT_MS = 30_000; // the query allows the server 25 s; the browser gives up on a mirror at 30 s

const text = (status, body) => ({ status, contentType: 'text/plain; charset=utf-8', body });

/** Forward one Overpass query. `site` is the app's own origin (e.g. https://depthwizard.app), named in the User-Agent.
 *  Resolves to { status, contentType, body } — never throws. */
export async function relayOverpass(mirror, body, site) {
  const target = OVERPASS_MIRRORS[mirror];
  if (!target) return text(404, `Unknown Overpass mirror: ${mirror}`);
  if (!body || body.length > MAX_BODY || !/(^|&)data=/.test(body)) return text(400, 'Expected a form post with data=<Overpass QL>');
  const agent = process.env.OVERPASS_USER_AGENT || `DepthWizard/1.0 (+${site || 'https://github.com/'})`;
  try {
    const up = await fetch(target, {
      method: 'POST',
      headers: {
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'user-agent': agent, // no Referer: overpass-api.de rejects any request that has one
      },
      body,
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    return { status: up.status, contentType: up.headers.get('content-type') || 'application/json', body: await up.text() };
  } catch (e) {
    return text(e?.name === 'TimeoutError' ? 504 : 502, `Upstream error: ${e?.message ?? e}`);
  }
}
