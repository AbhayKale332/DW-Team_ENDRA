// Vercel function for /overpass/<mirror> (see vercel.json): the same relay server/serve.mjs and the Vite dev server run.
import { relayOverpass } from '../server/overpass.mjs';

const MAX_BODY = 64 * 1024;

/** Vercel may already have parsed the form post into an object; otherwise read the raw body. */
async function formBody(req) {
  const b = req.body;
  if (typeof b === 'string') return b;
  if (Buffer.isBuffer(b)) return b.toString('utf-8');
  if (b && typeof b === 'object') return new URLSearchParams(b).toString();
  const chunks = [];
  let size = 0;
  for await (const c of req) {
    if ((size += c.length) > MAX_BODY) return null;
    chunks.push(c);
  }
  return Buffer.concat(chunks).toString('utf-8');
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.writeHead(405).end();
  const url = new URL(req.url, 'http://localhost');
  const mirror = url.searchParams.get('mirror') || decodeURIComponent(url.pathname.replace(/^\/(api\/)?overpass\/?/, ''));
  const body = await formBody(req);
  if (body === null) return res.writeHead(413).end();
  const site = req.headers.host ? `https://${req.headers.host}` : undefined;
  const r = await relayOverpass(mirror, body, site);
  res.writeHead(r.status, { 'content-type': r.contentType, 'cache-control': 'no-store' }).end(r.body);
}
