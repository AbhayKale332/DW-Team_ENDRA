// Vercel Function: the production equivalent of server/serve.mjs's proxy. vercel.json rewrites
// /hf-space/<path> → /api/hf-space?__path=<path>; this forwards it to the private Hugging Face Space
// with HF_TOKEN from the Vercel project's environment. The token never reaches the browser.
import { spaceUrlFromId } from '../server/space.mjs';

const TOKEN = process.env.HF_TOKEN;
const SPACE = new URL(process.env.HF_SPACE_URL || spaceUrlFromId(process.env.VITE_SPACE_ID || 'akashch1512/SingleViewHeigthEstimation'));
const FORWARD_REQ = ['accept', 'content-type', 'range', 'last-event-id'];
// fetch() already decoded the body, so the upstream encoding/length no longer describe it.
const DROP_RES = ['content-encoding', 'content-length', 'transfer-encoding', 'connection', 'set-cookie'];

async function proxy(request) {
  const url = new URL(request.url);
  const path = url.searchParams.get('__path') ?? '';
  url.searchParams.delete('__path');
  const target = new URL(`/${path}${url.search}`, SPACE);

  const headers = new Headers();
  for (const h of FORWARD_REQ) {
    const v = request.headers.get(h);
    if (v) headers.set(h, v);
  }
  if (TOKEN) headers.set('authorization', `Bearer ${TOKEN}`);

  let up;
  try {
    up = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === 'GET' || request.method === 'HEAD' ? undefined : await request.arrayBuffer(),
      redirect: 'manual',
    });
  } catch (e) {
    return new Response(`Upstream error: ${e.message}`, { status: 502, headers: { 'content-type': 'text/plain' } });
  }
  const out = new Headers(up.headers);
  for (const h of DROP_RES) out.delete(h);
  return new Response(up.body, { status: up.status, headers: out }); // streams SSE without buffering
}

export { proxy as GET, proxy as POST, proxy as HEAD };
