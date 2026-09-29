// Production server: serves the built app (dist/) and proxies /hf-space/* to the private
// Hugging Face Space, injecting HF_TOKEN from the environment. The token never reaches the browser.
// Also relays /overpass/<mirror> (OpenStreetMap overlay) — see server/overpass.mjs.
// No dependencies — Node ≥ 20.   Usage:  node server/serve.mjs   (PORT, HF_TOKEN, HF_SPACE_URL / VITE_SPACE_ID)
import { createServer } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { readFile, stat } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spaceUrlFromId } from './space.mjs';
import { relayOverpass } from './overpass.mjs';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
// Minimal .env loader (KEY=VALUE lines) so the same .env works in dev and production.
const envFile = join(root, '.env');
if (existsSync(envFile)) {
  for (const line of readFileSync(envFile, 'utf-8').split(/\r?\n/)) {
    const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/i.exec(line);
    if (m && !line.trim().startsWith('#') && process.env[m[1]] === undefined) process.env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
  }
}

const PORT = Number(process.env.PORT || 8080);
const TOKEN = process.env.HF_TOKEN;
const SPACE = new URL(process.env.HF_SPACE_URL || spaceUrlFromId(process.env.VITE_SPACE_ID || 'akashch1512/SingleViewHeigthEstimation'));
const DIST = join(root, 'dist');
if (!TOKEN) console.warn('[depthwizard] HF_TOKEN is not set — the private Space will reject requests.');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.npy': 'application/octet-stream',
  '.map': 'application/json',
};

function proxy(req, res) {
  const path = req.url.replace(/^\/hf-space/, '') || '/';
  const headers = { ...req.headers, host: SPACE.host };
  delete headers.cookie; // never forward app cookies to a third party
  delete headers.origin;
  delete headers.referer;
  if (TOKEN) headers.authorization = `Bearer ${TOKEN}`;
  const up = httpsRequest({ protocol: SPACE.protocol, hostname: SPACE.hostname, port: SPACE.port || 443, path, method: req.method, headers }, (r) => {
    const out = { ...r.headers };
    delete out['set-cookie'];
    res.writeHead(r.statusCode || 502, out);
    r.pipe(res); // streams SSE without buffering
  });
  up.on('error', (e) => {
    if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain' });
    res.end(`Upstream error: ${e.message}`);
  });
  req.pipe(up);
}

async function overpass(req, res) {
  const chunks = [];
  let size = 0;
  for await (const c of req) {
    if ((size += c.length) > 64 * 1024) return res.writeHead(413).end();
    chunks.push(c);
  }
  const mirror = decodeURIComponent(new URL(req.url, 'http://x').pathname.replace(/^\/overpass\//, ''));
  const site = req.headers.host ? `${req.headers['x-forwarded-proto'] || 'http'}://${req.headers.host}` : undefined;
  const r = await relayOverpass(mirror, Buffer.concat(chunks).toString('utf-8'), site);
  res.writeHead(r.status, { 'content-type': r.contentType, 'cache-control': 'no-store' }).end(r.body);
}

async function serveStatic(req, res) {
  const url = new URL(req.url, 'http://x');
  let file = normalize(join(DIST, decodeURIComponent(url.pathname)));
  if (!file.startsWith(DIST)) {
    res.writeHead(403).end();
    return;
  }
  try {
    if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
  } catch {
    file = join(DIST, 'index.html'); // single-page app fallback
  }
  try {
    const body = await readFile(file);
    const hashed = /[.-][A-Za-z0-9_-]{8,}\.(js|css|woff2?)$/.test(file);
    res.writeHead(200, {
      'content-type': TYPES[extname(file)] || 'application/octet-stream',
      'cache-control': hashed ? 'public, max-age=31536000, immutable' : 'no-cache',
      'x-content-type-options': 'nosniff',
    });
    res.end(body);
  } catch {
    res.writeHead(404).end('Not found');
  }
}

createServer((req, res) => {
  if (req.url.startsWith('/hf-space')) return proxy(req, res);
  if (req.url.startsWith('/overpass/') && req.method === 'POST') return overpass(req, res);
  if (req.method !== 'GET' && req.method !== 'HEAD') return res.writeHead(405).end();
  return serveStatic(req, res);
}).listen(PORT, () => console.log(`DepthWizard on http://localhost:${PORT}  →  model ${SPACE.origin}${TOKEN ? ' (token set)' : ' (NO TOKEN)'}`));
