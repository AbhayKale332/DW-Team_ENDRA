// Production server: serves the built app (dist/) and proxies /hf-space/* to the private
// Hugging Face Space, injecting an HF token from the environment (HF_TOKENS / HF_TOKEN — several rotate when one
// runs out of ZeroGPU quota, see server/tokens.mjs). The tokens never reach the browser.
// Only the Gradio endpoints the app uses are proxied, runs are rate-limited per client, and uploads are capped,
// so the proxy cannot be used as an open door to the Space. Also relays /overpass/<mirror> — see server/overpass.mjs.
// No dependencies — Node ≥ 20.   Usage:  node server/serve.mjs   (see .env.example for the variables)
import { createServer } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { readFile, stat } from 'node:fs/promises';
import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { pipeline } from 'node:stream/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { brotliCompressSync, constants as zlib, gzipSync } from 'node:zlib';
import { spaceUrlFromId } from './space.mjs';
import { relayOverpass } from './overpass.mjs';
import { samplesIndex } from './samples.mjs';
import { createRateLimiter, isAllowedSpaceRequest, pickForwardHeaders, MAX_UPLOAD_BYTES } from './guard.mjs';
import { createTokenPool, isWatchedPath, parseTokens } from './tokens.mjs';

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
const tokens = createTokenPool(parseTokens(process.env), { cooldownMs: Number(process.env.HF_TOKEN_COOLDOWN_MIN || 60) * 60_000 });
const SPACE = new URL(process.env.HF_SPACE_URL || spaceUrlFromId(process.env.VITE_SPACE_ID || 'akashch1512/SingleViewHeigthEstimation'));
const DIST = join(root, 'dist');
// Behind a reverse proxy / PaaS router the client address is in X-Forwarded-For; only trust it when told to.
const TRUST_PROXY = /^(1|true|yes)$/i.test(process.env.TRUST_PROXY || '');
// GPU runs (predict calls) per client and window; uploads get twice the allowance so a retry does not trip it.
const RUNS = Number(process.env.RATE_LIMIT_RUNS || 10);
const WINDOW_MS = Number(process.env.RATE_LIMIT_WINDOW_MIN || 10) * 60_000;
const runLimiter = createRateLimiter({ max: RUNS, windowMs: WINDOW_MS });
const uploadLimiter = createRateLimiter({ max: RUNS * 2, windowMs: WINDOW_MS });
if (!tokens.size) console.warn('[depthwizard] HF_TOKEN is not set — the private Space will reject requests.');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.tif': 'image/tiff',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.npy': 'application/octet-stream',
  '.dwproj': 'application/zip',
};
const COMPRESSIBLE = new Set(['.html', '.js', '.css', '.json', '.svg', '.npy']);

// Hosts the browser talks to directly: basemap tiles, DEM tiles and the direct Overpass fallbacks.
const EXTERNAL = 'https://server.arcgisonline.com https://tile.openstreetmap.org https://s3.amazonaws.com https://overpass.kumi.systems https://overpass.private.coffee';
const SECURITY_HEADERS = {
  'x-content-type-options': 'nosniff',
  'x-frame-options': 'DENY',
  // OSM tiles require a Referer; cross-origin requests disclose only the site's origin.
  'referrer-policy': 'strict-origin-when-cross-origin',
  ...(process.env.CSP === 'off'
    ? {}
    : {
        'content-security-policy': [
          "default-src 'self'",
          "script-src 'self'",
          "style-src 'self' 'unsafe-inline'",
          `img-src 'self' data: blob: ${EXTERNAL}`,
          `connect-src 'self' data: blob: ${EXTERNAL}`,
          "worker-src 'self' blob:",
          "font-src 'self' data:",
          "object-src 'none'",
          "base-uri 'self'",
          "frame-ancestors 'none'",
        ].join('; '),
      }),
};

function clientIp(req) {
  const fwd = TRUST_PROXY ? String(req.headers['x-forwarded-for'] || '').split(',')[0].trim() : '';
  return fwd || req.socket.remoteAddress || 'unknown';
}

function deny(res, status, message) {
  res.writeHead(status, { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' }).end(message);
}

function proxy(req, res) {
  const path = req.url.replace(/^\/hf-space/, '') || '/';
  if (!isAllowedSpaceRequest(req.method, path)) return deny(res, 404, 'Not found');

  // A model run is an upload followed by a predict call; each has its own per-client allowance.
  if (req.method === 'POST') {
    if (Number(req.headers['content-length'] || 0) > MAX_UPLOAD_BYTES) return deny(res, 413, 'Upload too large');
    const wait = (path.startsWith('/gradio_api/upload') ? uploadLimiter : runLimiter).hit(clientIp(req));
    if (wait > 0) {
      res.setHeader('retry-after', String(Math.ceil(wait / 1000)));
      return deny(res, 429, 'Too many model runs from this address. Try again later.');
    }
  }

  const headers = pickForwardHeaders(req.headers);
  headers.host = SPACE.host;
  const token = tokens.pick(path);
  if (token) headers.authorization = `Bearer ${token}`;
  if (isWatchedPath(path)) delete headers['accept-encoding'];
  const up = httpsRequest({ protocol: SPACE.protocol, hostname: SPACE.hostname, port: SPACE.port || 443, path, method: req.method, headers }, (r) => {
    tokens.observe(path, token, r);
    const out = { ...r.headers, 'x-dw-spare-tokens': String(tokens.spare(token)) };
    delete out['set-cookie'];
    res.writeHead(r.statusCode || 502, out);
    r.pipe(res); // streams SSE without buffering
  });
  up.on('error', (e) => {
    if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain' });
    res.end(`Upstream error: ${e.message}`);
  });

  // Enforce the cap on the actual bytes too (chunked uploads have no content-length).
  let size = 0;
  req.on('data', (c) => {
    size += c.length;
    if (size > MAX_UPLOAD_BYTES) {
      up.destroy();
      if (!res.headersSent) deny(res, 413, 'Upload too large');
      req.destroy();
    }
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

// dist/ does not change while the server runs, so each compressed variant is built once.
const compressed = new Map();
function encodeFor(file, body, acceptEncoding) {
  if (!COMPRESSIBLE.has(extname(file)) || body.length < 1024) return null;
  const enc = /\bbr\b/.test(acceptEncoding) ? 'br' : /\bgzip\b/.test(acceptEncoding) ? 'gzip' : null;
  if (!enc) return null;
  const key = `${enc}:${file}`;
  if (!compressed.has(key)) {
    const out = enc === 'br' ? brotliCompressSync(body, { params: { [zlib.BROTLI_PARAM_QUALITY]: 9 } }) : gzipSync(body, { level: 9 });
    compressed.set(key, out);
  }
  return { enc, body: compressed.get(key) };
}

async function serveStatic(req, res) {
  const url = new URL(req.url, 'http://x');
  // rescanned per request: a sample folder dropped into dist/samples is listed without a rebuild
  if (url.pathname === '/samples/index.json') {
    res.writeHead(200, { ...SECURITY_HEADERS, 'content-type': TYPES['.json'], 'cache-control': 'no-store' });
    return res.end(req.method === 'HEAD' ? undefined : samplesIndex(join(DIST, 'samples')));
  }
  let file;
  try {
    file = normalize(join(DIST, decodeURIComponent(url.pathname)));
  } catch {
    return deny(res, 400, 'Bad request');
  }
  if (file !== DIST && !file.startsWith(DIST + sep)) return deny(res, 403, 'Forbidden');
  // Source maps are built (hidden) for debugging but not published.
  if (extname(file) === '.map') return deny(res, 404, 'Not found');
  try {
    if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
  } catch {
    file = join(DIST, 'index.html'); // single-page app fallback
  }
  try {
    const hashed = /[.-][A-Za-z0-9_-]{8,}\.(js|css|woff2?)$/.test(file);
    const headers = {
      ...SECURITY_HEADERS,
      'content-type': TYPES[extname(file)] || 'application/octet-stream',
      'cache-control': hashed ? 'public, max-age=31536000, immutable' : 'no-cache',
      vary: 'accept-encoding',
    };
    // Streamed rather than buffered: sample .dwproj files run to hundreds of MB, and the sample loader's progress
    // bar needs the content-length (writeHead + end(buffer) would go out chunked, without one).
    if (!COMPRESSIBLE.has(extname(file))) {
      const { size } = await stat(file);
      res.writeHead(200, { ...headers, 'content-length': size });
      if (req.method === 'HEAD') return res.end();
      return pipeline(createReadStream(file), res).catch(() => {}); // client went away mid-download
    }
    const raw = await readFile(file);
    const packed = encodeFor(file, raw, String(req.headers['accept-encoding'] || ''));
    const body = packed ? packed.body : raw;
    res.writeHead(200, { ...headers, 'content-length': body.length, ...(packed ? { 'content-encoding': packed.enc } : {}) });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch {
    deny(res, 404, 'Not found');
  }
}

createServer((req, res) => {
  if (req.url.startsWith('/hf-space')) return proxy(req, res);
  if (req.url.startsWith('/overpass/') && req.method === 'POST') return overpass(req, res);
  if (req.method !== 'GET' && req.method !== 'HEAD') return res.writeHead(405).end();
  return serveStatic(req, res);
}).listen(PORT, () => console.log(`DepthWizard on http://localhost:${PORT}  →  model ${SPACE.origin}${tokens.size ? ` (${tokens.size} token${tokens.size > 1 ? 's' : ''})` : ' (NO TOKEN)'}`));
