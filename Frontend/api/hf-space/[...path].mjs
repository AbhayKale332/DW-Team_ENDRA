import { request as httpsRequest } from 'node:https';
import { spaceUrlFromId } from '../../server/space.mjs';
import { createRateLimiter, isAllowedSpaceRequest, pickForwardHeaders, MAX_UPLOAD_BYTES } from '../../server/guard.mjs';

const RUNS = Number(process.env.RATE_LIMIT_RUNS || 10);
const WINDOW_MS = Number(process.env.RATE_LIMIT_WINDOW_MIN || 10) * 60_000;
const runLimiter = createRateLimiter({ max: RUNS, windowMs: WINDOW_MS });
const uploadLimiter = createRateLimiter({ max: RUNS * 2, windowMs: WINDOW_MS });

function deny(res, status, message) {
  res.writeHead(status, { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' }).end(message);
}

export default function handler(req, res) {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  const prefix = pathname.startsWith('/api/hf-space/') ? '/api/hf-space' : '/hf-space';
  const path = pathname.slice(prefix.length) || '/';
  if (!isAllowedSpaceRequest(req.method, path)) return deny(res, 404, 'Not found');

  const requestBody = req.body === undefined
    ? null
    : Buffer.isBuffer(req.body)
      ? req.body
      : Buffer.from(typeof req.body === 'string' ? req.body : JSON.stringify(req.body));
  const token = process.env.HF_TOKEN;
  if (!token) return deny(res, 503, 'HF_TOKEN is not configured for this deployment.');

  if (req.method === 'POST') {
    if (Number(req.headers['content-length'] || 0) > MAX_UPLOAD_BYTES || (requestBody?.length ?? 0) > MAX_UPLOAD_BYTES)
      return deny(res, 413, 'Upload too large');
    const forwardedFor = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim();
    const ip = forwardedFor || req.socket.remoteAddress || 'unknown';
    const limiter = path.startsWith('/gradio_api/upload') ? uploadLimiter : runLimiter;
    const wait = limiter.hit(ip);
    if (wait > 0) {
      res.setHeader('retry-after', String(Math.ceil(wait / 1000)));
      return deny(res, 429, 'Too many model runs from this address. Try again later.');
    }
  }

  const spaceUrl = process.env.HF_SPACE_URL || spaceUrlFromId(process.env.VITE_SPACE_ID || 'akashch1512/SingleViewHeigthEstimation');
  const space = new URL(spaceUrl);
  const headers = pickForwardHeaders(req.headers);
  headers.host = space.host;
  headers.authorization = `Bearer ${token}`;
  if (requestBody) {
    headers['content-length'] = String(requestBody.length);
    delete headers['transfer-encoding'];
  }

  const upstream = httpsRequest(
    {
      protocol: space.protocol,
      hostname: space.hostname,
      port: space.port || 443,
      path: `${path}${new URL(req.url, 'http://localhost').search}`,
      method: req.method,
      headers,
    },
    (response) => {
      const responseHeaders = { ...response.headers };
      delete responseHeaders['set-cookie'];
      res.writeHead(response.statusCode || 502, responseHeaders);
      response.pipe(res);
    },
  );
  upstream.on('error', (error) => {
    if (res.writableEnded) return;
    if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain' });
    res.end(`Upstream error: ${error.message}`);
  });

  if (requestBody) {
    upstream.end(requestBody);
  } else if (req.method === 'POST') {
    let size = 0;
    let tooLarge = false;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_UPLOAD_BYTES && !tooLarge) {
        tooLarge = true;
        req.pause();
        req.unpipe(upstream);
        upstream.destroy();
        if (!res.headersSent) deny(res, 413, 'Upload too large');
      }
    });
    req.pipe(upstream);
  } else {
    req.pipe(upstream);
  }
}