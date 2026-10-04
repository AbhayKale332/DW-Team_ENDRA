import { request } from 'node:http';
import { request as httpsRequest } from 'node:https';

/** Keep backend credentials in the main process and relay only the UI's API. */
export function desktopHandler(getBackend, { getModel = () => null, hosted = null, hostedBase = 'https://depthwizard.teamendra.tech' } = {}) {
  return (req, res) => {
    const origin = `http://${req.headers.host}`;
    if (!/^127\.0\.0\.1:\d+$/.test(req.headers.host ?? '') ||
        (req.headers.origin && req.headers.origin !== origin) ||
        (req.headers['sec-fetch-site'] && !['same-origin', 'none'].includes(req.headers['sec-fetch-site']))) {
      res.writeHead(403).end('Forbidden');
      return true;
    }
    const url = new URL(req.url, origin);
    if (req.method === 'GET' && url.pathname === '/desktop-api/config') {
      res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' }).end(JSON.stringify({ modelKey: getModel() }));
      return true;
    }
    if (url.pathname.startsWith('/hf-space')) {
      const path = req.url.slice('/hf-space'.length);
      if (!hosted || !hosted.isAllowedSpaceRequest(req.method, path) ||
          (req.method === 'POST' && req.headers.origin !== origin)) {
        res.writeHead(404).end();
        return true;
      }
      if (Number(req.headers['content-length'] ?? 0) > hosted.MAX_UPLOAD_BYTES) {
        res.writeHead(413).end('Image is too large');
        return true;
      }
      const destination = new URL(hostedBase);
      const upstream = (destination.protocol === 'https:' ? httpsRequest : request)({
        protocol: destination.protocol, hostname: destination.hostname, port: destination.port,
        path: `/hf-space${path}`, method: req.method, headers: hosted.pickForwardHeaders(req.headers),
      }, (response) => {
        const headers = { 'cache-control': 'no-store' };
        for (const key of ['content-type', 'content-length', 'x-dw-spare-tokens', 'retry-after']) {
          if (response.headers[key]) headers[key] = response.headers[key];
        }
        res.writeHead(response.statusCode ?? 502, headers);
        response.pipe(res);
      });
      upstream.on('error', () => {
        if (res.writableEnded) return;
        if (!res.headersSent) res.writeHead(503, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ detail: 'Hosted inference is unavailable. Check your internet connection or install a local ONNX model.' }));
      });
      let size = 0;
      req.on('data', (chunk) => {
        size += chunk.length;
        if (size > hosted.MAX_UPLOAD_BYTES) {
          res.writeHead(413).end('Image is too large');
          upstream.destroy();
          req.destroy();
        }
      });
      req.on('aborted', () => upstream.destroy());
      res.on('close', () => { if (!res.writableFinished) upstream.destroy(); });
      req.pipe(upstream);
      return true;
    }
    if (url.pathname.startsWith('/overpass/')) {
      res.writeHead(503, { 'content-type': 'application/json' }).end(JSON.stringify({ detail: 'Desktop inference uses the local ONNX model.' }));
      return true;
    }
    if (!url.pathname.startsWith('/local-api/')) return false;
    const path = url.pathname.slice('/local-api'.length);
    const allowed = (req.method === 'GET' && /^\/api\/(health|job\/[a-f0-9]{12}|result\/[a-f0-9]{12}\/[A-Za-z0-9_.%-]+)$/.test(path)) ||
      (req.method === 'POST' && path === '/api/predict' && req.headers.origin === origin);
    if (!allowed) { res.writeHead(404).end(); return true; }
    const backend = getBackend();
    if (!backend.port) {
      res.writeHead(503, { 'content-type': 'application/json', 'cache-control': 'no-store' }).end(JSON.stringify({ detail: backend.message }));
      return true;
    }
    const headers = { authorization: `Bearer ${backend.token}` };
    for (const key of ['content-type', 'content-length']) if (req.headers[key]) headers[key] = req.headers[key];
    const upstream = request({ hostname: '127.0.0.1', port: backend.port, path, method: req.method, headers }, (response) => {
      res.writeHead(response.statusCode ?? 502, { 'content-type': response.headers['content-type'] ?? 'application/octet-stream', 'cache-control': 'no-store', ...(response.headers['content-length'] ? { 'content-length': response.headers['content-length'] } : {}) });
      response.pipe(res);
    });
    upstream.on('error', () => {
      if (!res.headersSent) res.writeHead(503, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ detail: 'Local model stopped. Use Model → Restart inference service.' }));
    });
    req.on('aborted', () => upstream.destroy());
    res.on('close', () => { if (!res.writableFinished) upstream.destroy(); });
    req.pipe(upstream);
    return true;
  };
}
