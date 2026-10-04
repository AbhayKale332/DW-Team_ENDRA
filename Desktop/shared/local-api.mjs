import { request } from 'node:http';

/** Keep backend credentials in the main process and relay only the UI's API. */
export function desktopHandler(getBackend) {
  return (req, res) => {
    const origin = `http://${req.headers.host}`;
    if (!/^127\.0\.0\.1:\d+$/.test(req.headers.host ?? '') ||
        (req.headers.origin && req.headers.origin !== origin) ||
        (req.headers['sec-fetch-site'] && !['same-origin', 'none'].includes(req.headers['sec-fetch-site']))) {
      res.writeHead(403).end('Forbidden');
      return true;
    }
    const url = new URL(req.url, origin);
    if (url.pathname.startsWith('/hf-space') || url.pathname.startsWith('/overpass/')) {
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
