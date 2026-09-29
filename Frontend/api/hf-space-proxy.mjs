import proxyHandler from './hf-space/[...path].mjs';

export default function handler(req, res) {
  const url = new URL(req.url, 'http://localhost');
  const proxyPath = url.searchParams.get('proxyPath');
  if (!proxyPath) return res.writeHead(404).end('Not found');

  const query = new URLSearchParams(url.searchParams);
  query.delete('proxyPath');
  const forwardedReq = Object.create(req);
  forwardedReq.url = `/hf-space/${proxyPath}${query.size ? `?${query}` : ''}`;
  return proxyHandler(forwardedReq, res);
}