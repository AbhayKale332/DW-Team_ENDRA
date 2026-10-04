import { github } from '../src/lib/github-server.mjs';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.statusCode = 405;
    return res.end();
  }

  try {
    const repository = await (await github('')).json();
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'public, max-age=300, s-maxage=3600');
    res.end(JSON.stringify({ stars: repository.stargazers_count }));
  } catch {
    res.statusCode = 503;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'GitHub stars are temporarily unavailable' }));
  }
}
