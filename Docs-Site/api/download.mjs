import { github, githubHeaders, assetNameAllowed } from '../src/lib/github-server.mjs';

export default async function handler(req, res) {
  if (req.method !== 'GET') { res.setHeader('Allow', 'GET'); res.statusCode = 405; return res.end(); }
  const id = new URL(req.url, 'https://depthwizard-docs.vercel.app').searchParams.get('asset');
  if (!id || !/^\d{1,16}$/.test(id)) { res.statusCode = 400; return res.end('Invalid download'); }
  try {
    // Authorize against published release assets, excluding drafts and all source archives.
    const releases = await (await github('releases?per_page=100')).json();
    const asset = releases.filter((release) => !release.draft && release.tag_name.startsWith('desktop-v'))
      .flatMap((release) => release.assets).find((asset) => String(asset.id) === id && asset.state === 'uploaded' && assetNameAllowed(asset.name));
    if (!asset) { res.statusCode = 404; return res.end('Download not available'); }
    const response = await github(`releases/assets/${id}`, { headers: githubHeaders('application/octet-stream'), redirect: 'manual' });
    const location = response.headers.get('location');
    if (response.status !== 302 || !location) throw new Error('No download redirect');
    const target = new URL(location);
    if (target.protocol !== 'https:' || !['release-assets.githubusercontent.com', 'objects.githubusercontent.com'].includes(target.hostname)) throw new Error('Unexpected download host');
    res.statusCode = 302;
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Location', location);
    res.end();
  } catch {
    res.statusCode = 503;
    res.end('Download temporarily unavailable. Please try again.');
  }
}
