import { github, assetNameAllowed } from '../src/lib/github-server.mjs';

export default async function handler(req, res) {
  if (req.method !== 'GET') { res.setHeader('Allow', 'GET'); res.statusCode = 405; return res.end(); }
  try {
    const releases = await (await github('releases?per_page=100')).json();
    const published = releases.filter((release) => !release.draft && release.tag_name.startsWith('desktop-v')).map((release) => ({
      id: release.id, tag_name: release.tag_name, name: release.name, body: release.body,
      published_at: release.published_at, prerelease: release.prerelease, draft: false,
      assets: release.assets.filter((asset) => asset.state === 'uploaded' && assetNameAllowed(asset.name)).map((asset) => ({
        name: asset.name, size: asset.size, state: asset.state,
        browser_download_url: process.env.GITHUB_RELEASES_TOKEN ? `/api/download?asset=${asset.id}` : asset.browser_download_url,
      })),
    }));
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'public, max-age=60, s-maxage=120');
    res.end(JSON.stringify(published));
  } catch {
    res.statusCode = 503;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Downloads are temporarily unavailable. Please try again.' }));
  }
}
