import { list } from '@vercel/blob';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.statusCode = 405;
    return res.end();
  }

  try {
    let cursor;
    let hasMore = true;
    let homepage = 0;
    let ppt = 0;
    let latest = null;

    while (hasMore) {
      const page = await list({ prefix: 'visitor-counts/', cursor, limit: 1000 });
      for (const blob of page.blobs) {
        if (blob.pathname.includes('/homepage/')) homepage++;
        if (blob.pathname.includes('/ppt/')) ppt++;
        if (!latest || blob.uploadedAt > latest) latest = blob.uploadedAt;
      }
      hasMore = page.hasMore;
      cursor = page.cursor;
    }

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'no-store');
    res.end(JSON.stringify({ homepage, ppt, latest: latest?.toISOString() ?? null }));
  } catch {
    res.statusCode = 503;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Visitor counts are temporarily unavailable' }));
  }
}
