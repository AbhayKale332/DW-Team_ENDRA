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
    const byDay = new Map();

    while (hasMore) {
      const page = await list({ prefix: 'visitor-counts/', cursor, limit: 1000 });
      for (const blob of page.blobs) {
        const [, date, type] = blob.pathname.match(/^visitor-counts\/(\d{4}-\d{2}-\d{2})\/(homepage|ppt)\//) ?? [];
        if (!type) continue;
        if (type === 'homepage') homepage++;
        else ppt++;
        const counts = byDay.get(date) ?? { homepage: 0, ppt: 0 };
        counts[type]++;
        byDay.set(date, counts);
        if (!latest || blob.uploadedAt > latest) latest = blob.uploadedAt;
      }
      hasMore = page.hasMore;
      cursor = page.cursor;
    }

    const today = new Date();
    today.setUTCHours(0, 0, 0, 0);
    const daily = Array.from({ length: 30 }, (_, index) => {
      const date = new Date(today);
      date.setUTCDate(date.getUTCDate() - 29 + index);
      const day = date.toISOString().slice(0, 10);
      return { date: day, ...(byDay.get(day) ?? { homepage: 0, ppt: 0 }) };
    });

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'no-store');
    res.end(JSON.stringify({ homepage, ppt, latest: latest?.toISOString() ?? null, daily }));
  } catch {
    res.statusCode = 503;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Visitor counts are temporarily unavailable' }));
  }
}
