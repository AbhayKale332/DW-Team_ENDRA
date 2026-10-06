import { put } from '@vercel/blob';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.statusCode = 405;
    return res.end();
  }

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
    const page = body?.page ?? 'homepage';
    if (!['homepage', 'judges', 'downloadPpt'].includes(page)) {
      res.statusCode = 400;
      return res.end(JSON.stringify({ error: 'Unknown tracked page' }));
    }
    const timestamp = new Date().toISOString();
    const date = timestamp.slice(0, 10);
    const events = [page, ...(page === 'homepage' && body?.ppt === true ? ['ppt'] : [])];

    await Promise.all(events.map((type) => put(
      `visitor-counts/${date}/${type}/${crypto.randomUUID()}.json`,
      JSON.stringify({ timestamp, type }),
      { access: 'private', addRandomSuffix: false, contentType: 'application/json' },
    )));

    res.statusCode = 204;
    res.end();
  } catch {
    res.statusCode = 503;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Visit could not be recorded' }));
  }
}
