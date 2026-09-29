// Vercel Function: vercel.json rewrites /overpass/<mirror> → /api/overpass?__mirror=<mirror>; this relays the
// OpenStreetMap query to that Overpass mirror with a User-Agent that identifies the app (see server/overpass.mjs).
import { relayOverpass } from '../server/overpass.mjs';

export async function POST(request) {
  const url = new URL(request.url);
  const r = await relayOverpass(url.searchParams.get('__mirror') ?? '', await request.text(), url.origin);
  return new Response(r.body, { status: r.status, headers: { 'content-type': r.contentType, 'cache-control': 'no-store' } });
}
