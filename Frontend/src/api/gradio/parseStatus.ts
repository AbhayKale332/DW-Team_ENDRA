import type { ParsedStatus } from '../provider';
import type { GsdSource, Product } from '@/domain/types';

/** Parse the Space's status markdown (see app.py, success branch):
 *
 *  **Height** {lo} – {hi} m · mean {mean} m · median {median} m
 *  **GSD** {gsd} m/px (effective, {source}) · **scene** {W} × {H} px
 *  **{pct} %** of pixels below 1 m (…)
 *  Product `{rDSM|nDSM}` · {n} artefacts written
 *  [Scene resampled {orig} → {new} px (GSD adjusted to {g} m/px to match).]
 */
export function parseStatus(md: string | null | undefined): ParsedStatus {
  const text = (md ?? '').trim();
  const lines = text
    .split(/\n+/)
    .map((l) => l.trim())
    .filter(Boolean);
  const out: ParsedStatus = { lines };
  const num = (s: string | undefined) => (s === undefined ? undefined : Number(s.replace(/,/g, '')));

  const h = /\*\*Height\*\*\s*(-?[\d.]+)\s*[–-]\s*(-?[\d.]+)\s*m.*?mean\s*(-?[\d.]+)\s*m.*?median\s*(-?[\d.]+)\s*m/i.exec(text);
  if (h) {
    out.heightLow = num(h[1]);
    out.heightHigh = num(h[2]);
    out.mean = num(h[3]);
    out.median = num(h[4]);
  }
  const g = /\*\*GSD\*\*\s*([\d.]+)\s*m\/px\s*\(effective,\s*([a-z]+)\)/i.exec(text);
  if (g) {
    out.gsd = num(g[1]);
    const src = g[2].toLowerCase();
    out.gsdSource = (['user', 'geotiff', 'assumed'].includes(src) ? src : 'assumed') as GsdSource;
  }
  const sc = /\*\*scene\*\*\s*([\d,]+)\s*[×x]\s*([\d,]+)\s*px/i.exec(text);
  if (sc) {
    out.sceneW = num(sc[1]);
    out.sceneH = num(sc[2]);
  }
  const p = /\*\*([\d.]+)\s*%\*\*\s*of pixels below 1 m/i.exec(text);
  if (p) out.pctBelow1m = num(p[1]);
  const prod = /Product\s*`(rDSM|nDSM|DSM)`\s*·\s*(\d+)\s*artefacts/i.exec(text);
  if (prod) {
    out.product = prod[1] as Product;
    out.artefactCount = num(prod[2]);
  }
  const rs = /(Scene resampled[^\n]*)/i.exec(text);
  if (rs) out.resampleNote = rs[1].replace(/\*\*/g, '').trim();
  return out;
}

/** Strip markdown emphasis/emoji for plain display. */
export function plainText(md: string | null | undefined): string {
  return (md ?? '')
    .replace(/\*\*|__|`/g, '')
    .replace(/^#+\s*/gm, '')
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]\u{FE0F}?/gu, '')
    .trim();
}
