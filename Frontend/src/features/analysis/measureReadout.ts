import type { Scene } from '@/domain/types';
import { sampleBilinear } from '@/lib/heights';
import { lonLatAt, formatLonLat } from '@/lib/georef';
import { formatHeight, productInfo } from '@/lib/product';
import { plusMinus, sigmaAt } from '@/lib/uncertainty';
import { formatArea, formatLength, formatSigned, heightDiff, pathLength, polygonArea, surfaceArea, type HeightAt } from '@/lib/measure';
import { MEASURE_MIN_POINTS, type GridPoint, type MeasureMode } from '@/store/tool';

export interface MeasureReadout {
  /** The headline number, e.g. ["Total length", "412.3 m"]; null until there are enough points. */
  primary: [string, string] | null;
  rows: Array<[string, string]>;
  /** Why the numbers are approximate or relative, if they are. */
  notes: string[];
}

export const heightSampler = (scene: Scene): HeightAt => (c, r) => sampleBilinear(scene.heights, c, r);

/** Horizontal values carry "≈" when the ground resolution was assumed rather than given or read from the file. */
export const approx = (scene: Scene) => (scene.gsdSource === 'assumed' ? '≈ ' : '');

const slope = (deg: number, pct: number) => `${deg.toFixed(1)}° · ${pct.toFixed(1)} %`;

/** The measurement of `pts` in `mode` on this scene, formatted for the panel and the clipboard. */
export function measureReadout(scene: Scene, mode: MeasureMode, pts: GridPoint[]): MeasureReadout {
  const notes: string[] = [];
  const ax = approx(scene);
  if (scene.gsdSource === 'assumed') notes.push(`Ground resolution assumed (${scene.gsd.toFixed(2)} m/px): distances and areas are approximate.`);
  const enough = pts.length >= MEASURE_MIN_POINTS[mode];
  const h = heightSampler(scene);

  if (mode === 'length') {
    if (!enough) return { primary: null, rows: [], notes };
    const r = pathLength(pts, scene.gsd, h);
    return {
      primary: ['Total length', ax + formatLength(r.horizontal)],
      rows: [
        ['Horizontal (2D)', ax + formatLength(r.horizontal)],
        ['Along surface (3D)', ax + formatLength(r.surface)],
        ['Segments', String(r.segments.length)],
      ],
      notes,
    };
  }

  if (mode === 'area') {
    if (!enough) return { primary: null, rows: [], notes };
    const plan = polygonArea(pts, scene.gsd);
    const perim = pathLength(pts, scene.gsd, undefined, true).horizontal;
    return {
      primary: ['Area (planimetric)', ax + formatArea(plan)],
      rows: [
        ['Perimeter', ax + formatLength(perim)],
        ['Surface area', ax + formatArea(surfaceArea(pts, scene.gsd, h))],
        ['Corners', String(pts.length)],
      ],
      notes,
    };
  }

  // height: the datum (or "relative") comes from the product, the same wording as the status bar
  if (scene.product === 'rDSM') notes.push('Image not georeferenced: heights are relative to local ground, not elevations.');
  if (!enough) return { primary: null, rows: [], notes };
  const [a, b] = pts;
  const ha = h(a.col, a.row);
  const ref: [string, string] = ['Reference', productInfo(scene).unit];
  if (!b) {
    const rows: Array<[string, string]> = [ref];
    if (scene.product === 'DSM' && scene.ndsm) rows.push(['Above ground', formatLength(sampleBilinear(scene.ndsm, a.col, a.row))]);
    const ll = scene.georef ? lonLatAt(scene.georef, a.col, a.row) : null;
    rows.push(['Pixel', `x ${a.col.toFixed(0)}  y ${a.row.toFixed(0)}`]);
    if (ll) rows.push(['Lon / lat', formatLonLat(ll)]);
    return { primary: ['Height at A', formatHeight(scene, ha) + plusMinus(sigmaAt(scene, a))], rows, notes };
  }
  const d = heightDiff(a, b, ha, h(b.col, b.row), scene.gsd);
  return {
    primary: ['Δh (B − A)', formatSigned(d.dh)],
    rows: [
      ['A', formatHeight(scene, d.a)],
      ['B', formatHeight(scene, d.b)],
      ['Horizontal distance', ax + formatLength(d.horizontal)],
      ['Slope', d.horizontal > 0 ? slope(d.slopeDeg, d.slopePct) : '—'],
      ['3D distance', ax + formatLength(d.distance3d)],
      ref,
    ],
    notes,
  };
}

/** One line on what to do next. */
export function measureHint(mode: MeasureMode, n: number, done: boolean): string {
  if (mode === 'height') {
    if (n === 0) return 'Click a point to read its height';
    if (n === 1) return 'Click a second point for the height difference Δh · Esc to clear';
    return 'Click to start again · Esc to clear';
  }
  if (done) return `${mode === 'area' ? 'Polygon closed' : 'Line finished'} · click to start a new one · Esc to clear`;
  if (mode === 'length') {
    if (n === 0) return 'Click the terrain to start a line';
    return 'Click to add points · double-click or Enter to finish · Backspace to undo · Esc to clear';
  }
  if (n === 0) return 'Click the terrain to place the first corner';
  if (n < 3) return 'Click to add corners (3 or more) · Backspace to undo · Esc to clear';
  return 'Click the first point, double-click or Enter to close · Backspace to undo · Esc to clear';
}
