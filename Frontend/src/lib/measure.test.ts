import { describe, expect, it } from 'vitest';
import { cleanPoints, formatArea, formatLength, formatSigned, heightDiff, insidePolygon, measureText, pathLength, polygonArea, surfaceArea, surfaceSegmentLength } from './measure';

const P = (col: number, row: number) => ({ col, row });
const square = [P(0, 0), P(10, 0), P(10, 10), P(0, 10)];
const flat = () => 5;
// a plane rising 1 m per metre eastwards at gsd 0.5 (0.5 m per pixel)
const rampEast = (col: number) => col * 0.5;

describe('measure: lengths', () => {
  it('measures a right triangle as 3-4-5', () => {
    const r = pathLength([P(0, 0), P(3, 0), P(3, 4)], 1);
    expect(r.horizontal).toBeCloseTo(7, 9);
    expect(r.segments).toEqual([3, 4]);
    const ring = pathLength([P(0, 0), P(3, 0), P(3, 4)], 1, undefined, true);
    expect(ring.horizontal).toBeCloseTo(12, 9);
    expect(ring.segments[2]).toBeCloseTo(5, 9);
  });

  it('scales by a non-square GSD per axis', () => {
    const g = { x: 2, y: 0.5 };
    expect(pathLength([P(0, 0), P(3, 0)], g).horizontal).toBeCloseTo(6, 9);
    expect(pathLength([P(0, 0), P(0, 8)], g).horizontal).toBeCloseTo(4, 9);
    expect(pathLength([P(0, 0), P(2, 6)], g).horizontal).toBeCloseTo(5, 9);
  });

  it('follows the terrain for the surface length', () => {
    // flat ground: surface = horizontal
    const lvl = pathLength([P(0, 0), P(40, 0)], 0.5, flat);
    expect(lvl.surface).toBeCloseTo(lvl.horizontal, 9);
    // a 45° ramp: surface = horizontal · √2, wherever the vertices are
    expect(surfaceSegmentLength(P(0, 3), P(40, 3), 0.5, rampEast)).toBeCloseTo(20 * Math.SQRT2, 6);
    // crossing it north-south stays level
    expect(surfaceSegmentLength(P(4, 0), P(4, 40), 0.5, rampEast)).toBeCloseTo(20, 9);
    // a 10 m wall in the middle adds both faces
    const wall = (c: number) => (c >= 10 && c < 20 ? 10 : 0);
    expect(surfaceSegmentLength(P(0, 0), P(30, 0), 1, wall)).toBeCloseTo(30 - 2 + 2 * Math.hypot(1, 10), 6);
  });

  it('handles 0, 1 and repeated points', () => {
    expect(pathLength([], 1)).toEqual({ horizontal: 0, surface: 0, segments: [] });
    expect(pathLength([P(2, 2)], 1, flat)).toEqual({ horizontal: 0, surface: 0, segments: [] });
    expect(pathLength([P(0, 0), P(0, 0), P(4, 0)], 1).segments).toEqual([4]);
    expect(pathLength([P(0, 0), P(4, 0)], 1, undefined, true).horizontal).toBeCloseTo(4, 9);
  });

  it('ignores no-data samples on the surface', () => {
    const holes = (c: number) => (c > 4 && c < 6 ? NaN : 0);
    expect(surfaceSegmentLength(P(0, 0), P(10, 0), 1, holes)).toBeCloseTo(10, 9);
  });
});

describe('measure: areas', () => {
  it('finds the unit square and a square at any GSD', () => {
    expect(polygonArea([P(0, 0), P(1, 0), P(1, 1), P(0, 1)], 1)).toBeCloseTo(1, 12);
    expect(polygonArea(square, 0.5)).toBeCloseTo(25, 9);
    // winding does not matter
    expect(polygonArea([...square].reverse(), 0.5)).toBeCloseTo(25, 9);
  });

  it('finds the right triangle and a non-square pixel', () => {
    expect(polygonArea([P(0, 0), P(3, 0), P(3, 4)], 1)).toBeCloseTo(6, 12);
    expect(polygonArea(square, { x: 2, y: 0.5 })).toBeCloseTo(100, 9);
  });

  it('treats a self-closed ring like an open one, and degenerate rings as empty', () => {
    expect(polygonArea([...square, P(0, 0)], 1)).toBeCloseTo(100, 9);
    expect(cleanPoints([...square, P(0, 0)], true)).toHaveLength(4);
    expect(polygonArea([], 1)).toBe(0);
    expect(polygonArea([P(1, 1)], 1)).toBe(0);
    expect(polygonArea([P(0, 0), P(5, 5)], 1)).toBe(0);
    expect(polygonArea([P(0, 0), P(5, 5), P(0, 0)], 1)).toBe(0);
    // collinear points enclose nothing
    expect(polygonArea([P(0, 0), P(5, 0), P(9, 0)], 1)).toBe(0);
    expect(surfaceArea([P(0, 0), P(5, 5)], 1, flat)).toBe(0);
  });

  it('tests points inside a polygon', () => {
    expect(insidePolygon(square, 5, 5)).toBe(true);
    expect(insidePolygon(square, 11, 5)).toBe(false);
    expect(insidePolygon([P(0, 0), P(10, 0), P(0, 10)], 8, 8)).toBe(false);
  });

  it('measures the surface area of flat and tilted ground', () => {
    expect(surfaceArea(square, 0.5, flat)).toBeCloseTo(25, 9);
    // the 45° ramp: plan area · √2
    expect(surfaceArea(square, 0.5, rampEast)).toBeCloseTo(25 * Math.SQRT2, 6);
    // a large polygon is sub-sampled and still exact on a plane
    const big = [P(0, 0), P(4000, 0), P(4000, 3000), P(0, 3000)];
    expect(surfaceArea(big, 0.5, rampEast, 2000)).toBeCloseTo(polygonArea(big, 0.5) * Math.SQRT2, 3);
  });
});

describe('measure: height difference', () => {
  it('gives Δh, distance and slope between two points', () => {
    const d = heightDiff(P(0, 0), P(4, 0), 10, 13, 1);
    expect(d.dh).toBe(3);
    expect(d.horizontal).toBe(4);
    expect(d.distance3d).toBeCloseTo(5, 12);
    expect(d.slopeDeg).toBeCloseTo((Math.atan2(3, 4) * 180) / Math.PI, 9);
    expect(d.slopePct).toBeCloseTo(75, 9);
    // downhill: negative Δh, same slope
    const down = heightDiff(P(4, 0), P(0, 0), 13, 10, 1);
    expect(down.dh).toBe(-3);
    expect(down.slopePct).toBeCloseTo(75, 9);
  });

  it('uses the per-axis GSD and gives no slope for one spot', () => {
    expect(heightDiff(P(0, 0), P(0, 8), 0, 4, { x: 2, y: 0.5 }).slopeDeg).toBeCloseTo(45, 9);
    const same = heightDiff(P(3, 3), P(3, 3), 2, 7, 1);
    expect(same).toMatchObject({ horizontal: 0, dh: 5, distance3d: 5, slopeDeg: 0, slopePct: 0 });
  });
});

describe('measure: formatting', () => {
  it('switches length units', () => {
    expect(formatLength(0)).toBe('0.00 m');
    expect(formatLength(8.254)).toBe('8.25 m');
    expect(formatLength(412.34)).toBe('412.3 m');
    expect(formatLength(1240)).toBe('1.24 km');
    expect(formatLength(250_000)).toBe('250.0 km');
    expect(formatLength(NaN)).toBe('—');
  });

  it('switches area units at a hectare and a square kilometre', () => {
    expect(formatArea(1)).toBe('1.00 m²');
    expect(formatArea(250)).toBe('250.0 m²');
    expect(formatArea(9999)).toBe('9999 m²');
    expect(formatArea(12_500)).toBe('1.25 ha');
    expect(formatArea(2_500_000)).toBe('2.50 km²');
    expect(formatArea(Infinity)).toBe('—');
  });

  it('signs height differences', () => {
    expect(formatSigned(3.2)).toBe('+3.20 m');
    expect(formatSigned(-3.2)).toBe('−3.20 m');
    expect(formatSigned(-0.001)).toBe('±0.00 m');
  });

  it('writes a clipboard read-out', () => {
    expect(measureText('Length', [['Horizontal', '5.00 m']], ['GSD assumed'])).toBe('Length\nHorizontal: 5.00 m\nNote: GSD assumed');
  });
});
