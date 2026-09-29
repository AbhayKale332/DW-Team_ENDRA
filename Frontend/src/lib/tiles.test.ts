import { describe, expect, it } from 'vitest';
import { lonLatToTile, planTiles, tileRange, tileToLonLat, tileZoomFor, type BBox } from './tiles';

describe('XYZ tiles', () => {
  it('converts lon/lat to tile coordinates and back', () => {
    for (const [lon, lat, z] of [
      [88.36, 22.33, 17],
      [-122.42, 37.77, 12],
      [0, 0, 1],
    ]) {
      const [x, y] = lonLatToTile(lon, lat, z);
      const [lon2, lat2] = tileToLonLat(x, y, z);
      expect(lon2).toBeCloseTo(lon, 9);
      expect(lat2).toBeCloseTo(lat, 9);
    }
    // zoom 1: the four quadrants, north-west first
    expect(lonLatToTile(-90, 45, 1).map(Math.floor)).toEqual([0, 0]);
    expect(lonLatToTile(90, -45, 1).map(Math.floor)).toEqual([1, 1]);
  });

  it('covers a box with an inclusive tile range', () => {
    const r = tileRange([22.3, 88.3, 22.4, 88.4], 14);
    const [x0, y0] = lonLatToTile(88.3, 22.4, 14).map(Math.floor);
    const [x1, y1] = lonLatToTile(88.4, 22.3, 14).map(Math.floor);
    expect(r).toEqual({ x0, x1, y0, y1 });
  });

  it('picks a zoom whose pixels are about twice the ground sampling distance', () => {
    // 0.6 m imagery at 22° N: z17 tiles have ~1.1 m pixels
    expect(tileZoomFor(0.6, 22.3, 19)).toBe(17);
    expect(tileZoomFor(0.05, 22.3, 19)).toBe(19); // capped by the provider
    expect(tileZoomFor(30, 22.3, 19)).toBe(11);
  });

  // ~614 m scene at 22.3° N, with 1× and 2× surroundings
  const d = 0.0056;
  const scene: BBox = [22.33, 88.36, 22.33 + d, 88.36 + d];
  const grow = ([s, w, n, e]: BBox, m: number): BBox => [s - m * d, w - m * d, n + m * d, e + m * d];

  it('plans a sharp inner ring and a coarse outer ring that does not repeat it', () => {
    const tiles = planTiles(grow(scene, 1), grow(scene, 2), 17);
    const inner = tiles.filter((t) => t.ring === 'inner');
    const outer = tiles.filter((t) => t.ring === 'outer');
    expect(inner.length).toBeGreaterThan(0);
    expect(outer.length).toBeGreaterThan(0);
    expect(new Set(inner.map((t) => t.z)).size).toBe(1);
    const zi = inner[0].z;
    expect(outer.every((t) => t.z === zi - 2)).toBe(true);
    const ri = tileRange(grow(scene, 1), zi);
    for (const t of outer) {
      const covered = t.x * 4 >= ri.x0 && t.x * 4 + 3 <= ri.x1 && t.y * 4 >= ri.y0 && t.y * 4 + 3 <= ri.y1;
      expect(covered).toBe(false);
    }
    // the coarse ring reaches the edges of the outer box
    const ro = tileRange(grow(scene, 2), zi - 2);
    expect(Math.min(...outer.map((t) => t.x))).toBe(ro.x0);
    expect(Math.max(...outer.map((t) => t.y))).toBe(ro.y1);
  });

  it('drops zoom until the plan fits the tile budget', () => {
    const tiles = planTiles(grow(scene, 1), grow(scene, 2), 19, 40);
    expect(tiles.length).toBeLessThanOrEqual(40);
    expect(tiles.length).toBeGreaterThan(0);
    expect(tiles[0].z).toBeLessThan(19);
  });
});
