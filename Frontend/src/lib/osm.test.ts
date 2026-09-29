import { describe, expect, it } from 'vitest';
import type { Georef } from '@/domain/types';
import { canGeolocate, classifyOsm, clipSegment, contextBBox, lonLatToGrid, MAX_BBOX_DEG2, osmSegments, overpassQuery, parseOverpass, sceneBBox } from './osm';
import { lonLatAt, proj4ForEpsg } from './georef';

// 1e-5° pixels, top-left corner at 77°E 28°N (north-up)
const WGS84: Georef = { epsg: 4326, proj4: proj4ForEpsg(4326), transform: [1e-5, 0, 77, 0, -1e-5, 28] };
const UTM43: Georef = { epsg: 32643, proj4: proj4ForEpsg(32643), transform: [0.5, 0, 500000, 0, -0.5, 3100000] };

describe('OpenStreetMap geometry', () => {
  it('knows which scenes can be placed on the globe', () => {
    expect(canGeolocate(WGS84)).toBe(true);
    expect(canGeolocate(UTM43)).toBe(true);
    expect(canGeolocate({ epsg: 7755, proj4: null, transform: [1, 0, 0, 0, -1, 0] })).toBe(false);
    expect(canGeolocate(null)).toBe(false);
  });

  it('bounds the grid by its outer edges', () => {
    const [s, w, n, e] = sceneBBox(WGS84, 100, 50)!;
    expect(w).toBeCloseTo(77, 9);
    expect(e).toBeCloseTo(77.001, 9);
    expect(n).toBeCloseTo(28, 9);
    expect(s).toBeCloseTo(27.9995, 9);
  });

  it('grows the box by whole scene sizes for the surroundings', () => {
    const [s, w, n, e] = sceneBBox(WGS84, 100, 50, 2)!;
    expect(w).toBeCloseTo(76.998, 9);
    expect(e).toBeCloseTo(77.003, 9);
    expect(n).toBeCloseTo(28.001, 9);
    expect(s).toBeCloseTo(27.9985, 9);
    expect(contextBBox(WGS84, 100, 50)).toEqual({ bbox: [s, w, n, e], margin: 2 });
  });

  it('shrinks the surroundings to what the public Overpass service accepts', () => {
    // a 0.25 deg² scene is too large even without surroundings
    expect(contextBBox({ ...WGS84, transform: [0.005, 0, 77, 0, -0.005, 28] }, 100, 100)).toBeNull();
    // a 0.02 deg² scene: 2× margin → 0.5 deg², 1× → 0.18, 0.5× → 0.08
    const ctx = contextBBox({ ...WGS84, transform: [0.001414, 0, 77, 0, -0.001414, 28] }, 100, 100)!;
    expect(ctx.margin).toBe(0.5);
    expect((ctx.bbox[2] - ctx.bbox[0]) * (ctx.bbox[3] - ctx.bbox[1])).toBeLessThanOrEqual(MAX_BBOX_DEG2);
  });

  it('maps lon/lat back onto the grid (geographic and projected CRSs)', () => {
    for (const g of [WGS84, UTM43]) {
      const toGrid = lonLatToGrid(g)!;
      const ll = lonLatAt(g, 10, 20)!;
      const [c, r] = toGrid(ll[0], ll[1])!;
      expect(c).toBeCloseTo(10, 4);
      expect(r).toBeCloseTo(20, 4);
    }
  });

  it('builds an Overpass query for the box', () => {
    const q = overpassQuery([27.9995, 77, 28, 77.001]);
    expect(q).toContain('[out:json]');
    expect(q).toContain('way["building"](27.9995000,77.0000000,28.0000000,77.0010000);');
    expect(q).toContain('out geom tags;');
  });

  it('classifies tags', () => {
    expect(classifyOsm({ building: 'yes' })).toBe('building');
    expect(classifyOsm({ highway: 'residential' })).toBe('road');
    expect(classifyOsm({ highway: 'footway' })).toBe('path');
    expect(classifyOsm({ railway: 'rail' })).toBe('rail');
    expect(classifyOsm({ natural: 'water' })).toBe('water');
    expect(classifyOsm({ waterway: 'canal' })).toBe('waterway');
    expect(classifyOsm({ amenity: 'bench' })).toBeNull();
  });

  it('parses Overpass `out geom` output onto the grid', () => {
    const json = {
      elements: [
        {
          type: 'way',
          id: 1,
          tags: { building: 'house', name: 'A' },
          geometry: [
            { lat: 28, lon: 77 },
            { lat: 28, lon: 77.0001 },
            { lat: 27.9999, lon: 77.0001 },
            { lat: 28, lon: 77 },
          ],
        },
        { type: 'way', id: 2, tags: { highway: 'primary' }, geometry: [{ lat: 27.9998, lon: 77 }, null, { lat: 27.9998, lon: 77.001 }] },
        { type: 'way', id: 3, tags: { shop: 'bakery' }, geometry: [{ lat: 28, lon: 77 }, { lat: 28, lon: 77.1 }] },
        { type: 'node', id: 4, lat: 28, lon: 77, tags: { building: 'yes' } },
        { type: 'way', id: 5, tags: { highway: 'service' }, geometry: [{ lat: 28, lon: 77 }] },
      ],
    };
    const f = parseOverpass(json, lonLatToGrid(WGS84)!);
    expect(f.map((x) => [x.id, x.kind, x.closed])).toEqual([
      [1, 'building', true],
      [2, 'road', false],
    ]);
    expect(f[0].name).toBe('A');
    expect(f[0].pts[1][0]).toBeCloseTo(9.5, 6); // 0.0001° = 10 px from the corner → centre index 9.5
    expect(parseOverpass(null, () => [0, 0])).toEqual([]);
  });

  it('clips segments to the grid', () => {
    expect(clipSegment([-5, 5], [15, 5], 0, 0, 10, 10)).toEqual([
      [0, 5],
      [10, 5],
    ]);
    expect(clipSegment([-5, -5], [-1, -1], 0, 0, 10, 10)).toBeNull();
  });

  it('subdivides clipped lines into short segments per kind', () => {
    const segs = osmSegments([{ id: 1, kind: 'road', name: null, tags: {}, pts: [[-10, 5], [20, 5]], closed: false }], 10, 10, 2);
    const road = segs.get('road')!;
    // clipped to [-0.5, 9.5]: 10 px long → 5 segments of 2 px, 4 numbers each
    expect(road.length).toBe(5 * 4);
    expect(road[0]).toBeCloseTo(-0.5);
    expect(road[road.length - 2]).toBeCloseTo(9.5);
  });
});
