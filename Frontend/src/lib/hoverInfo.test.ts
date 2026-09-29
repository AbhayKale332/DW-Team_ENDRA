import { describe, expect, it } from 'vitest';
import type { ClassMap, Scene, SceneObjects } from '@/domain/types';
import type { OsmFeature } from './osm';
import { describeAt, pointInRing } from './hoverInfo';

const W = 60;
const H = 40;
const GSD = 0.5;
const NAMES = ['other', 'ground', 'low_veg', 'building', 'water', 'road', 'tree'];

/** Flat 0.1 m ground; a 12 m block (building) at cols 10–19 rows 10–17; a 7 m block (not an object) at cols 40–47 rows 5–9. */
function scene(withObjects = true, withClasses = true): Scene {
  const data = new Float32Array(W * H).fill(0.1);
  const cls = new Uint8Array(W * H).fill(1);
  for (let r = 10; r < 18; r++)
    for (let c = 10; c < 20; c++) {
      data[r * W + c] = 12;
      cls[r * W + c] = 3;
    }
  for (let r = 5; r < 10; r++) for (let c = 40; c < 48; c++) data[r * W + c] = 7;
  const objects: SceneObjects = {
    version: 1,
    width: W,
    height: H,
    gsd: GSD,
    trees: [{ x: 30.5, y: 30.5, h: 9, r: 2 }],
    buildings: [
      {
        poly: [
          [10, 10],
          [20, 10],
          [20, 18],
          [10, 18],
        ],
        h: 12,
        hMax: 12.4,
        areaM2: 20,
      },
    ],
    water: [
      {
        poly: [
          [50, 30],
          [58, 30],
          [58, 38],
          [50, 38],
        ],
        areaM2: 16,
      },
    ],
    truncated: { trees: false, buildings: false, water: false },
  };
  const classes: ClassMap = { data: cls, width: W, height: H, names: NAMES };
  return {
    heights: { data, width: W, height: H },
    gsd: GSD,
    product: 'rDSM',
    classes: withClasses ? classes : null,
    objects: withObjects ? objects : null,
    georef: null,
  } as unknown as Scene;
}

const row = (info: { rows: Array<[string, string]> }, k: string) => info.rows.find(([key]) => key === k)?.[1];

describe('hover details', () => {
  it('point in polygon', () => {
    const sq: Array<[number, number]> = [
      [0, 0],
      [4, 0],
      [4, 4],
      [0, 4],
    ];
    expect(pointInRing(sq, 2, 2)).toBe(true);
    expect(pointInRing(sq, 5, 2)).toBe(false);
  });

  it('describes a building with its roof height, footprint and floors', () => {
    const info = describeAt(scene(), 14, 13);
    expect(info.kind).toBe('building');
    expect(info.height).toBeCloseTo(12);
    expect(row(info, 'Roof height')).toBe('12.0 m');
    expect(row(info, 'Highest point')).toBe('12.4 m');
    expect(row(info, 'Footprint')).toBe('5 × 4 m · 20 m²');
    expect(info.subtitle).toContain('4 floors');
    expect(row(info, 'Class map')).toBe('Building');
  });

  it('describes a tree from its crown', () => {
    const info = describeAt(scene(), 30.5, 30);
    expect(info.kind).toBe('tree');
    expect(row(info, 'Tree height')).toBe('9.0 m');
    expect(row(info, 'Crown diameter')).toBe('4.0 m');
  });

  it('describes water', () => {
    const info = describeAt(scene(), 54, 34);
    expect(info.kind).toBe('water');
    expect(row(info, 'Surface area')).toBe('16 m²');
  });

  it('describes a raised area the object step did not extract', () => {
    const info = describeAt(scene(), 43, 7);
    expect(info.kind).toBe('elevated');
    expect(row(info, 'Highest point')).toBe('7.0 m');
    expect(row(info, 'Extent')).toBe(`${8 * 5 * GSD * GSD} m²`);
  });

  it('without objects, a building is still an elevated area', () => {
    expect(describeAt(scene(false), 14, 13).kind).toBe('elevated');
  });

  it('describes plain ground by its class, or as ground level', () => {
    expect(describeAt(scene(), 2, 35).title).toBe('Ground');
    expect(describeAt(scene(true, false), 2, 35).title).toBe('Ground level');
  });

  it('adds OpenStreetMap details under the pointer', () => {
    const osm: OsmFeature[] = [
      {
        id: 1,
        kind: 'building',
        name: 'City Library',
        tags: { building: 'library', name: 'City Library', 'building:levels': '3' },
        pts: [
          [9.5, 9.5],
          [19.5, 9.5],
          [19.5, 17.5],
          [9.5, 17.5],
          [9.5, 9.5],
        ],
        closed: true,
      },
      {
        id: 2,
        kind: 'road',
        name: 'MG Road',
        tags: { highway: 'primary', name: 'MG Road' },
        pts: [
          [0, 20],
          [59, 20],
        ],
        closed: false,
      },
    ];
    const b = describeAt(scene(), 14, 13, osm);
    expect(row(b, 'OSM building')).toBe('City Library');
    expect(row(b, 'OSM type')).toBe('library');
    expect(row(b, 'OSM levels')).toBe('3');
    const s = describeAt(scene(), 30, 22, osm); // 2 px = 1 m from the road
    expect(row(s, 'Street')).toBe('MG Road (primary)');
    expect(row(describeAt(scene(), 30, 36, osm), 'Street')).toBeUndefined(); // 8 m away
  });
});
