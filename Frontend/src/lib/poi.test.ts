import { describe, expect, it } from 'vitest';
import { classifyPoi, parsePois, poiQuery, visiblePois, type Poi } from './poi';

// 1 grid pixel per 1e-5° (about 1 m), origin at 77°E 28°N
const toGrid = (lon: number, lat: number): [number, number] => [(lon - 77) / 1e-5, (28 - lat) / 1e-5];

describe('OpenStreetMap facilities', () => {
  it('builds one Overpass query for every facility type, as points', () => {
    const q = poiQuery([27.99, 77, 28, 77.01]);
    expect(q).toContain('[out:json]');
    for (const sel of ['nwr["amenity"~', 'nwr["emergency"~', 'nwr["healthcare"~', 'nwr["social_facility"="shelter"]', 'nwr["railway"~', 'nwr["aeroway"~']) {
      expect(q).toContain(sel);
    }
    expect(q).toContain('(27.9900000,77.0000000,28.0000000,77.0100000);');
    expect(q).toMatch(/hospital\|clinic\|doctors\|fire_station\|police\|school/);
    expect(q).toContain('out center tags;');
  });

  it('classifies tags, most critical role first', () => {
    expect(classifyPoi({ amenity: 'hospital' })).toBe('hospital');
    expect(classifyPoi({ healthcare: 'hospital', building: 'yes' })).toBe('hospital');
    expect(classifyPoi({ amenity: 'fire_station' })).toBe('fire_station');
    expect(classifyPoi({ amenity: 'police' })).toBe('police');
    expect(classifyPoi({ emergency: 'ambulance_station' })).toBe('ambulance_station');
    expect(classifyPoi({ amenity: 'doctors' })).toBe('clinic');
    expect(classifyPoi({ amenity: 'school', emergency: 'assembly_point' })).toBe('school');
    expect(classifyPoi({ emergency: 'assembly_point' })).toBe('shelter');
    expect(classifyPoi({ amenity: 'university' })).toBe('university');
    expect(classifyPoi({ railway: 'halt' })).toBe('rail_station');
    expect(classifyPoi({ aeroway: 'heliport' })).toBe('helipad');
    expect(classifyPoi({ aeroway: 'aerodrome' })).toBe('aerodrome');
    expect(classifyPoi({ amenity: 'bench' })).toBeNull();
    expect(classifyPoi({ railway: 'rail' })).toBeNull();
  });

  it('parses nodes, and ways and relations by their centre', () => {
    const json = {
      elements: [
        { type: 'node', id: 1, lat: 27.999, lon: 77.001, tags: { amenity: 'hospital', name: 'City Hospital' } },
        { type: 'way', id: 2, center: { lat: 27.998, lon: 77.005 }, tags: { amenity: 'school' } },
        { type: 'relation', id: 3, center: { lat: 27.99, lon: 77.009 }, tags: { amenity: 'university', name: 'Tech' } },
        { type: 'way', id: 4, tags: { amenity: 'police' } }, // no centre
        { type: 'node', id: 5, lat: 27.995, lon: 77.002, tags: { amenity: 'bench' } },
      ],
    };
    const pois = parsePois(json, toGrid, 1);
    expect(pois.map((p) => [p.id, p.kind])).toEqual([
      ['node/1', 'hospital'],
      ['way/2', 'school'],
      ['relation/3', 'university'],
    ]);
    expect(pois[0].name).toBe('City Hospital');
    expect(pois[0].col).toBeCloseTo(100, 6);
    expect(pois[0].row).toBeCloseTo(100, 6);
    expect(parsePois(null, toGrid, 1)).toEqual([]);
  });

  it('merges one place mapped twice, keeping the named copy', () => {
    const json = {
      elements: [
        { type: 'node', id: 1, lat: 27.999, lon: 77.001, tags: { railway: 'station' } },
        { type: 'way', id: 2, center: { lat: 27.9991, lon: 77.0011 }, tags: { railway: 'station', name: 'Central' } }, // ~14 px away
        { type: 'node', id: 3, lat: 27.999, lon: 77.0012, tags: { amenity: 'pharmacy' } }, // another kind: kept
        { type: 'node', id: 4, lat: 27.995, lon: 77.001, tags: { railway: 'station' } }, // ~400 px away: kept
      ],
    };
    const pois = parsePois(json, toGrid, 1);
    expect(pois).toHaveLength(3);
    expect(pois[0].name).toBe('Central');
  });

  it('filters by category and keeps emergency services first when capped', () => {
    const p = (kind: Poi['kind'], i: number): Poi => ({ id: String(i), kind, name: null, tags: {}, col: i, row: 0 });
    const pois = [p('school', 1), p('bus_station', 2), p('hospital', 3), p('pharmacy', 4)];
    const all = { emergency: true, education: true, civic: true, transport: true };
    expect(visiblePois(pois, all, 2).map((x) => x.kind)).toEqual(['hospital', 'school']);
    expect(visiblePois(pois, { ...all, education: false, emergency: false }, 10).map((x) => x.kind)).toEqual(['pharmacy', 'bus_station']);
  });
});
