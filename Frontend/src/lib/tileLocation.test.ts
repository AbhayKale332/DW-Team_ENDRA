import { describe, expect, it } from 'vitest';
import type { Scene } from '@/domain/types';
import { proj4ForEpsg } from './georef';
import { parseTileLocation, tileCentre, tileLocationQuery } from './tileLocation';

describe('tile location', () => {
  it('places the pin at the full grid centre and converts projected coordinates', () => {
    // The centre lies on UTM zone 45's central meridian at the equator (500 km easting).
    const tile = { heights: { width: 1024, height: 1024 }, georef: { epsg: 32645, proj4: proj4ForEpsg(32645), transform: [1, 0, 499488, 0, -1, 512] } } as Pick<Scene, 'heights' | 'georef'>;
    const centre = tileCentre(tile)!;
    expect(centre[0]).toBeCloseTo(87, 6);
    expect(centre[1]).toBeCloseTo(0, 6);
    expect(tileLocationQuery(centre)).toContain(`is_in(${centre[1].toFixed(6)},${centre[0].toFixed(6)})`);
    const geographic = { ...tile, georef: { epsg: 4326, proj4: null, transform: [0.01, 0, 70, 0, -0.01, 30] as const } };
    expect(tileCentre(geographic as unknown as Pick<Scene, 'heights' | 'georef'>)).toEqual([75.12, 24.88]);
  });

  it('withholds links when georeferencing is missing, unsupported or invalid', () => {
    const tile = { heights: { width: 2, height: 2 }, georef: null } as Pick<Scene, 'heights' | 'georef'>;
    expect(tileCentre(tile)).toBeNull();
    for (const [epsg, x, y] of [[9999, 70, 30], [4326, 181, 30], [4326, 70, 91], [4326, NaN, 30]]) {
      expect(tileCentre({ ...tile, georef: { epsg, proj4: null, transform: [0, 0, x, 0, 0, y] } })).toBeNull();
    }
  });

  it('summarises enclosing areas from local to country, preferring English and removing duplicates', () => {
    const area = (level: string, name: string, english?: string) => ({ tags: { boundary: 'administrative', admin_level: level, name, ...(english ? { 'name:en': english } : {}) } });
    expect(parseTileLocation({ elements: [area('2', 'भारत', 'India'), area('4', 'Odisha'), area('6', 'Khordha'), area('8', 'ଭୁବନେଶ୍ୱର', 'Bhubaneswar'), area('8', 'Bhubaneswar'), { tags: { name: 'Nearby school' } }] })).toBe('Bhubaneswar, Odisha, India');
    expect(parseTileLocation({ elements: [area('4', 'Singapore'), area('2', 'Singapore')] })).toBe('Singapore');
    expect(parseTileLocation({ elements: [area('6', 'Khordha'), area('2', 'India')] })).toBe('Khordha, India');
    for (const response of [null, {}, { elements: [] }, { elements: [null, {}, { tags: { name: 42 } }] }]) expect(parseTileLocation(response)).toBeNull();
  });
});
