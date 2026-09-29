import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { Georef, SceneMeta } from '@/domain/types';
import { SAMPLES } from './samples';
import { parseNpy } from './npy';
import { proj4ForEpsg } from './georef';
import { canGeolocate, CONTEXT_MARGIN, contextBBox, MAX_BBOX_DEG2, sceneBBox } from './osm';

const buf = (p: string) => {
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

describe('bundled sample scenes', () => {
  for (const s of SAMPLES) {
    const dir = s.base.replace(/^\.\//, 'public/');
    it(`${s.id}: height grid, metadata and georeference line up`, () => {
      const meta = JSON.parse(readFileSync(`${dir}/meta.json`, 'utf8')) as SceneMeta & { size_px: [number, number] };
      const arr = parseNpy(buf(`${dir}/ndsm_m.npy`));
      expect(arr.shape).toEqual(meta.size_px);

      // what lib/sceneBuilder.ts derives from meta.scene: without it there is no compass and no OSM overlay
      const t = meta.scene?.transform as Georef['transform'];
      const epsg = meta.scene?.crs_epsg as number;
      const georef: Georef = { epsg, proj4: proj4ForEpsg(epsg), transform: t };
      expect(canGeolocate(georef)).toBe(true);
      const [south, west, north, east] = sceneBBox(georef, arr.shape[1], arr.shape[0])!;
      expect(north).toBeGreaterThan(south);
      expect(east).toBeGreaterThan(west);
      expect((north - south) * (east - west)).toBeLessThan(MAX_BBOX_DEG2);
      // the full surroundings window (basemap + facilities) fits the public Overpass limit too
      expect(contextBBox(georef, arr.shape[1], arr.shape[0])?.margin).toBe(CONTEXT_MARGIN);
    });
  }
});
