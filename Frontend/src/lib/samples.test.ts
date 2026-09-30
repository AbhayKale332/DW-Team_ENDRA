import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { Georef } from '@/domain/types';
import { listSamples } from '../../server/samples.mjs';
import { readProject } from './dwproj';
import { proj4ForEpsg } from './georef';
import { canGeolocate, CONTEXT_MARGIN, contextBBox, MAX_BBOX_DEG2, sceneBBox } from './osm';

describe('bundled sample scenes', () => {
  for (const s of listSamples('public/samples')) {
    it(`${s.id}: a complete project whose height grid and georeference line up`, () => {
      const p = readProject(new Uint8Array(readFileSync(`public/samples/${s.file}`)));
      const { width: W, height: H } = p.heights;
      if (p.classes) expect([p.classes.width, p.classes.height]).toEqual([W, H]);

      // what lib/sceneBuilder.ts derives when the manifest carries no georef: without it there is no compass and no OSM overlay
      const sc = p.manifest.meta.scene;
      const georef: Georef | null = p.manifest.georef ?? (sc?.transform && sc.crs_epsg ? { epsg: sc.crs_epsg, proj4: proj4ForEpsg(sc.crs_epsg), transform: sc.transform as Georef['transform'] } : null);
      if (!georef) return;
      expect(canGeolocate(georef)).toBe(true);
      const [south, west, north, east] = sceneBBox(georef, W, H)!;
      expect(north).toBeGreaterThan(south);
      expect(east).toBeGreaterThan(west);
      expect((north - south) * (east - west)).toBeLessThan(MAX_BBOX_DEG2);
      // the full surroundings window (basemap + facilities) fits the public Overpass limit too
      expect(contextBBox(georef, W, H)?.margin).toBe(CONTEXT_MARGIN);
    });
  }
});
