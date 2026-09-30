import { describe, expect, it } from 'vitest';
import proj4 from 'proj4';
import type { GroundControlPoint, Scene } from '@/domain/types';
import { applyGcpsToScene, fitGeoref, fitGround } from './gcp';
import { computeStats } from './heights';
import { pixelToMap, proj4ForEpsg } from './georef';

const W = 40;
const H = 30;
const ndsm = { data: new Float32Array(W * H), width: W, height: H };
ndsm.data[10 * W + 10] = 12; // a 12 m building under point 1

const pt = (id: string, col: number, row: number, lat: number | null, lon: number | null, elev: number | null): GroundControlPoint => ({ id, col, row, lat, lon, elev });

const scene = {
  id: 's',
  name: 'plain',
  heights: ndsm,
  gsd: 0.5,
  gsdSource: 'assumed',
  product: 'rDSM',
  stats: computeStats(ndsm.data),
  georef: null,
  objects: null,
  warnings: [],
} as unknown as Scene;

describe('ground control points', () => {
  it('one elevation: level ground, the DSM passes through it', () => {
    const r = applyGcpsToScene(scene, [pt('a', 10, 10, null, null, 112)]);
    expect(r.scene.product).toBe('DSM');
    expect(r.scene.anchoring?.sourceId).toBe('gcp');
    expect(r.scene.heights.data[10 * W + 10]).toBeCloseTo(112, 4);
    expect(r.scene.terrain!.data[0]).toBeCloseTo(100, 4);
    expect(r.scene.georef).toBeNull();
  });

  it('three elevations: a tilted plane through all of them', () => {
    const g = fitGround([pt('a', 0, 0, null, null, 100), pt('b', 20, 0, null, null, 110), pt('c', 0, 20, null, null, 90)], ndsm)!;
    expect(g.kind).toBe('plane');
    expect(g.rmsM).toBeLessThan(1e-6);
    expect(g.plane[0] * 20 + g.plane[1] * 20 + g.plane[2]).toBeCloseTo(100, 6);
  });

  it('three lat/lon points: recovers a north-up 0.6 m/px UTM georeference', () => {
    const def = proj4ForEpsg(32643)!;
    const t: [number, number, number, number, number, number] = [0.6, 0, 500000, 0, -0.6, 2000000];
    const at = (col: number, row: number, id: string) => {
      const [lon, lat] = proj4(def, 'EPSG:4326', pixelToMap(t, col, row));
      return pt(id, col, row, lat, lon, null);
    };
    const fit = fitGeoref([at(2, 3, 'a'), at(35, 4, 'b'), at(20, 27, 'c'), at(5, 25, 'd')])!;
    expect(fit.georef.epsg).toBe(32643);
    expect(fit.gsd).toBeCloseTo(0.6, 4);
    expect(fit.rmsM).toBeLessThan(0.01);
    fit.georef.transform.forEach((v, i) => expect(v).toBeCloseTo(t[i], 2));
  });

  it('lat/lon points on one line are rejected', () => {
    expect(() => fitGeoref([pt('a', 0, 0, 20, 75, null), pt('b', 5, 5, 20.001, 75.001, null), pt('c', 10, 10, 20.002, 75.002, null)])).toThrow(/one line/);
  });
});
