import { describe, expect, it } from 'vitest';
import { fromArrayBuffer } from 'geotiff';
import type { Scene } from '@/domain/types';
import { exportGeoTiff, layerDescription } from './raster';

const readBlob = (b: Blob) =>
  new Promise<ArrayBuffer>((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result as ArrayBuffer);
    r.onerror = () => rej(r.error);
    r.readAsArrayBuffer(b);
  });

function scene(p: Partial<Scene>): Scene {
  const grid = { data: new Float32Array([1, 2, 3, 4]), width: 2, height: 2 };
  return {
    id: 't',
    name: 't',
    image: new Blob(),
    imageWidth: 2,
    imageHeight: 2,
    heights: grid,
    gsd: 0.6,
    gsdSource: 'user',
    product: 'nDSM',
    stats: { min: 1, max: 4, mean: 2.5, median: 2.5, p2: 1, p98: 4, fracBelow1m: 0, valid: 4 },
    georef: { epsg: 32645, proj4: null, transform: [0.6, 0, 374684, 0, -0.6, 2250121] },
    meta: {},
    artefacts: [],
    warnings: [],
    statusLines: [],
    provenance: { provider: 'test', source: 'sample', createdAt: '' },
    ...p,
  };
}

describe('GeoTIFF export honesty', () => {
  it('never calls a relative or above-ground grid a DSM', () => {
    expect(layerDescription(scene({ product: 'rDSM', georef: null }), 'primary').product).toBe('rDSM');
    expect(layerDescription(scene({ product: 'nDSM' }), 'primary').product).toBe('nDSM');
  });

  it('records the vertical datum of an absolute DSM', async () => {
    const s = scene({
      product: 'DSM',
      anchoring: { source: 'Terrain Tiles', sourceId: 'terrain-tiles', datum: 'EGM96', cellM: 30, cellPx: 50, demMinM: 1, demMaxM: 4, cellMeanRmseM: 0, detailGain: 1, structureShare: 0, meanOffsetM: 0, fetchedAt: '', notes: [] },
    });
    const tiff = await fromArrayBuffer(await readBlob(exportGeoTiff(s)));
    const image = await tiff.getImage();
    const keys = image.getGeoKeys() as Record<string, unknown>;
    expect(keys.ProjectedCSTypeGeoKey).toBe(32645);
    expect(keys.VerticalCSTypeGeoKey).toBe(5773);
    expect(String(keys.GTCitationGeoKey)).toContain('Absolute DSM');
  });
});
