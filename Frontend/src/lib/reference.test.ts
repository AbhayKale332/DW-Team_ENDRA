import { describe, expect, it } from 'vitest';
import type { Scene } from '@/domain/types';
import { loadReference, referenceKinds } from './reference';
import { writeNpyF32 } from './npy';

describe('reference choices follow the original input', () => {
  const georef: NonNullable<Scene['georef']> = {
    epsg: 32644,
    proj4: null,
    transform: [1, 0, 0, 0, -1, 0],
  };

  it('offers only nDSM for PNG even when the scene has been anchored', () => {
    expect(
      referenceKinds({
        meta: { source_image_name: 'judge.PNG' },
        product: 'DSM',
        georef,
      }),
    ).toEqual(['nDSM']);
  });

  it.each(['judge.tif', 'judge.TIFF', 'judge.GEOTIFF'])('keeps DSM first for %s despite the PNG upload path', (name) => {
    expect(
      referenceKinds({
        meta: { source_image_name: name, scene: { path: '/tmp/upload.png' } },
        product: 'nDSM',
        georef,
      }),
    ).toEqual(['DSM', 'nDSM']);
  });

  it('recognises original TIFF paths in older result bundles', () => {
    expect(
      referenceKinds({
        meta: { scene: { path: '/inputs/judge.tif' } },
        product: 'nDSM',
        georef: null,
      }),
    ).toEqual(['DSM', 'nDSM']);
  });

  it('uses the scene capability when a saved sample has no filename', () => {
    expect(referenceKinds({ meta: {}, product: 'DSM', georef })).toEqual(['DSM', 'nDSM']);
    expect(referenceKinds({ meta: {}, product: 'rDSM', georef: null })).toEqual(['nDSM']);
  });
});

describe('judge reference uploads', () => {
  const scene = { heights: { width: 2, height: 2 }, georef: null } as Scene;

  it('loads a 2D NumPy reference and retains its invalid-pixel mask', async () => {
    const bytes = writeNpyF32(Float32Array.from([0, 3, NaN, 12]), [2, 2]);
    const file = {
      name: 'ground-truth.npy',
      arrayBuffer: async () => bytes.slice().buffer,
    } as File;
    const reference = await loadReference(file, 'nDSM', scene);
    expect(Array.from(reference.data)).toEqual([0, 3, NaN, 12]);
  });

  it('rejects a whole multi-tile shard with a useful 2D-array error', async () => {
    const bytes = writeNpyF32(new Float32Array(8), [2, 2, 2]);
    const file = {
      name: 'shard.npy',
      arrayBuffer: async () => bytes.slice().buffer,
    } as File;
    await expect(loadReference(file, 'nDSM', scene)).rejects.toThrow('2-D array');
  });
});
