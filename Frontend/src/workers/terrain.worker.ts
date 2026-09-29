import { expose, transfer } from 'comlink';
import { buildTerrain, type TerrainBuildInput, type TerrainBuildOutput } from './terrainMesh';
import { computeObjectSurface, type ObjectSurfaceInput } from '@/lib/objectSurface';
import type { ObjectKinds } from '@/domain/types';

const api = {
  build(input: TerrainBuildInput): TerrainBuildOutput {
    const out = buildTerrain(input);
    const bufs = [out.terrain, out.skirt].flatMap((g) => [g.positions.buffer, g.normals.buffer, g.uvs.buffer, g.shade.buffer, g.index.buffer]);
    return transfer(out, bufs as ArrayBuffer[]);
  },
  /** The flattened "3D objects" surface — ~1 s of full-image passes on a large scene, so never on the UI thread. */
  objectSurface(input: ObjectSurfaceInput, kinds: ObjectKinds): Float32Array {
    const out = computeObjectSurface(input, kinds);
    return transfer(out, [out.buffer as ArrayBuffer]);
  },
};

export type TerrainWorkerApi = typeof api;
expose(api);
