import type { ObjectKinds, Scene } from '@/domain/types';
import { objectSurfaceHeights } from '@/lib/objectSurface';
import { activeObjectKinds, anyObjectKind, objectKindsKey } from '@/store/view';
import { terrainWorker } from '@/workers/clients';

/** The flattened 3D-objects surface (lib/objectSurface), computed in the terrain worker once per scene and
 *  class selection. The terrain mesh, the Objects layer and the GLB export all await the same promise;
 *  walk and flight cameras read it once it is ready (`objectSurfaceIfReady`). */

const pending = new WeakMap<Scene, Map<string, Promise<Float32Array>>>();
const ready = new WeakMap<Scene, Map<string, Float32Array>>();

/** Flattened under the classes in `kinds` that the scene actually has; the raw heights when there are none. */
export function loadObjectSurface(scene: Scene, kinds: ObjectKinds): Promise<Float32Array> {
  const k = activeObjectKinds(kinds, scene.objects);
  if (!scene.objects || !anyObjectKind(k)) return Promise.resolve(scene.heights.data);
  const key = objectKindsKey(k);
  let perScene = pending.get(scene);
  if (!perScene) pending.set(scene, (perScene = new Map()));
  const hit = perScene.get(key);
  if (hit) return hit;
  const p = terrainWorker()
    .objectSurface(
      {
        // copies: the worker gets its own, the scene keeps its arrays
        heights: { ...scene.heights, data: scene.heights.data.slice() },
        classes: scene.classes ? { ...scene.classes, data: scene.classes.data.slice() } : null,
        objects: scene.objects,
        gsd: scene.gsd,
        product: scene.product,
        ndsm: scene.ndsm ? { ...scene.ndsm, data: scene.ndsm.data.slice() } : undefined,
        terrain: scene.terrain ? { ...scene.terrain, data: scene.terrain.data.slice() } : undefined,
      },
      k,
    )
    .catch((e) => {
      // a worker failure costs responsiveness, never the feature
      console.warn('Object surface worker failed; computing on the main thread', e);
      return objectSurfaceHeights(scene, k);
    })
    .then((out) => {
      let done = ready.get(scene);
      if (!done) ready.set(scene, (done = new Map()));
      done.set(key, out);
      return out;
    });
  perScene.set(key, p);
  return p;
}

/** The surface for this selection if the worker has delivered it; the raw heights when no class applies; else null. */
export function objectSurfaceIfReady(scene: Scene, kinds: ObjectKinds): Float32Array | null {
  const k = activeObjectKinds(kinds, scene.objects);
  if (!anyObjectKind(k)) return scene.heights.data;
  return ready.get(scene)?.get(objectKindsKey(k)) ?? null;
}
