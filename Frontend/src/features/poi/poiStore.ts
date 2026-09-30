import { create } from 'zustand';
import { notifications } from '@mantine/notifications';
import type { Scene } from '@/domain/types';
import { canGeolocate, contextBBox, fetchOverpass, lonLatToGrid } from '@/lib/osm';
import { parsePois, poiQuery, type Poi } from '@/lib/poi';

export type PoiStatus = 'idle' | 'loading' | 'ready' | 'error';

interface PoiState {
  /** Scene the facilities belong to (they are in its grid coordinates). */
  sceneId: string | null;
  status: PoiStatus;
  pois: Poi[];
  error: string | null;
}

export const usePoi = create<PoiState>()(() => ({ sceneId: null, status: 'idle', pois: [], error: null }));

/** Facilities for this scene, or null when not loaded for it. */
export function poisFor(s: PoiState, scene: Scene | null): Poi[] | null {
  return scene && s.sceneId === scene.id && s.status === 'ready' ? s.pois : null;
}

let controller: AbortController | null = null;
/** Overpass answers shipped with a bundled sample (scripts/cache-osm.mjs), so its layer needs no network. */
const bundled = new Map<string, unknown>();

export function registerBundledPois(sceneId: string, json: unknown) {
  bundled.set(sceneId, json);
}

/** Fetch OpenStreetMap facilities for a georeferenced scene and its surroundings once (again after an error).
 *  Sends the surroundings' bounding box to the public Overpass API, only while the layer is switched on. */
export async function loadPois(scene: Scene) {
  const cur = usePoi.getState();
  if (cur.sceneId === scene.id && (cur.status === 'loading' || cur.status === 'ready')) return;
  controller?.abort();
  const fail = (error: string) => usePoi.setState({ sceneId: scene.id, status: 'error', pois: [], error });
  const g = scene.georef;
  if (!canGeolocate(g)) return fail('The image is not georeferenced (or its CRS is unknown).');
  const { width, height } = scene.heights;
  const ctx = contextBBox(g, width, height);
  const toGrid = lonLatToGrid(g);
  if (!ctx || !toGrid) return fail('The scene covers too large an area for the public OpenStreetMap service.');

  const ctl = new AbortController();
  controller = ctl;
  usePoi.setState({ sceneId: scene.id, status: 'loading', pois: [], error: null });
  try {
    const json = bundled.get(scene.id) ?? (await fetchOverpass(poiQuery(ctx.bbox), ctl.signal));
    if (ctl.signal.aborted || usePoi.getState().sceneId !== scene.id) return;
    const pois = parsePois(json, toGrid, scene.gsd);
    usePoi.setState({ status: 'ready', pois, error: null });
    if (!pois.length) notifications.show({ title: 'Facilities', message: 'No mapped hospitals, schools, stations or shelters in this area.', color: 'gray' });
  } catch (e) {
    if (ctl.signal.aborted) return;
    const msg = e instanceof Error ? e.message : String(e);
    fail(msg);
    notifications.show({ title: 'Facilities overlay unavailable', message: msg, color: 'red' });
  }
}
