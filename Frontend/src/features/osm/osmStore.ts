import { create } from 'zustand';
import { notifications } from '@mantine/notifications';
import type { Scene } from '@/domain/types';
import { canGeolocate, fetchOverpass, lonLatToGrid, MAX_BBOX_DEG2, overpassQuery, parseOverpass, sceneBBox, type OsmFeature } from '@/lib/osm';

export type OsmStatus = 'idle' | 'loading' | 'ready' | 'error';

interface OsmState {
  /** Scene the features belong to (they are in its grid coordinates). */
  sceneId: string | null;
  status: OsmStatus;
  features: OsmFeature[];
  error: string | null;
}

export const useOsm = create<OsmState>()(() => ({ sceneId: null, status: 'idle', features: [], error: null }));

/** OSM features for this scene, or null when not loaded for it. */
export function osmFeaturesFor(s: OsmState, scene: Scene | null): OsmFeature[] | null {
  return scene && s.sceneId === scene.id && s.status === 'ready' ? s.features : null;
}

let controller: AbortController | null = null;
/** Overpass answers shipped with a bundled sample (scripts/cache-osm.mjs), so its overlay needs no network. */
const bundled = new Map<string, unknown>();

export function registerBundledOsm(sceneId: string, json: unknown) {
  bundled.set(sceneId, json);
}

/** Fetch OpenStreetMap data for a georeferenced scene once (again after an error). The request sends the
 *  scene's bounding box to the public Overpass API — only ever called when the user switches the overlay on. */
export async function loadOsm(scene: Scene) {
  const cur = useOsm.getState();
  if (cur.sceneId === scene.id && (cur.status === 'loading' || cur.status === 'ready')) return;
  controller?.abort();
  const fail = (error: string) => useOsm.setState({ sceneId: scene.id, status: 'error', features: [], error });
  const g = scene.georef;
  if (!canGeolocate(g)) return fail('The image is not georeferenced (or its CRS is unknown).');
  const { width, height } = scene.heights;
  const bbox = sceneBBox(g, width, height);
  const toGrid = lonLatToGrid(g);
  if (!bbox || !toGrid) return fail('Could not convert the scene extent to latitude / longitude.');
  const deg2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
  if (!(deg2 > 0) || deg2 > MAX_BBOX_DEG2) return fail('The scene covers too large an area for the public OpenStreetMap service.');

  const ctl = new AbortController();
  controller = ctl;
  useOsm.setState({ sceneId: scene.id, status: 'loading', features: [], error: null });
  try {
    const json = bundled.get(scene.id) ?? (await fetchOverpass(overpassQuery(bbox), ctl.signal));
    if (ctl.signal.aborted || useOsm.getState().sceneId !== scene.id) return;
    const features = parseOverpass(json, toGrid);
    useOsm.setState({ status: 'ready', features, error: null });
    if (!features.length) notifications.show({ title: 'OpenStreetMap', message: 'No mapped buildings, roads or water in this area.', color: 'gray' });
  } catch (e) {
    if (ctl.signal.aborted) return;
    const msg = e instanceof Error ? e.message : String(e);
    fail(msg);
    notifications.show({ title: 'OpenStreetMap overlay unavailable', message: msg, color: 'red' });
  }
}
