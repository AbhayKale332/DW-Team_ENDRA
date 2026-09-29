import { canGeolocate } from '@/lib/osm';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';

/** Is the surrounding basemap drawn? (switched on and the scene can be placed on the globe) Terrain, fog and
 *  the orbit camera adapt to it: no diorama skirt, fog and zoom-out pushed back to the edge of the window. */
export function useBasemapShown(): boolean {
  const on = useView((s) => s.basemap);
  const geo = useScene((s) => canGeolocate(s.scene?.georef));
  return on && geo;
}
