import { useMemo } from 'react';
import { Loader } from '@mantine/core';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { useSettings } from '@/store/settings';
import { canGeolocate } from '@/lib/osm';
import { BASEMAPS } from '@/lib/tiles';
import { POI_CATEGORIES, POI_CATEGORY_COLORS, POI_CATEGORY_LABELS, POI_KIND_META, type PoiCategory } from '@/lib/poi';
import { POI_CATEGORY_ICONS } from '@/features/poi/poiIcons';
import { usePoi } from '@/features/poi/poiStore';
import classes from '@/features/viewport/overlays/overlays.module.css';

/** Facility counts per category, loading / error state, and the attributions the basemap tiles and the
 *  OpenStreetMap data require while they are shown. */
export function SurroundingsLegend() {
  const scene = useScene((s) => s.scene);
  const basemap = useView((s) => s.basemap);
  const poiOn = useView((s) => s.poi);
  const categories = useView((s) => s.poiCategories);
  const provider = useSettings((s) => s.basemapProvider);
  const poi = usePoi();
  const mine = !!scene && poi.sceneId === scene.id;
  const counts = useMemo(() => {
    const m = new Map<PoiCategory, number>();
    if (mine && poi.status === 'ready') for (const p of poi.pois) m.set(POI_KIND_META[p.kind].category, (m.get(POI_KIND_META[p.kind].category) ?? 0) + 1);
    return m;
  }, [mine, poi.status, poi.pois]);

  if (!scene || !canGeolocate(scene.georef) || (!basemap && !poiOn)) return null;
  const tiles = BASEMAPS[provider];
  return (
    <div className={`dw-float ${classes.classLegend}`} role="group" aria-label="Surroundings">
      {poiOn && (
        <>
          <div className={classes.classLegendTitle}>Facilities</div>
          {mine && poi.status === 'loading' && (
            <div className={classes.classLegendRow}>
              <Loader size={10} />
              <span>Loading facilities…</span>
            </div>
          )}
          {mine && poi.status === 'error' && <div style={{ maxWidth: 220 }}>Unavailable: {poi.error}</div>}
          {POI_CATEGORIES.filter((c) => categories[c] && counts.get(c)).map((c) => {
            const Icon = POI_CATEGORY_ICONS[c];
            return (
              <div key={c} className={classes.classLegendRow}>
                <Icon size={10} stroke={2.2} color={POI_CATEGORY_COLORS[c]} aria-hidden />
                <span>{POI_CATEGORY_LABELS[c]}</span>
                <span className={classes.classShare}>{counts.get(c)}</span>
              </div>
            );
          })}
          <a className={classes.osmAttribution} href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">
            © OpenStreetMap contributors
          </a>
        </>
      )}
      {basemap && (
        <a className={classes.osmAttribution} href={tiles.attributionUrl} target="_blank" rel="noopener noreferrer">
          Basemap: {tiles.attribution}
        </a>
      )}
    </div>
  );
}
