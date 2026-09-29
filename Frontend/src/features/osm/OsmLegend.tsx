import { useMemo } from 'react';
import { Loader } from '@mantine/core';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { canGeolocate, OSM_KIND_COLORS, OSM_KIND_LABELS, type OsmKind } from '@/lib/osm';
import { useOsm } from './osmStore';
import classes from '@/features/viewport/overlays/overlays.module.css';

/** Legend, status and the attribution OpenStreetMap's licence (ODbL) requires while the overlay is shown. */
export function OsmLegend() {
  const scene = useScene((s) => s.scene);
  const on = useView((s) => s.osm);
  const osm = useOsm();
  const mine = !!scene && osm.sceneId === scene.id;
  const counts = useMemo(() => {
    const m = new Map<OsmKind, number>();
    if (mine && osm.status === 'ready') for (const f of osm.features) m.set(f.kind, (m.get(f.kind) ?? 0) + 1);
    return m;
  }, [mine, osm.status, osm.features]);

  if (!on || !scene || !canGeolocate(scene.georef)) return null;
  return (
    <div className={`dw-float ${classes.classLegend}`} role="group" aria-label="OpenStreetMap overlay">
      <div className={classes.classLegendTitle}>OpenStreetMap</div>
      {mine && osm.status === 'loading' && (
        <div className={classes.classLegendRow}>
          <Loader size={10} />
          <span>Loading map data…</span>
        </div>
      )}
      {mine && osm.status === 'error' && <div style={{ maxWidth: 220 }}>Unavailable: {osm.error}</div>}
      {[...counts.entries()].map(([kind, n]) => (
        <div key={kind} className={classes.classLegendRow}>
          <span className={classes.classSwatch} style={{ background: OSM_KIND_COLORS[kind] }} aria-hidden />
          <span>{OSM_KIND_LABELS[kind]}</span>
          <span className={classes.classShare}>{n}</span>
        </div>
      ))}
      <a className={classes.osmAttribution} href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">
        © OpenStreetMap contributors
      </a>
    </div>
  );
}
