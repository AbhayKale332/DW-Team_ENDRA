import { useMemo } from 'react';
import { useScene } from '@/store/scene';
import { useTool } from '@/store/tool';
import { useSettings } from '@/store/settings';
import { useCamera, CAMERA_MODE_LABELS } from '@/store/camera';
import { useTerrainInfo } from '@/features/viewport/terrainState';
import { probeAt } from '@/lib/analysis';
import { formatLonLat } from '@/lib/georef';
import classes from './shell.module.css';

/** SNAP-style status bar: cursor read-out · resolution · product · mesh · frame rate. */
export function StatusBar() {
  const show = useSettings((s) => s.showStatusBar);
  const scene = useScene((s) => s.scene);
  const reference = useScene((s) => s.reference);
  const hover = useTool((s) => s.hover);
  const camMode = useCamera((s) => s.mode);
  const info = useTerrainInfo((s) => s.info);
  const fps = useTerrainInfo((s) => s.fps);

  const cursor = useMemo(() => {
    if (!scene || !hover) return null;
    return probeAt(scene, hover, reference?.data);
  }, [scene, hover, reference]);

  if (!show) return null;
  return (
    <footer className={`${classes.statusbar} dw-no-print`} aria-label="Status bar">
      <span className={`${classes.statusCell} ${classes.statusGrow}`} aria-live="off">
        {cursor
          ? `x ${cursor.col.toFixed(0)}  y ${cursor.row.toFixed(0)}  ·  h ${cursor.height.toFixed(2)} m  ·  slope ${cursor.slope.toFixed(1)}°${cursor.lonLat ? `  ·  ${formatLonLat(cursor.lonLat)}` : ''}${cursor.error !== null ? `  ·  Δref ${cursor.error >= 0 ? '+' : ''}${cursor.error.toFixed(2)} m` : ''}`
          : scene
            ? 'Hover the terrain for height, slope and position'
            : 'No scene loaded'}
      </span>
      {scene && (
        <>
          <span className={`${classes.statusCell} ${classes.statusHideSm}`}>
            GSD {scene.gsd.toFixed(3)} m/px ({scene.gsdSource})
          </span>
          <span className={classes.statusCell}>{scene.product}</span>
          <span className={`${classes.statusCell} ${classes.statusHideSm}`}>
            {scene.heights.width}×{scene.heights.height}
            {scene.georef?.epsg ? ` · EPSG:${scene.georef.epsg}` : ''}
          </span>
          {info && (
            <span className={`${classes.statusCell} ${classes.statusHideSm}`}>
              {(info.vertices / 1000).toFixed(0)}k verts{info.step > 1 ? ` (1:${info.step})` : ''}
            </span>
          )}
          <span className={`${classes.statusCell} ${classes.statusHideSm}`}>{CAMERA_MODE_LABELS[camMode]}</span>
        </>
      )}
      <span className={classes.statusCell} style={{ color: fps && fps < 30 ? 'var(--mantine-color-dwOrange-6)' : undefined }}>
        {fps ? `${fps} fps` : '—'}
      </span>
    </footer>
  );
}
