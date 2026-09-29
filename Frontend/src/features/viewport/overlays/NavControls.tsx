import { Tooltip } from '@mantine/core';
import { useFullscreenElement } from '@mantine/hooks';
import { IconMaximize, IconMinimize, IconMinus, IconPlus, IconRefresh } from '@tabler/icons-react';
import { useEffect } from 'react';
import { useCamera, viewportApi } from '@/store/camera';
import { useScene } from '@/store/scene';
import classes from './overlays.module.css';

/** Right-side viewport stack: zoom in / zoom out / fullscreen / reset (reset only once the view has moved). */
export function NavControls() {
  const isHome = useCamera((s) => s.isHome);
  const mode = useCamera((s) => s.mode);
  const hasScene = useScene((s) => !!s.scene);
  const { toggle, fullscreen, ref } = useFullscreenElement<HTMLElement>();

  useEffect(() => {
    ref(document.getElementById('dw-viewport'));
  }, [ref]);

  const zoomLabel = mode === 'flight' ? 'throttle' : mode === 'tour' ? 'tour speed' : 'zoom';
  return (
    <div className={`dw-float ${classes.navStack}`} role="toolbar" aria-label="Viewport navigation" aria-orientation="vertical">
      <Tooltip label={mode === 'orbit' ? 'Zoom in (+)' : `Increase ${zoomLabel}`} position="left">
        <button type="button" className={classes.navBtn} onClick={() => viewportApi.zoom(1)} disabled={!hasScene} aria-label={`Increase ${zoomLabel}`}>
          <IconPlus size={20} stroke={1.8} />
        </button>
      </Tooltip>
      <Tooltip label={mode === 'orbit' ? 'Zoom out (−)' : `Decrease ${zoomLabel}`} position="left">
        <button type="button" className={classes.navBtn} onClick={() => viewportApi.zoom(-1)} disabled={!hasScene} aria-label={`Decrease ${zoomLabel}`}>
          <IconMinus size={20} stroke={1.8} />
        </button>
      </Tooltip>
      <Tooltip label={fullscreen ? 'Exit fullscreen' : 'Fullscreen'} position="left">
        <button type="button" className={classes.navBtn} onClick={() => void toggle()} aria-label={fullscreen ? 'Exit fullscreen' : 'Enter fullscreen'} aria-pressed={fullscreen}>
          {fullscreen ? <IconMinimize size={20} stroke={1.8} /> : <IconMaximize size={20} stroke={1.8} />}
        </button>
      </Tooltip>
      {!isHome && hasScene && (
        <Tooltip label="Reset view (R)" position="left">
          <button type="button" className={classes.navBtn} onClick={() => viewportApi.reset()} aria-label="Reset view">
            <IconRefresh size={20} stroke={1.8} />
          </button>
        </Tooltip>
      )}
    </div>
  );
}
