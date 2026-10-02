import { useMemo } from 'react';
import { useScene } from '@/store/scene';
import { activeLayer, useView } from '@/store/view';
import { useCamera } from '@/store/camera';
import { cssGradient } from '@/theme/colormaps';
import { classColor, classLabel, classShares } from '@/theme/classes';
import { formatDistance } from '@/lib/heights';
import { productInfo } from '@/lib/product';
import { displayRange } from '../terrainState';
import classes from './overlays.module.css';

function fmt(v: number, unit: string) {
  const d = Math.abs(v) >= 100 ? 0 : 1;
  return `${v.toFixed(d)} ${unit}`;
}

/** Vertical colorbar for the active height-derived layer, in physical units. */
export function Colorbar() {
  const scene = useScene((s) => s.scene);
  const view = useView();
  const layer = activeLayer(view);
  const spec = useMemo(() => {
    if (!scene) return null;
    if (layer === 'optical' || layer === 'hillshade' || layer === 'classes') return null;
    if (layer === 'slope') return { title: 'Slope', ramp: cssGradient('slope'), ticks: [view.slopeMax, view.slopeMax / 2, 0].map((v) => `${v.toFixed(0)}°`) };
    if (layer === 'uncertainty') {
      const u = scene.uncertainty;
      if (!u) return null;
      return { title: 'Uncertainty (1σ)', ramp: cssGradient('confidence'), ticks: [`≥ ${fmt(2 * u.confidentM, 'm')}`, fmt(u.confidentM, 'm'), '0 m'] };
    }
    if (layer === 'error') {
      const r = Math.max(1, (scene.stats.p98 - scene.stats.p2) * 0.25);
      return { title: 'Prediction − reference', ramp: cssGradient('diverging'), ticks: [`+${fmt(r, 'm')}`, '0 m', `−${fmt(r, 'm')}`] };
    }
    const [lo, hi] = displayRange(scene.stats, view.rangeMode, view.customRange);
    const title = layer === 'reference' ? 'Reference height' : productInfo(scene).axis;
    const unit = scene.product === 'DSM' ? 'm a.s.l.' : scene.product === 'rDSM' ? 'm rel.' : 'm';
    return { title, ramp: cssGradient(view.colormap), ticks: [fmt(hi, unit), fmt((lo + hi) / 2, unit), fmt(lo, unit)] };
  }, [scene, layer, view.slopeMax, view.colormap, view.rangeMode, view.customRange]);

  if (!spec) return null;
  return (
    <div className={`dw-float ${classes.colorbar}`} role="img" aria-label={`${spec.title} legend from ${spec.ticks[2]} to ${spec.ticks[0]}`}>
      <div className={classes.colorbarTitle}>{spec.title}</div>
      <div className={classes.colorbarRamp} style={{ background: spec.ramp }} />
      <div className={classes.colorbarTicks}>
        {spec.ticks.map((t) => (
          <span key={t}>{t}</span>
        ))}
      </div>
    </div>
  );
}

/** Categorical legend for the object-classes layer: only classes present in the scene, largest share first. */
export function ClassLegend() {
  const map = useScene((s) => s.scene?.classes);
  const layer = useView(activeLayer);
  const rows = useMemo(() => (map ? classShares(map) : []), [map]);
  if (!map || layer !== 'classes' || !rows.length) return null;
  return (
    <div className={`dw-float ${classes.classLegend}`} role="list" aria-label="Object classes legend">
      <div className={classes.classLegendTitle}>Object classes</div>
      {rows.map(({ id, share }) => (
        <div key={id} role="listitem" className={classes.classLegendRow}>
          <span className={classes.classSwatch} style={{ background: classColor(map.names[id]) }} aria-hidden />
          <span>{classLabel(map.names[id], id)}</span>
          <span className={classes.classShare}>{share < 0.001 ? '<0.1' : (share * 100).toFixed(1)} %</span>
        </div>
      ))}
    </div>
  );
}

const NICE = [1, 2, 5];

/** Ground-distance scale bar for the 2D (orthographic) views. */
export function ScaleBar() {
  const mpp = useCamera((s) => s.mpp);
  const mode = useView((s) => s.mode);
  const bar = useMemo(() => {
    const target = 120 * mpp;
    const exp = Math.floor(Math.log10(target));
    let best = 1;
    for (const n of NICE) {
      const v = n * 10 ** exp;
      if (v <= target) best = v;
    }
    return { metres: best, px: best / mpp };
  }, [mpp]);
  if (mode === 'dsm3d' || !Number.isFinite(bar.px) || bar.px <= 0) return null;
  return (
    <div className={`dw-float ${classes.scalebar}`} aria-label={`Scale: ${formatDistance(bar.metres)}`} role="img">
      <div>{formatDistance(bar.metres)}</div>
      <div className={classes.scalebarLine} style={{ width: `${bar.px}px` }} />
    </div>
  );
}
