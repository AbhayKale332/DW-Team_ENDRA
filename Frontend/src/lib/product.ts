import type { Scene } from '@/domain/types';

export type ProductTone = 'orange' | 'blue' | 'green';

export interface ProductInfo {
  /** Badge text. */
  short: string;
  /** One-line name. */
  title: string;
  tone: ProductTone;
  /** Legend / axis title. */
  axis: string;
  /** Unit string for readouts: "m", "m a.s.l. (EGM96)", ... */
  unit: string;
  /** Prefix for a single-point readout: "h" or "elev". */
  reading: string;
  /** What the numbers mean, and what they do not. */
  summary: string;
}

const datumLabel = (s: Pick<Scene, 'anchoring'>) => (s.anchoring?.datum && s.anchoring.datum !== 'unknown' ? s.anchoring.datum : 'DEM datum');

/** The honest description of a scene's height product. Absolute only when a DEM has anchored it. */
export function productInfo(scene: Pick<Scene, 'product' | 'anchoring'>): ProductInfo {
  if (scene.product === 'DSM') {
    const datum = datumLabel(scene);
    return {
      short: 'DSM · absolute',
      title: 'Absolute DSM',
      tone: 'green',
      axis: 'Elevation',
      unit: `m a.s.l. (${datum})`,
      reading: 'elev',
      summary: `Elevation above the vertical datum of the DEM it was anchored to (${datum}). Terrain comes from the DEM, structure height from the model.`,
    };
  }
  if (scene.product === 'nDSM') {
    return {
      short: 'nDSM · above ground',
      title: 'Height above ground (nDSM)',
      tone: 'blue',
      axis: 'Height above ground',
      unit: 'm above ground',
      reading: 'h',
      summary: 'Georeferenced heights above local ground. Not an elevation: no terrain or vertical datum has been applied.',
    };
  }
  return {
    short: 'rDSM · relative',
    title: 'Relative DSM (rDSM)',
    tone: 'orange',
    axis: 'Relative height',
    unit: 'm (relative)',
    reading: 'h',
    summary: 'Relative surface model: heights above local ground at the stated ground resolution. The image has no coordinate system, so there is no absolute elevation.',
  };
}

/** A height in metres with the product's meaning attached, e.g. "elev 55.2 m a.s.l." or "h 12.0 m (relative)". */
export function formatHeight(scene: Pick<Scene, 'product' | 'anchoring'>, v: number, digits = 2): string {
  const p = productInfo(scene);
  const unit = scene.product === 'DSM' ? 'm a.s.l.' : scene.product === 'rDSM' ? 'm (rel.)' : 'm';
  return `${p.reading} ${v.toFixed(digits)} ${unit}`;
}
