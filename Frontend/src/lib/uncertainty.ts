import type { GridPoint, HeightGrid, Scene, SceneMeta, Sparsification, UncertaintyGrid } from '@/domain/types';
import { parseNpy } from './npy';
import { computeStats, sampleBilinear } from './heights';

/** Per-pixel uncertainty σ (`ndsm_std_m.npy`, the spread of the model's height bins, one standard deviation in metres)
 *  and the checks that tell whether it tracks the real error. Only v5 backends write it; everything here returns null
 *  when it is missing, so a result without σ shows no uncertainty rather than a made-up one. */

/** The backend's "confident" cut is never below 1 m (v5 infer/predict.py `uncertainty_summary`). */
export const CONFIDENT_FLOOR_M = 1;

/** σ on the height grid from `ndsm_std_m.npy`; null when the file is missing, unreadable, the wrong size, or carries no
 *  spread at all (every value zero or non-finite). Negative values are not a spread: they become no-data. */
export function parseUncertainty(buf: ArrayBuffer | null | undefined, width: number, height: number): HeightGrid | null {
  if (!buf) return null;
  let arr;
  try {
    arr = parseNpy(buf);
  } catch (e) {
    console.warn('ndsm_std_m.npy unreadable; uncertainty layer disabled', e);
    return null;
  }
  if (arr.shape.length !== 2 || arr.shape[0] !== height || arr.shape[1] !== width) {
    console.warn(`ndsm_std_m.npy is ${arr.shape.join('×')}, the heights ${height}×${width}; uncertainty layer disabled`);
    return null;
  }
  const data = arr.data;
  let spread = false;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v) || v < 0) data[i] = NaN;
    else if (v > 0) spread = true;
  }
  return spread ? { data, width, height } : null;
}

/** σ with its confident cut: `meta.json` `uncertainty.confident_threshold_m` when the backend wrote one, else the same rule
 *  computed here — max(1 m, the scene's median σ). Pixels under clouds hold filled heights, so they get no σ. */
export function withConfidentCut(grid: HeightGrid, meta: SceneMeta, cloudUnder?: Uint8Array | null): UncertaintyGrid {
  let data = grid.data;
  if (cloudUnder && cloudUnder.length === data.length) {
    data = data.slice();
    for (let i = 0; i < cloudUnder.length; i++) if (cloudUnder[i] >= 128) data[i] = NaN;
  }
  const given = meta.uncertainty?.confident_threshold_m;
  const confidentM = typeof given === 'number' && given > 0 ? given : Math.max(CONFIDENT_FLOOR_M, computeStats(data).median);
  return { ...grid, data, confidentM };
}

/** σ at a grid point (pixel centres at integers, as the pick tools use); null where there is none. */
export function sigmaAt(scene: Pick<Scene, 'uncertainty'>, p: GridPoint | { col: number; row: number }): number | null {
  const u = scene.uncertainty;
  if (!u) return null;
  const [col, row] = Array.isArray(p) ? p : [p.col, p.row];
  const v = sampleBilinear(u, col, row);
  return Number.isFinite(v) ? v : null;
}

/** "± 0.8 m", or '' without σ. */
export const plusMinus = (sigma: number | null, digits = 1) => (sigma === null ? '' : ` ± ${sigma.toFixed(digits)} m`);

// ---------------------------------------------------------------------------------------------
// calibration: does σ rank the pixels the way the error does?

/** Big scenes are strided down to this many pixels for the curves: the shape of a curve needs far fewer. */
export const MAX_SPARSIFY_PX = 2_000_000;

/** Linear-interpolated percentile of an ascending array, `p` in 0..1 (numpy's default). */
function quantile(sorted: Float64Array, p: number) {
  const pos = p * (sorted.length - 1);
  const i = Math.floor(pos);
  const f = pos - i;
  return i + 1 < sorted.length ? sorted[i] + (sorted[i + 1] - sorted[i]) * f : sorted[i];
}

/** RMSE of the pixels kept at each step: step j keeps u <= percentile(u, 1 - j/K). The cuts only fall as j grows, so every
 *  pixel is kept for a prefix of steps; bucket each pixel by its last step, then sum the buckets from the end. */
function sparsifyBy(err2: Float64Array, u: Float64Array, K: number): number[] {
  const sorted = u.slice().sort();
  const cut = Array.from({ length: K }, (_, j) => quantile(sorted, 1 - j / K));
  const se = new Float64Array(K);
  const cnt = new Float64Array(K);
  for (let i = 0; i < u.length; i++) {
    // last step j with u <= cut[j] (cut[0] is the maximum, so j >= 0)
    let lo = 0;
    let hi = K - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (u[i] <= cut[mid]) lo = mid;
      else hi = mid - 1;
    }
    se[lo] += err2[i];
    cnt[lo]++;
  }
  const out = new Array<number>(K + 1);
  let s = 0;
  let c = 0;
  for (let j = K - 1; j >= 0; j--) {
    s += se[j];
    c += cnt[j];
    out[j] = c ? Math.sqrt(s / c) : 0;
  }
  out[K] = 0;
  return out;
}

/** Trapezoidal area under evenly spaced samples. */
const trapz = (y: number[], dx: number) => y.reduce((a, v, i) => a + (i ? ((y[i - 1] + v) / 2) * dx : 0), 0);

/** Sparsification of squared errors `err2` against their σ (same length, all finite), after Poggi et al., CVPR 2020
 *  (arXiv 2005.06209) with the maths of their reference code (mono-uncertainty `compute_aucs`); see `Sparsification`.
 *  Model_Traning/v5/eval/sparsification.py computes the same numbers on the held-out sets. Null with fewer than 2 pixels. */
export function sparsification(err2: Float64Array, sigma: Float64Array, intervals = 50): Sparsification | null {
  const n = err2.length;
  if (n < 2 || sigma.length !== n) return null;
  const K = Math.max(1, Math.floor(intervals));
  let s = 0;
  for (let i = 0; i < n; i++) s += err2[i];
  const rmse = Math.sqrt(s / n);
  const bySigma = sparsifyBy(err2, sigma, K);
  const oracle = sparsifyBy(err2, err2, K);
  const area = trapz(bySigma, 1 / K);
  return {
    removed: Array.from({ length: K + 1 }, (_, j) => j / K),
    bySigma,
    oracle,
    rmse,
    ause: area - trapz(oracle, 1 / K),
    aurg: rmse - area,
    n,
  };
}
