import type { Metrics, StratumMetrics, UncertaintyGrid, UncertaintyValidation, ValidationResult } from '@/domain/types';
import { MAX_SPARSIFY_PX, sparsification } from './uncertainty';

/** Height strata used by the model's evaluation (v5 eval/metrics.py). Bounds are on the reference height. */
export const STRATA: Array<[number, number, string]> = [
  [-Infinity, 2, '0–2 m'],
  [2, 5, '2–5 m'],
  [5, 10, '5–10 m'],
  [10, 20, '10–20 m'],
  [20, Infinity, '20+ m'],
];

class Acc {
  n = 0;
  se = 0;
  ae = 0;
  sd = 0;
  sx = 0;
  sy = 0;
  sxx = 0;
  syy = 0;
  sxy = 0;
  w1 = 0;
  w2 = 0;
  add(p: number, t: number) {
    const d = p - t;
    this.n++;
    this.se += d * d;
    this.ae += Math.abs(d);
    this.sd += d;
    this.sx += p;
    this.sy += t;
    this.sxx += p * p;
    this.syy += t * t;
    this.sxy += p * t;
    if (Math.abs(d) <= 1) this.w1++;
    if (Math.abs(d) <= 2) this.w2++;
  }
  result(): Metrics {
    const n = this.n;
    if (!n) return { n: 0, rmse: NaN, mae: NaN, bias: NaN, r: NaN, within1m: NaN, within2m: NaN };
    const mx = this.sx / n;
    const my = this.sy / n;
    const cov = this.sxy / n - mx * my;
    const vx = this.sxx / n - mx * mx;
    const vy = this.syy / n - my * my;
    const den = Math.sqrt(Math.max(0, vx * vy));
    return {
      n,
      rmse: Math.sqrt(this.se / n),
      mae: this.ae / n,
      bias: this.sd / n,
      r: den > 1e-12 ? cov / den : NaN,
      within1m: this.w1 / n,
      within2m: this.w2 / n,
    };
  }
}

/** Plain pairwise metrics — prediction vs reference over pixels where both are finite. */
export function computeMetrics(pred: Float32Array, ref: Float32Array): Metrics {
  const a = new Acc();
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i];
    const t = ref[i];
    if (Number.isFinite(p) && Number.isFinite(t)) a.add(p, t);
  }
  return a.result();
}

/** Median of (ref - pred) on pixels the prediction calls ground (< 1 m): the offset that aligns an
 *  absolute reference DSM with a height-above-ground prediction. */
export function groundOffset(pred: Float32Array, ref: Float32Array): number {
  const diffs: number[] = [];
  const stride = Math.max(1, Math.floor(pred.length / 200_000));
  for (let i = 0; i < pred.length; i += stride) {
    const p = pred[i];
    const t = ref[i];
    if (Number.isFinite(p) && Number.isFinite(t) && p < 1) diffs.push(t - p);
  }
  if (!diffs.length) return 0;
  diffs.sort((x, y) => x - y);
  return diffs[Math.floor(diffs.length / 2)];
}

/** Error on the confident pixels (σ at or below the scene's cut) and the sparsification curves, over the pixels where
 *  prediction, reference (less `offset`) and σ are all finite. Null when no pixel qualifies. */
export function validateUncertainty(pred: Float32Array, refIn: Float32Array, sigma: UncertaintyGrid, offset = 0): UncertaintyValidation | null {
  const s = sigma.data;
  const conf = new Acc();
  let compared = 0;
  let withSigma = 0;
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i];
    const t = refIn[i] - offset;
    if (!Number.isFinite(p) || !Number.isFinite(t)) continue;
    compared++;
    if (!Number.isFinite(s[i])) continue;
    withSigma++;
    if (s[i] <= sigma.confidentM) conf.add(p, t);
  }
  if (!withSigma) return null;
  // the curves rank pixels, so an even stride through a big scene keeps their shape
  const stride = Math.max(1, Math.ceil(withSigma / MAX_SPARSIFY_PX));
  const err2 = new Float64Array(Math.ceil(withSigma / stride));
  const sig = new Float64Array(err2.length);
  let k = 0;
  let seen = 0;
  for (let i = 0; i < pred.length && k < err2.length; i++) {
    const p = pred[i];
    const t = refIn[i] - offset;
    if (!Number.isFinite(p) || !Number.isFinite(t) || !Number.isFinite(s[i])) continue;
    if (seen++ % stride) continue;
    err2[k] = (p - t) * (p - t);
    sig[k++] = s[i];
  }
  return {
    confidentM: sigma.confidentM,
    confident: conf.result(),
    coverage: conf.n / compared,
    sparsification: sparsification(err2.subarray(0, k), sig.subarray(0, k)),
  };
}

/** Full validation: global + per-stratum metrics, density scatter and signed-error histogram, and what σ says about the
 *  error when the scene carries one. */
export function validate(pred: Float32Array, refIn: Float32Array, opts: { removeOffset: boolean; sigma?: UncertaintyGrid | null }): ValidationResult {
  const offset = opts.removeOffset ? groundOffset(pred, refIn) : 0;
  const all = new Acc();
  const strata = STRATA.map(() => new Acc());
  let lo = Infinity;
  let hi = -Infinity;
  let emax = 0;
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i];
    const t = refIn[i] - offset;
    if (!Number.isFinite(p) || !Number.isFinite(t)) continue;
    all.add(p, t);
    for (let s = 0; s < STRATA.length; s++) {
      if (t >= STRATA[s][0] && t < STRATA[s][1]) {
        strata[s].add(p, t);
        break;
      }
    }
    lo = Math.min(lo, p, t);
    hi = Math.max(hi, p, t);
    emax = Math.max(emax, Math.abs(p - t));
  }
  if (!Number.isFinite(lo)) {
    lo = 0;
    hi = 1;
  }
  const bins = 96;
  const counts = new Uint32Array(bins * bins);
  const span = hi - lo || 1;
  const eBins = 80;
  const eRange = Math.max(1, Math.min(emax, 50));
  const eCounts = new Uint32Array(eBins);
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i];
    const t = refIn[i] - offset;
    if (!Number.isFinite(p) || !Number.isFinite(t)) continue;
    const bx = Math.min(bins - 1, Math.floor(((t - lo) / span) * bins));
    const by = Math.min(bins - 1, Math.floor(((p - lo) / span) * bins));
    counts[by * bins + bx]++;
    const e = Math.floor(((p - t + eRange) / (2 * eRange)) * eBins);
    eCounts[e < 0 ? 0 : e >= eBins ? eBins - 1 : e]++;
  }
  const strataRes: StratumMetrics[] = strata.map((a, s) => ({
    ...a.result(),
    label: STRATA[s][2],
    lo: STRATA[s][0],
    hi: STRATA[s][1],
  }));
  const populated = strataRes.filter((s) => s.n > 0);
  const balancedRmse = populated.length ? populated.reduce((acc, s) => acc + s.rmse, 0) / populated.length : NaN;
  return {
    all: all.result(),
    strata: strataRes,
    balancedRmse,
    biasRemoved: opts.removeOffset,
    offset,
    scatter: { bins, lo, hi, counts },
    errorHist: { lo: -eRange, hi: eRange, counts: eCounts },
    uncertainty: opts.sigma && opts.sigma.data.length === pred.length ? validateUncertainty(pred, refIn, opts.sigma, offset) : null,
  };
}
