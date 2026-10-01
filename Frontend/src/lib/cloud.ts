/** Cloud detection and fill. Pure functions on typed arrays (the canvas side lives in cloudImage.ts).
 *
 *  A cloud is bright, nearly colourless and smooth; a white roof is just as bright but small and edged by
 *  shadow and street. Detection runs on a copy whose long side is at most `ANALYSIS_SIDE`; the soft mask it
 *  returns stays at that size and is resampled bilinearly wherever it is used. */

export interface CloudMask {
  /** Soft alpha, 0 clear .. 255 cloud. `>= 128` is the cloud proper; the ramp below covers its hazy edge. */
  mask: Uint8Array;
  width: number;
  height: number;
  /** Share of the image with alpha >= 128. */
  coverage: number;
}

/** Long side of the image copy the detector works on. */
export const ANALYSIS_SIDE = 1024;
/** Below this share of the image a detection is treated as noise: no prompt, no masking. */
export const MIN_COVERAGE = 0.005;
/** Smallest cloud kept, as a share of the image. Smaller bright smooth blobs are roofs, stands, tanks. */
const MIN_BLOB_SHARE = 0.01;
/** Mean luminance gradient (per pixel, 0..1 scale, at analysis size) above which a bright blob is ground, not cloud. */
const MAX_BLOB_GRADIENT = 0.018;
/** Mean saturation above which a bright blob is soil or a coloured roof. */
const MAX_BLOB_SATURATION = 0.09;

/** Box sum over a (2r+1)² window (clipped at the edges) via an integral image. */
function boxSum(src: Float32Array | Uint8Array, w: number, h: number, r: number): Float32Array {
  const W = w + 1;
  const ii = new Float64Array(W * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += src[y * w + x];
      ii[(y + 1) * W + x + 1] = ii[y * W + x + 1] + row;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w, x + r + 1);
      out[y * w + x] = ii[y1 * W + x1] - ii[y0 * W + x1] - ii[y1 * W + x0] + ii[y0 * W + x0];
    }
  }
  return out;
}

/** Number of in-image pixels in each pixel's (2r+1)² window. */
function boxArea(w: number, h: number, r: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ny = Math.min(h, y + r + 1) - Math.max(0, y - r);
    for (let x = 0; x < w; x++) out[y * w + x] = ny * (Math.min(w, x + r + 1) - Math.max(0, x - r));
  }
  return out;
}

export function dilate(m: Uint8Array, w: number, h: number, r: number): Uint8Array {
  if (r <= 0) return m.slice();
  const s = boxSum(m, w, h, r);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = s[i] > 0.5 ? 1 : 0;
  return out;
}

export function erode(m: Uint8Array, w: number, h: number, r: number): Uint8Array {
  if (r <= 0) return m.slice();
  const s = boxSum(m, w, h, r);
  const a = boxArea(w, h, r);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = s[i] > a[i] - 0.5 ? 1 : 0;
  return out;
}

/** Set every background region not connected to the image border (a hole inside a cloud). */
function fillHoles(m: Uint8Array, w: number, h: number): Uint8Array {
  const seen = new Uint8Array(w * h);
  const stack: number[] = [];
  const push = (i: number) => {
    if (!m[i] && !seen[i]) {
      seen[i] = 1;
      stack.push(i);
    }
  };
  for (let x = 0; x < w; x++) {
    push(x);
    push((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    push(y * w);
    push(y * w + w - 1);
  }
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % w;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (i >= w) push(i - w);
    if (i < (h - 1) * w) push(i + w);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = m[i] || !seen[i] ? 1 : 0;
  return out;
}

/** Keep the 4-connected components of `m` that `keep(pixelIndices)` accepts. */
function filterComponents(m: Uint8Array, w: number, h: number, keep: (comp: number[]) => boolean): Uint8Array {
  const out = new Uint8Array(w * h);
  const seen = new Uint8Array(w * h);
  const comp: number[] = [];
  for (let s = 0; s < m.length; s++) {
    if (!m[s] || seen[s]) continue;
    comp.length = 0;
    seen[s] = 1;
    comp.push(s);
    for (let k = 0; k < comp.length; k++) {
      const i = comp[k];
      const x = i % w;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i >= w ? i - w : -1, i < (h - 1) * w ? i + w : -1];
      for (const j of nb) {
        if (j >= 0 && m[j] && !seen[j]) {
          seen[j] = 1;
          comp.push(j);
        }
      }
    }
    if (keep(comp)) for (const i of comp) out[i] = 1;
  }
  return out;
}

/** Find clouds in an RGBA image (already at analysis size; see `analysisSize`). */
export function detectClouds(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): CloudMask {
  const n = w * h;
  const lum = new Float32Array(n);
  const sat = new Float32Array(n);
  const hist = new Uint32Array(256);
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    const l = (r + g + b) / 3;
    lum[i] = l / 255;
    sat[i] = mx > 0 ? (mx - mn) / mx : 0;
    hist[Math.round(l)]++;
  }
  // Brightness cut relative to the scene: a dark rural scene and a bright, hazy one both work.
  let acc = 0;
  let median = 0.5;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc >= n / 2) {
      median = v / 255;
      break;
    }
  }
  const tBright = Math.min(0.75, Math.max(0.5, median + 0.22));

  // Local texture: luminance standard deviation in a small window.
  const r = Math.max(2, Math.round(Math.max(w, h) / 256));
  const lum2 = new Float32Array(n);
  for (let i = 0; i < n; i++) lum2[i] = lum[i] * lum[i];
  const s1 = boxSum(lum, w, h, r);
  const s2 = boxSum(lum2, w, h, r);
  const area = boxArea(w, h, r);

  let cand: Uint8Array = new Uint8Array(n);
  // haze: the cloud's thin edge — dimmer and tinted by the ground beneath, kept only where it touches a cloud
  const haze = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const mean = s1[i] / area[i];
    const std = Math.sqrt(Math.max(0, s2[i] / area[i] - mean * mean));
    cand[i] = lum[i] > tBright && sat[i] < 0.22 && std < 0.075 ? 1 : 0;
    haze[i] = lum[i] > tBright - 0.12 && sat[i] < 0.3 && std < 0.06 ? 1 : 0;
  }

  const side = Math.max(w, h);
  const o = Math.max(1, Math.round(side / 512));
  cand = dilate(erode(cand, w, h, o), w, h, o); // open: speckle and thin bright edges go
  const c = Math.max(2, Math.round(side / 256));
  cand = erode(dilate(cand, w, h, c), w, h, c); // close: knit the cloud body together
  cand = fillHoles(cand, w, h);
  // Per blob: large, and smooth and colourless *as a whole*. Bright soil, rock and mine spoil pass the
  // per-pixel test in patches but carry field edges and rubble, so their mean gradient is well above a
  // cloud's. Big flat roofs and stadium stands are as smooth as a cloud but small: on the Cartosat test
  // crops every such blob stayed under 0.9 % of the image, the cloud was 5 %.
  const lumSum = boxSum(lum, w, h, 1);
  const grad = new Float32Array(n);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      grad[i] = Math.hypot(lumSum[i + 1] - lumSum[i - 1], lumSum[i + w] - lumSum[i - w]) / 18;
    }
  const minArea = Math.max(64, Math.round(MIN_BLOB_SHARE * n));
  cand = filterComponents(cand, w, h, (comp) => {
    if (comp.length < minArea) return false;
    let gs = 0;
    let ss = 0;
    for (const i of comp) {
      gs += grad[i];
      ss += sat[i];
    }
    return gs / comp.length < MAX_BLOB_GRADIENT && ss / comp.length < MAX_BLOB_SATURATION;
  });

  let core = 0;
  for (let i = 0; i < n; i++) core += cand[i];
  if (core === 0) return { mask: new Uint8Array(n), width: w, height: h, coverage: 0 };

  // hysteresis: add the hazy blobs connected to a confirmed cloud
  const both = new Uint8Array(n);
  for (let i = 0; i < n; i++) both[i] = cand[i] || haze[i] ? 1 : 0;
  const body = cand;
  cand = filterComponents(erode(dilate(both, w, h, o), w, h, o), w, h, (comp) => comp.some((i) => body[i]));

  // Grow over the thin, semi-transparent edge, then feather into a soft alpha.
  const g = Math.max(2, Math.round(side * 0.015));
  const grown = dilate(cand, w, h, g);
  const fr = Math.max(1, Math.round(g / 2));
  const blur = boxSum(grown, w, h, fr);
  const ba = boxArea(w, h, fr);
  const mask = new Uint8Array(n);
  let covered = 0;
  for (let i = 0; i < n; i++) {
    // remap so the grown edge sits at 128 and the cloud body is fully opaque
    const a = Math.min(1, Math.max(0, (blur[i] / ba[i]) * 1.2 - 0.1));
    mask[i] = grown[i] ? Math.max(128, Math.round(a * 255)) : Math.min(127, Math.round(a * 255));
    if (mask[i] >= 128) covered++;
  }
  return { mask, width: w, height: h, coverage: covered / n };
}

/** Size of the copy the detector works on. */
export function analysisSize(w: number, h: number, side = ANALYSIS_SIDE) {
  const s = Math.min(1, side / Math.max(w, h));
  return { width: Math.max(1, Math.round(w * s)), height: Math.max(1, Math.round(h * s)) };
}

/** Bilinear resample of a soft mask to another grid (same extent). */
export function resampleMask(m: Uint8Array, mw: number, mh: number, w: number, h: number): Uint8Array {
  if (mw === w && mh === h) return m.slice();
  const out = new Uint8Array(w * h);
  const sx = mw / w;
  const sy = mh / h;
  for (let y = 0; y < h; y++) {
    const fy = Math.min(mh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(mh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(mw - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(mw - 1, x0 + 1);
      const tx = fx - x0;
      const top = m[y0 * mw + x0] * (1 - tx) + m[y0 * mw + x1] * tx;
      const bot = m[y1 * mw + x0] * (1 - tx) + m[y1 * mw + x1] * tx;
      out[y * w + x] = Math.round(top * (1 - ty) + bot * ty);
    }
  }
  return out;
}

/** Push-pull fill: every cell with `weight` 0 gets a smooth blend of the known cells around it.
 *  `values` holds `channels` interleaved floats per cell; known cells are returned unchanged. */
export function pushPull(values: Float32Array, weight: Float32Array, w: number, h: number, channels: number): Float32Array {
  // pull: a pyramid of premultiplied values and weights clamped to 1
  const levels: { p: Float32Array; wt: Float32Array; w: number; h: number }[] = [];
  let p = new Float32Array(w * h * channels);
  let wt = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const a = Math.min(1, Math.max(0, weight[i]));
    wt[i] = a;
    for (let c = 0; c < channels; c++) p[i * channels + c] = a ? values[i * channels + c] * a : 0;
  }
  let lw = w;
  let lh = h;
  levels.push({ p, wt, w: lw, h: lh });
  while (lw > 1 || lh > 1) {
    const nw = Math.ceil(lw / 2);
    const nh = Math.ceil(lh / 2);
    const np = new Float32Array(nw * nh * channels);
    const nwt = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        let ws = 0;
        const ps = new Float64Array(channels);
        for (let dy = 0; dy < 2; dy++) {
          const yy = 2 * y + dy;
          if (yy >= lh) continue;
          for (let dx = 0; dx < 2; dx++) {
            const xx = 2 * x + dx;
            if (xx >= lw) continue;
            const j = yy * lw + xx;
            ws += wt[j];
            for (let c = 0; c < channels; c++) ps[c] += p[j * channels + c];
          }
        }
        const k = y * nw + x;
        if (ws > 0) {
          const a = Math.min(1, ws);
          nwt[k] = a;
          for (let c = 0; c < channels; c++) np[k * channels + c] = (ps[c] / ws) * a;
        }
      }
    }
    p = np;
    wt = nwt;
    lw = nw;
    lh = nh;
    levels.push({ p, wt, w: lw, h: lh });
  }
  // push: fill each level's missing share from the (already complete) level above, sampled bilinearly
  const top = levels[levels.length - 1];
  for (let c = 0; c < channels; c++) top.p[c] = top.wt[0] > 0 ? top.p[c] / top.wt[0] : 0;
  top.wt[0] = 1;
  for (let l = levels.length - 2; l >= 0; l--) {
    const cur = levels[l];
    const par = levels[l + 1];
    for (let y = 0; y < cur.h; y++) {
      const fy = Math.min(par.h - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
      const y0 = Math.floor(fy);
      const y1 = Math.min(par.h - 1, y0 + 1);
      const ty = fy - y0;
      for (let x = 0; x < cur.w; x++) {
        const k = y * cur.w + x;
        const a = cur.wt[k];
        if (a >= 1) continue;
        const fx = Math.min(par.w - 1, Math.max(0, (x + 0.5) / 2 - 0.5));
        const x0 = Math.floor(fx);
        const x1 = Math.min(par.w - 1, x0 + 1);
        const tx = fx - x0;
        for (let c = 0; c < channels; c++) {
          const v =
            (par.p[(y0 * par.w + x0) * channels + c] * (1 - tx) + par.p[(y0 * par.w + x1) * channels + c] * tx) * (1 - ty) +
            (par.p[(y1 * par.w + x0) * channels + c] * (1 - tx) + par.p[(y1 * par.w + x1) * channels + c] * tx) * ty;
          cur.p[k * channels + c] += (1 - a) * v;
        }
        cur.wt[k] = 1;
      }
    }
  }
  // level 0 is premultiplied by 1 everywhere now; restore known cells exactly
  const out = levels[0].p;
  for (let i = 0; i < w * h; i++) if (weight[i] >= 1) for (let c = 0; c < channels; c++) out[i * channels + c] = values[i * channels + c];
  return out;
}

/** RGBA with the cloud painted over by the surrounding colours. `mask` must be on the same grid.
 *  Known colours come only from clearly cloud-free pixels (alpha < ~8 %), so the hazy edge does not whiten the fill. */
export function fillImage(rgba: Uint8ClampedArray | Uint8Array, mask: Uint8Array, w: number, h: number): Uint8ClampedArray<ArrayBuffer> {
  const n = w * h;
  const vals = new Float32Array(n * 3);
  const wt = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    vals[i * 3] = rgba[i * 4];
    vals[i * 3 + 1] = rgba[i * 4 + 1];
    vals[i * 3 + 2] = rgba[i * 4 + 2];
    wt[i] = mask[i] < 20 ? 1 : 0;
  }
  const filled = pushPull(vals, wt, w, h, 3);
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const a = Math.min(1, mask[i] / 160);
    for (let c = 0; c < 3; c++) out[i * 4 + c] = rgba[i * 4 + c] * (1 - a) + filled[i * 3 + c] * a;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Heights under the cloud replaced by a smooth surface through the surrounding *ground*.
 *  `mask` must be on the height grid. Only a ring of cells around the cloud, and only those at ground level
 *  (at most the ring's 30th percentile + 1 m), feed the fill — a tower beside the cloud must not lift it. */
export function fillHeights(heights: Float32Array, mask: Uint8Array, w: number, h: number): Float32Array {
  const n = w * h;
  const cloud = new Uint8Array(n);
  let any = false;
  for (let i = 0; i < n; i++) {
    if (mask[i] < 128) continue;
    cloud[i] = 1;
    any = true;
  }
  if (!any) return heights.slice();
  const ringR = Math.max(4, Math.round(Math.max(w, h) * 0.01));
  const near = dilate(cloud, w, h, ringR);
  const ringVals: number[] = [];
  for (let i = 0; i < n; i++) if (near[i] && !cloud[i] && Number.isFinite(heights[i])) ringVals.push(heights[i]);
  const out = heights.slice();
  if (!ringVals.length) {
    // nothing to anchor to (the whole scene is cloud): flat at the scene's lowest finite height, else 0
    let lo = Infinity;
    for (let i = 0; i < n; i++) if (Number.isFinite(heights[i])) lo = Math.min(lo, heights[i]);
    const v = Number.isFinite(lo) ? lo : 0;
    for (let i = 0; i < n; i++) if (cloud[i]) out[i] = v;
    return out;
  }
  ringVals.sort((a, b) => a - b);
  const ground = ringVals[Math.floor(0.3 * (ringVals.length - 1))] + 1;
  const vals = new Float32Array(n);
  const wt = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (near[i] && !cloud[i] && Number.isFinite(heights[i]) && heights[i] <= ground) {
      vals[i] = heights[i];
      wt[i] = 1;
    }
  }
  const filled = pushPull(vals, wt, w, h, 1);
  for (let i = 0; i < n; i++) if (cloud[i]) out[i] = filled[i];
  return out;
}
