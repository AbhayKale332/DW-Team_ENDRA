/** nDSM + coarse DEM -> absolute DSM, without double-counting buildings.
 *
 *  A TypeScript port of `Model_Traning/v5/geo/calibrate.py` (`dem_anchored`). SRTM / Copernicus / Terrain Tiles
 *  are *surface* models: their ~30 m cells already contain part of every building and tree, so `DEM + nDSM`
 *  counts structures twice. Instead:
 *
 *      A  = valid-pixel block mean onto ~30 m cells
 *      U  = bilinear interpolation from cell centres back to pixels
 *      D  = U(X) + gain * (nDSM - U(A(nDSM)))
 *      X <- X + (DEM_c - A(D))          (Tobler iteration; X starts at DEM_c)
 *
 *  so every 30 m cell of the DSM averages to the DEM while the model supplies everything finer than a cell.
 *  The bare-earth terrain is `U(X - A(nDSM))`. */

export interface CellGrid {
  data: Float64Array;
  rows: number;
  cols: number;
}

/** Number of cells per axis for `n` pixels in blocks of `k` (edge blocks partial). */
export const cellCount = (n: number, k: number) => Math.ceil(n / k);

/** Anchor cell size in pixels: ~`cellM` metres at this GSD. */
export function anchorCellPx(gsd: number, cellM = 30): number {
  return Math.max(1, Math.round(cellM / Math.max(gsd, 1e-3)));
}

/** Mean over k x k blocks (edge blocks partial) of finite pixels; NaN where a block has none. */
export function blockMean(a: ArrayLike<number>, W: number, H: number, k: number): CellGrid {
  const cols = cellCount(W, k);
  const rows = cellCount(H, k);
  const sum = new Float64Array(rows * cols);
  const cnt = new Float64Array(rows * cols);
  for (let y = 0; y < H; y++) {
    const cy = Math.floor(y / k) * cols;
    for (let x = 0; x < W; x++) {
      const v = a[y * W + x];
      if (!Number.isFinite(v)) continue;
      const c = cy + Math.floor(x / k);
      sum[c] += v;
      cnt[c]++;
    }
  }
  const data = new Float64Array(rows * cols);
  for (let c = 0; c < data.length; c++) data[c] = cnt[c] > 0 ? sum[c] / cnt[c] : NaN;
  return { data, rows, cols };
}

/** Indices + weights that interpolate cell-centred samples to pixel centres. The outer half cell is
 *  extrapolated linearly (not clamped): clamping pushes a ripple inwards from every scene edge in the Tobler loop. */
function linWeights(nOut: number, k: number, nIn: number): { i0: Int32Array; i1: Int32Array; w: Float64Array } {
  const i0 = new Int32Array(nOut);
  const i1 = new Int32Array(nOut);
  const w = new Float64Array(nOut);
  if (nIn === 1) return { i0, i1, w };
  for (let p = 0; p < nOut; p++) {
    const t = (p + 0.5) / k - 0.5;
    const a = Math.min(Math.max(Math.floor(t), 0), nIn - 2);
    i0[p] = a;
    i1[p] = a + 1;
    w[p] = t - a;
  }
  return { i0, i1, w };
}

/** Bilinear upsampling U of a cell grid to `outW` x `outH` pixels, `k` pixels per cell. */
export function upsampleCells(c: CellGrid, k: number, outW: number, outH: number): Float32Array {
  const out = new Float32Array(outW * outH);
  const ry = linWeights(outH, k, c.rows);
  const rx = linWeights(outW, k, c.cols);
  for (let y = 0; y < outH; y++) {
    const a = ry.i0[y] * c.cols;
    const b = ry.i1[y] * c.cols;
    const wy = ry.w[y];
    for (let x = 0; x < outW; x++) {
      const j0 = rx.i0[x];
      const j1 = rx.i1[x];
      const wx = rx.w[x];
      const top = (1 - wx) * c.data[a + j0] + wx * c.data[a + j1];
      const bot = (1 - wx) * c.data[b + j0] + wx * c.data[b + j1];
      out[y * outW + x] = (1 - wy) * top + wy * bot;
    }
  }
  return out;
}

/** Fill NaN cells from their neighbours (repeated 8-neighbour averaging); all-NaN becomes 0. */
export function fillNan(c: CellGrid): CellGrid {
  const data = Float64Array.from(c.data);
  if (!data.some(Number.isFinite)) return { ...c, data: new Float64Array(data.length) };
  for (let guard = 0; guard < c.rows + c.cols && data.some((v) => !Number.isFinite(v)); guard++) {
    const next = Float64Array.from(data);
    for (let r = 0; r < c.rows; r++) {
      for (let q = 0; q < c.cols; q++) {
        if (Number.isFinite(data[r * c.cols + q])) continue;
        let s = 0;
        let n = 0;
        for (let dr = -1; dr <= 1; dr++) {
          for (let dq = -1; dq <= 1; dq++) {
            const rr = r + dr;
            const qq = q + dq;
            if (rr < 0 || qq < 0 || rr >= c.rows || qq >= c.cols) continue;
            const v = data[rr * c.cols + qq];
            if (Number.isFinite(v)) {
              s += v;
              n++;
            }
          }
        }
        if (n) next[r * c.cols + q] = s / n;
      }
    }
    data.set(next);
  }
  return { ...c, data };
}

export interface AnchorOptions {
  /** Weight of the model's within-cell detail. 1 keeps all of it (v5 default). */
  detailGain?: number;
  /** Share of the cell-mean structure height (trees, buildings) that the DEM already contains, 0..1.
   *  0: the DEM is taken as bare terrain and the model's heights are added on top (smooth terrain, buildings never
   *  sink; the DSM sits above the DEM by the mean structure height). 1: the DEM is a full surface model, so its
   *  cells are matched exactly and the structure mean is carved out of the terrain (can dig holes where the DEM in
   *  fact does not see the buildings). SRTM at 30 m sits in between and is not knowable per scene. Default 0. */
  structureShare?: number;
  iterations?: number;
}

export interface AnchorResult {
  /** Absolute DSM, NaN where the nDSM is NaN. */
  dsm: Float32Array;
  /** Bare-earth terrain elevation. */
  dtm: Float32Array;
  cellPx: number;
  cells: [rows: number, cols: number];
  /** RMS over cells of (DSM cell mean - DEM cell). Near 0 only when structureShare = 1. */
  cellMeanRmseM: number | null;
  /** Mean of (DSM cell mean - DEM cell): how far the DSM sits above the DEM on average. */
  meanOffsetM: number | null;
  toblerResidualRmsM: number[];
}

/** Anchor `ndsm` (metres above ground, W x H) to `demCells` (DEM value per k-pixel cell, same layout as `blockMean`). */
export function anchorToDem(ndsm: Float32Array, W: number, H: number, k: number, demCells: CellGrid, opts: AnchorOptions = {}): AnchorResult {
  const lam = opts.detailGain ?? 1;
  const iters = opts.iterations ?? 20;
  const share = Math.min(1, Math.max(0, opts.structureShare ?? 0));
  const dem0 = fillNan(demCells);
  const m = fillNan(blockMean(ndsm, W, H, k));
  if (dem0.rows !== m.rows || dem0.cols !== m.cols) throw new Error('DEM cell grid does not match the scene grid');
  // DEM = terrain + share * structure  =>  the DSM's cell mean must be DEM + (1 - share) * structure
  const dem: CellGrid = { rows: dem0.rows, cols: dem0.cols, data: Float64Array.from(dem0.data, (v, i) => v + (1 - share) * m.data[i]) };

  // A(U(.)) on a virtual grid of s pixels per cell: bilinear U in cell units is independent of k.
  const s = Math.max(1, Math.min(k, 8));
  const AU = (c: CellGrid): Float64Array => blockMean(upsampleCells(c, s, c.cols * s, c.rows * s), c.cols * s, c.rows * s, s).data;
  const detailMean = AU(m);
  for (let i = 0; i < detailMean.length; i++) detailMean[i] = lam * (m.data[i] - detailMean[i]);
  const x: CellGrid = { data: Float64Array.from(dem.data), rows: dem.rows, cols: dem.cols };
  const res: number[] = [];
  for (let it = 0; it < iters; it++) {
    const au = AU(x);
    let ss = 0;
    for (let i = 0; i < x.data.length; i++) {
      const r = dem.data[i] - (au[i] + detailMean[i]);
      x.data[i] += r;
      ss += r * r;
    }
    res.push(Math.sqrt(ss / x.data.length));
  }

  const ux = upsampleCells(x, k, W, H);
  const um = upsampleCells(m, k, W, H);
  const dsm = new Float32Array(W * H);
  const dtm = new Float32Array(W * H);
  for (let i = 0; i < dsm.length; i++) {
    if (!Number.isFinite(ndsm[i])) {
      dsm[i] = NaN;
      dtm[i] = NaN;
      continue;
    }
    dsm[i] = ux[i] + lam * (ndsm[i] - um[i]);
    dtm[i] = ux[i] - um[i];
  }
  const a = blockMean(dsm, W, H, k);
  let ss = 0;
  let sd = 0;
  let n = 0;
  for (let i = 0; i < a.data.length; i++) {
    if (Number.isFinite(a.data[i]) && Number.isFinite(demCells.data[i])) {
      ss += (a.data[i] - demCells.data[i]) ** 2;
      sd += a.data[i] - demCells.data[i];
      n++;
    }
  }
  return { dsm, dtm, cellPx: k, cells: [dem.rows, dem.cols], cellMeanRmseM: n ? Math.sqrt(ss / n) : null, meanOffsetM: n ? sd / n : null, toblerResidualRmsM: res };
}
