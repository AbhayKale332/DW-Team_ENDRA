import type { ObjectKinds, Scene } from '@/domain/types';

const ALL_KINDS: ObjectKinds = { buildings: true, trees: true, water: true };

/** The terrain surface to mesh when objects are drawn as 3D models.
 *
 *  A predicted nDSM carries every tree and building as a smooth bump. Once the Objects layer draws a model
 *  there, the bump would poke through it, so the *mesh* gets it replaced by the ground around it — crowns,
 *  building footprints and water. Ground, road and low vegetation are smoothed so flat ground reads flat.
 *  Only the mesh: probe, profile, validation and every colour layer keep reading the raw `scene.heights`. */

type SurfaceScene = Pick<Scene, 'heights' | 'classes' | 'objects' | 'gsd'> & Partial<Pick<Scene, 'terrain' | 'ndsm' | 'product'>>;
/** What the terrain worker needs to compute the surface (structured-cloneable). */
export type ObjectSurfaceInput = SurfaceScene;

/** How far past the crown radius (from the crown's area) canopy pixels are still claimed by a tree —
 *  crowns are not circles. Without a class map there is nothing to trim the disc with, so stay closer. */
const REACH_WITH_CLASSES = 1.5;
const REACH_WITHOUT_CLASSES = 1.15;

/** Tree-class pixels grown by one pixel (3×3), so the soft fringe the head leaves round a crown goes too. */
function dilatedClassMask(data: Uint8Array, W: number, H: number, id: number): Uint8Array {
  const rows = new Uint8Array(W * H);
  for (let r = 0; r < H; r++) {
    const o = r * W;
    for (let c = 0; c < W; c++) {
      rows[o + c] = data[o + c] === id || (c > 0 && data[o + c - 1] === id) || (c < W - 1 && data[o + c + 1] === id) ? 1 : 0;
    }
  }
  const out = new Uint8Array(W * H);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      out[i] = rows[i] || (r > 0 && rows[i - W]) || (r < H - 1 && rows[i + W]) ? 1 : 0;
    }
  }
  return out;
}

/** Pixels whose height is a crown that the Trees layer draws as a model; null when there are no trees. */
export function treeCrownMask(scene: SurfaceScene): Uint8Array | null {
  const trees = scene.objects?.trees;
  if (!trees?.length) return null;
  const { width: W, height: H } = scene.heights;
  const treeId = scene.classes ? scene.classes.names.indexOf('tree') : -1;
  const near = scene.classes && treeId >= 0 ? dilatedClassMask(scene.classes.data, W, H, treeId) : null;
  const reach = near ? REACH_WITH_CLASSES : REACH_WITHOUT_CLASSES;
  const mask = new Uint8Array(W * H);
  for (const t of trees) {
    const R = (reach * t.r) / scene.gsd;
    // objects use corner-origin pixels; pixel indices are centres
    const cx = t.x - 0.5;
    const cy = t.y - 0.5;
    const r0 = Math.max(0, Math.floor(cy - R));
    const r1 = Math.min(H - 1, Math.ceil(cy + R));
    const c0 = Math.max(0, Math.floor(cx - R));
    const c1 = Math.min(W - 1, Math.ceil(cx + R));
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        if ((c - cx) ** 2 + (r - cy) ** 2 > R * R) continue;
        const i = r * W + c;
        if (!near || near[i]) mask[i] = 1;
      }
    }
  }
  return mask;
}

/** Replace masked pixels with the ground around them: a breadth-first sweep inward from the mask's edge,
 *  each ring taking the lowest already-known 4-neighbour. The minimum (not the mean) keeps unmasked canopy
 *  fringe from leaking a raised plateau into the hole. A masked region with no unmasked pixel to grow from
 *  gets the scene minimum. */
export function fillFromEdges(src: Float32Array, W: number, H: number, mask: Uint8Array): Float32Array {
  const N = W * H;
  const out = src.slice();
  // 0 = known, 1 = masked and not reached yet, 2 = in the current / next ring.
  // Typed-array queues and inlined neighbours: this runs over millions of pixels on the main thread.
  const state = new Uint8Array(N);
  for (let i = 0; i < N; i++) state[i] = mask[i] ? 1 : 0;
  const lowest = (i: number) => {
    const c = i % W;
    let m = Infinity;
    let v: number;
    if (c > 0 && state[i - 1] === 0 && (v = out[i - 1]) < m) m = v;
    if (c < W - 1 && state[i + 1] === 0 && (v = out[i + 1]) < m) m = v;
    if (i >= W && state[i - W] === 0 && (v = out[i - W]) < m) m = v;
    if (i < N - W && state[i + W] === 0 && (v = out[i + W]) < m) m = v;
    return m; // NaN never compares below m, so non-finite neighbours are skipped
  };

  let front = new Int32Array(N);
  let next = new Int32Array(N);
  const vals = new Float32Array(N);
  let fn = 0;
  for (let i = 0; i < N; i++) {
    if (state[i] !== 1) continue;
    const c = i % W;
    if ((c > 0 && state[i - 1] === 0) || (c < W - 1 && state[i + 1] === 0) || (i >= W && state[i - W] === 0) || (i < N - W && state[i + W] === 0)) {
      front[fn++] = i;
      state[i] = 2;
    }
  }
  while (fn) {
    // compute the whole ring first, so the sweep order inside a ring cannot bias it
    for (let k = 0; k < fn; k++) vals[k] = lowest(front[k]);
    for (let k = 0; k < fn; k++) {
      if (vals[k] !== Infinity) out[front[k]] = vals[k];
      state[front[k]] = 0;
    }
    let nn = 0;
    const enqueue = (j: number) => {
      if (state[j] === 1) {
        state[j] = 2;
        next[nn++] = j;
      }
    };
    for (let k = 0; k < fn; k++) {
      const i = front[k];
      const c = i % W;
      if (c > 0) enqueue(i - 1);
      if (c < W - 1) enqueue(i + 1);
      if (i >= W) enqueue(i - W);
      if (i < N - W) enqueue(i + W);
    }
    [front, next] = [next, front];
    fn = nn;
  }

  let floor = Infinity;
  for (let i = 0; i < N; i++) if (Number.isFinite(src[i]) && src[i] < floor) floor = src[i];
  for (let i = 0; i < N; i++) if (state[i] !== 0) out[i] = Number.isFinite(floor) ? floor : 0;
  return out;
}

/** Even-odd fill of a corner-origin pixel polygon: a pixel is inside when its centre (col + 0.5, row + 0.5) is. */
export function rasterPolygon(mask: Uint8Array, W: number, H: number, poly: Array<[number, number]>) {
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const [, y] of poly) {
    y0 = Math.min(y0, y);
    y1 = Math.max(y1, y);
  }
  const xs: number[] = [];
  for (let r = Math.max(0, Math.floor(y0 - 0.5)); r <= Math.min(H - 1, Math.ceil(y1 - 0.5)); r++) {
    const yc = r + 0.5;
    xs.length = 0;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i];
      const [xj, yj] = poly[j];
      if (yi <= yc !== yj <= yc) xs.push(xi + ((yc - yi) * (xj - xi)) / (yj - yi));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const c0 = Math.max(0, Math.ceil(xs[k] - 0.5));
      const c1 = Math.min(W - 1, Math.ceil(xs[k + 1] - 0.5) - 1);
      for (let c = c0; c <= c1; c++) mask[r * W + c] = 1;
    }
  }
}

/** Grow a mask by `k` pixels (square neighbourhood), separably. */
export function dilate(mask: Uint8Array, W: number, H: number, k: number): Uint8Array {
  if (k <= 0) return mask;
  const rows = new Uint8Array(W * H);
  for (let r = 0; r < H; r++) {
    let last = -Infinity; // column of the most recent set pixel
    for (let c = 0; c < W + k; c++) {
      if (c < W && mask[r * W + c]) last = c;
      const target = c - k;
      if (target >= 0 && c - last <= 2 * k) rows[r * W + target] = 1;
    }
  }
  const out = new Uint8Array(W * H);
  for (let c = 0; c < W; c++) {
    let last = -Infinity;
    for (let r = 0; r < H + k; r++) {
      if (r < H && rows[r * W + c]) last = r;
      const target = r - k;
      if (target >= 0 && r - last <= 2 * k) out[target * W + c] = 1;
    }
  }
  return out;
}

/** Footprints the Objects layer draws as solids: buildings (grown by ~1 m, where the head blurs roof into
 *  ground) and water — only the classes switched on. Null when there are none. */
export function footprintMask(scene: SurfaceScene, kinds: ObjectKinds = ALL_KINDS): Uint8Array | null {
  const o = scene.objects;
  const buildingPolys = kinds.buildings ? (o?.buildings ?? []) : [];
  const waterPolys = kinds.water ? (o?.water ?? []) : [];
  if (!buildingPolys.length && !waterPolys.length) return null;
  const { width: W, height: H } = scene.heights;
  let mask: Uint8Array = new Uint8Array(W * H);
  if (buildingPolys.length) {
    for (const b of buildingPolys) rasterPolygon(mask, W, H, b.poly);
    mask = dilate(mask, W, H, Math.max(1, Math.round(1 / scene.gsd)));
  }
  if (waterPolys.length) {
    const water = new Uint8Array(W * H);
    for (const w of waterPolys) rasterPolygon(water, W, H, w.poly);
    const wet = dilate(water, W, H, 1);
    for (let i = 0; i < mask.length; i++) mask[i] |= wet[i];
  }
  return mask;
}

const GROUND_LIKE = ['ground', 'road', 'low_veg', 'water'];

/** Box-smooth ground-like pixels over ground-like neighbours only (normalised convolution, radius ~1.5 m),
 *  so predicted noise on open ground does not render as rubble — and a roof never bleeds into a road. */
export function smoothGround(src: Float32Array, W: number, H: number, groundLike: Uint8Array, radius: number): Float32Array {
  const out = src.slice();
  if (radius < 1) return out;
  // integral images of value·mask and mask, (W+1)×(H+1)
  const S = new Float64Array((W + 1) * (H + 1));
  const N = new Float64Array((W + 1) * (H + 1));
  for (let r = 0; r < H; r++) {
    let rs = 0;
    let rn = 0;
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      if (groundLike[i] && Number.isFinite(src[i])) {
        rs += src[i];
        rn++;
      }
      const o = (r + 1) * (W + 1) + c + 1;
      S[o] = S[o - W - 1] + rs;
      N[o] = N[o - W - 1] + rn;
    }
  }
  for (let r = 0; r < H; r++) {
    const ra = Math.max(0, r - radius);
    const rb = Math.min(H, r + radius + 1);
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      if (!groundLike[i]) continue;
      const ca = Math.max(0, c - radius);
      const cb = Math.min(W, c + radius + 1);
      const at = (rr: number, cc: number) => rr * (W + 1) + cc;
      const n = N[at(rb, cb)] - N[at(ra, cb)] - N[at(rb, ca)] + N[at(ra, ca)];
      if (n > 0) out[i] = (S[at(rb, cb)] - S[at(ra, cb)] - S[at(rb, ca)] + S[at(ra, ca)]) / n;
    }
  }
  return out;
}

const cache = new WeakMap<object, Map<string, Float32Array>>();
const kindsKey = (k: ObjectKinds) => `b${+k.buildings}t${+k.trees}w${+k.water}`;

/** Heights for the terrain mesh with the given object classes drawn as models (raw heights when none apply).
 *  Cached per scene and selection. Synchronous — the app computes this in the terrain worker
 *  (features/viewport/objectSurfaceClient). */
export function objectSurfaceHeights(scene: SurfaceScene, kinds: ObjectKinds = ALL_KINDS): Float32Array {
  let perScene = cache.get(scene);
  if (!perScene) cache.set(scene, (perScene = new Map()));
  const key = kindsKey(kinds);
  const hit = perScene.get(key);
  if (hit) return hit;
  const out = computeObjectSurface(scene, kinds);
  perScene.set(key, out);
  return out;
}

/** The uncached computation (runs inside the terrain worker). Flattens under the classes in `kinds` only —
 *  a class that is off keeps its bump in the mesh. Returns `scene.heights.data` itself when nothing applies. */
export function computeObjectSurface(scene: SurfaceScene, kinds: ObjectKinds = ALL_KINDS): Float32Array {
  // An absolute DSM is bare terrain + above-ground height. Objects are flattened in the above-ground domain, where
  // ground is ~0 and the fill is well behaved, and the terrain is added back, so they stand on the real ground.
  if (scene.product === 'DSM' && scene.terrain && scene.ndsm && scene.terrain.data.length === scene.ndsm.data.length) {
    const flat = computeObjectSurface({ ...scene, heights: scene.ndsm, product: 'nDSM', terrain: undefined, ndsm: undefined }, kinds);
    if (flat === scene.ndsm.data) return scene.heights.data;
    const t = scene.terrain.data;
    const out = new Float32Array(flat.length);
    for (let i = 0; i < out.length; i++) out[i] = flat[i] + t[i];
    return out;
  }
  const { data, width: W, height: H } = scene.heights;
  const crowns = kinds.trees ? treeCrownMask(scene) : null;
  const solids = footprintMask(scene, kinds);
  let out = data;
  if (crowns || solids) {
    const mask = crowns ?? new Uint8Array(W * H);
    if (solids) for (let i = 0; i < mask.length; i++) mask[i] |= solids[i];
    let base = data;
    const cls = scene.classes;
    if (cls) {
      const isGround = new Uint8Array(256);
      for (const n of GROUND_LIKE) {
        const id = cls.names.indexOf(n);
        if (id >= 0) isGround[id] = 1;
      }
      const groundLike = new Uint8Array(W * H);
      for (let i = 0; i < groundLike.length; i++) groundLike[i] = isGround[cls.data[i]] && !mask[i] ? 1 : 0;
      base = smoothGround(data, W, H, groundLike, Math.round(1.5 / scene.gsd));
    }
    out = fillFromEdges(base, W, H, mask);
  }
  return out;
}
