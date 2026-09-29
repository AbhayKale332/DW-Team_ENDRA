/** The analysis grid shared by the use cases: the scene reduced to at most MAX_SIDE pixels per side.
 *
 *  Surface heights are max-pooled so a thin building or wall is never averaged away (obstacles matter for radio
 *  paths); terrain is mean-pooled (it is smooth by construction). Positions in the scene grid convert with
 *  `toAnalysis` / `toScene`. */
import type { Scene } from '@/domain/types';

export const MAX_SIDE = 640;

export interface AnalysisGrid {
  w: number;
  h: number;
  /** Scene pixels per analysis pixel (integer >= 1). */
  f: number;
  /** Ground size of one analysis pixel, metres. */
  cellM: number;
  /** Top of everything (DSM, or nDSM when the scene has no terrain), metres. Same datum as `ground`. */
  surface: Float32Array;
  /** Bare-earth elevation. All zeros when the scene has no terrain (rDSM / nDSM): flat ground is assumed. */
  ground: Float32Array;
  hasTerrain: boolean;
  /** 1 where the class map says building / water; null without a class map. */
  building: Uint8Array | null;
  water: Uint8Array | null;
  /** Scene size, for converting coordinates back. */
  sceneW: number;
  sceneH: number;
}

/** Scene-grid pixel-centre coordinates -> analysis-grid pixel-centre coordinates. */
export const toAnalysis = (g: Pick<AnalysisGrid, 'f'>, col: number, row: number): [number, number] => [(col + 0.5) / g.f - 0.5, (row + 0.5) / g.f - 0.5];
export const toScene = (g: Pick<AnalysisGrid, 'f'>, x: number, y: number): [number, number] => [(x + 0.5) * g.f - 0.5, (y + 0.5) * g.f - 0.5];

export function buildAnalysisGrid(scene: Scene, maxSide = MAX_SIDE): AnalysisGrid {
  const { width: W, height: H } = scene.heights;
  const f = Math.max(1, Math.ceil(Math.max(W, H) / maxSide));
  const w = Math.ceil(W / f);
  const h = Math.ceil(H / f);
  const hasTerrain = !!scene.terrain && !!scene.ndsm;
  // With terrain the primary product may be the nDSM view; the surface is always terrain + above-ground height.
  const top = hasTerrain ? scene.terrain!.data : null;
  const nd = scene.ndsm?.data ?? scene.heights.data;
  const src = scene.heights.data;
  const surface = new Float32Array(w * h).fill(-Infinity);
  const ground = new Float32Array(w * h);
  const gCnt = new Uint16Array(w * h);
  const cls = scene.classes && scene.classes.width === W && scene.classes.height === H ? scene.classes : null;
  const bId = cls ? cls.names.indexOf('building') : -1;
  const wId = cls ? cls.names.indexOf('water') : -1;
  const bCnt = bId >= 0 ? new Uint16Array(w * h) : null;
  const wCnt = wId >= 0 ? new Uint16Array(w * h) : null;
  const cnt = new Uint16Array(w * h);
  for (let y = 0; y < H; y++) {
    const ay = Math.floor(y / f) * w;
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const a = ay + Math.floor(x / f);
      const v = top ? top[i] + nd[i] : src[i];
      if (Number.isFinite(v)) {
        if (v > surface[a]) surface[a] = v;
        if (top) {
          ground[a] += top[i];
          gCnt[a]++;
        }
      }
      cnt[a]++;
      if (cls) {
        const c = cls.data[i];
        if (bCnt && c === bId) bCnt[a]++;
        if (wCnt && c === wId) wCnt[a]++;
      }
    }
  }
  for (let a = 0; a < w * h; a++) {
    if (surface[a] === -Infinity) surface[a] = NaN;
    ground[a] = top ? (gCnt[a] ? ground[a] / gCnt[a] : NaN) : 0;
  }
  const majority = (c: Uint16Array | null) => {
    if (!c) return null;
    const m = new Uint8Array(w * h);
    for (let a = 0; a < m.length; a++) m[a] = c[a] * 2 > cnt[a] ? 1 : 0;
    return m;
  };
  return { w, h, f, cellM: scene.gsd * f, surface, ground, hasTerrain, building: majority(bCnt), water: majority(wCnt), sceneW: W, sceneH: H };
}
