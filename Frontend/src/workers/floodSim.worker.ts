import { expose, transfer } from 'comlink';
import { FloodSim, simGridFrom, type FrameStats } from '@/lib/usecases/floodSim';

/** The flood hydraulics, stepped continuously while the flood scenario is on screen. Own worker: a telecom
 *  analysis on the analysis worker must not freeze the water. */
let sim: FloodSim | null = null;
let render = { base: 0, absolute: true };
let probes: Int32Array = new Int32Array(0);
let lastLevel = NaN;

export interface SimInit {
  /** The flood model's analysis grid (FloodModel w, h, cellM, ground, seedMask). */
  w: number;
  h: number;
  cellM: number;
  ground: Float32Array;
  seedMask: Uint8Array | null;
  /** Analysis cells per simulation cell (simFactor). */
  s: number;
  /** Start at rest at this level. */
  level: number;
  /** Render frame: surface heights are written relative to `base`, on the ground when `absolute`. */
  base: number;
  absolute: boolean;
  /** Simulation cells to read the water surface at (one per building). */
  probes: Int32Array;
}

export interface SimFrame extends FrameStats {
  /** 4 floats per simulation cell, see FloodSim.writeFrame. */
  data: Float32Array;
  /** Water surface at each probe, absolute metres; NaN = dry. */
  stages: Float32Array;
  timeS: number;
  /** Level unchanged and the water no longer moving: stepping can pause until the level changes. */
  settled: boolean;
}

const api = {
  init(p: SimInit): { w: number; h: number; cellM: number } {
    sim = new FloodSim(simGridFrom(p, p.s));
    sim.settle(p.level);
    render = { base: p.base, absolute: p.absolute };
    probes = p.probes;
    lastLevel = p.level;
    return { w: sim.w, h: sim.h, cellM: sim.dx };
  },

  /** Step toward `level` for about `budgetMs` of compute, covering at most `maxSimS` simulated seconds.
   *  `reuse` hands back an old frame buffer so the steady stream allocates nothing. */
  advance(level: number, budgetMs: number, maxSimS: number, reuse: Float32Array | null): SimFrame {
    if (!sim) throw new Error('Flood simulation not initialised');
    const { maxChangeM } = sim.advance(level, budgetMs, maxSimS);
    const n = sim.w * sim.h * 4;
    const data = reuse && reuse.length === n ? reuse : new Float32Array(n);
    const stages = new Float32Array(probes.length);
    const stats = sim.writeFrame(data, render.base, render.absolute, probes, stages);
    const settled = level === lastLevel && maxChangeM < 2e-3;
    lastLevel = level;
    return transfer({ ...stats, data, stages, timeS: sim.time, settled }, [data.buffer, stages.buffer]);
  },
};

export type FloodSimWorkerApi = typeof api;
expose(api);
