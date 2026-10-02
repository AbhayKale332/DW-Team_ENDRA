import { create } from 'zustand';
import { useScene } from './scene';

export type Tool = 'none' | 'probe' | 'measure' | 'profile';

export const TOOL_LABELS: Record<Tool, string> = {
  none: 'Navigate',
  probe: 'Probe height',
  measure: 'Measure',
  profile: 'Elevation profile',
};

/** What the measure tool's points mean: a polyline, a polygon, or one/two height read-outs. */
export type MeasureMode = 'length' | 'area' | 'height';

export const MEASURE_MODES: MeasureMode[] = ['length', 'area', 'height'];

export const MEASURE_MODE_LABELS: Record<MeasureMode, string> = { length: 'Length', area: 'Area', height: 'Height' };

/** A point on the height grid in fractional pixel coordinates (pixel centres at integers). */
export interface GridPoint {
  col: number;
  row: number;
}

/** Enough points for a result: a segment, a polygon, a spot height. */
export const MEASURE_MIN_POINTS: Record<MeasureMode, number> = { length: 2, area: 3, height: 1 };
const MAX_MEASURE_POINTS = 500;

interface ToolState {
  tool: Tool;
  probe: GridPoint | null;
  /** Measure tool points: polyline (length), polygon corners (area) or one/two spots (height). */
  measure: GridPoint[];
  measureMode: MeasureMode;
  /** The shape is finished (Enter, double-click, closing the ring, or the second height point): no rubber band,
   *  and the next click starts a new one. */
  measureDone: boolean;
  profile: GridPoint[];
  hover: GridPoint | null;
  /** Pointer position over the canvas (CSS px from its top-left) for the hover card; null when off the terrain. */
  hoverScreen: { x: number; y: number } | null;
  /** 0..1 position along the profile highlighted from the chart. */
  profileCursor: number | null;
  set: (p: Partial<Omit<ToolState, 'set' | 'clear' | 'addPoint'>>) => void;
  addPoint: (p: GridPoint) => void;
  clear: () => void;
}

export const useTool = create<ToolState>()((set, get) => ({
  tool: 'none',
  probe: null,
  measure: [],
  measureMode: 'length',
  measureDone: false,
  profile: [],
  hover: null,
  hoverScreen: null,
  profileCursor: null,
  set: (p) => set(p),
  addPoint: (p) => {
    const { tool, measure, measureMode, measureDone, profile } = get();
    if (tool === 'probe') set({ probe: p });
    else if (tool === 'measure') {
      // a finished shape (or a full height pair) is replaced by the next click
      if (measureDone || (measureMode === 'height' && measure.length >= 2)) return set({ measure: [p], measureDone: false });
      const last = measure[measure.length - 1];
      if (last && last.col === p.col && last.row === p.row) return;
      const next = [...measure, p].slice(0, MAX_MEASURE_POINTS);
      set({ measure: next, measureDone: measureMode === 'height' && next.length === 2 });
    } else if (tool === 'profile') set({ profile: [...profile, p].slice(-24) });
  },
  clear: () => set({ probe: null, measure: [], measureDone: false, profile: [], profileCursor: null }),
}));

/** Turns the measure tool on in a mode. A different mode starts from no points. */
export function startMeasure(mode: MeasureMode) {
  const st = useTool.getState();
  const keep = st.tool === 'measure' && st.measureMode === mode;
  st.set({ tool: 'measure', measureMode: mode, ...(keep ? {} : { measure: [], measureDone: false }) });
}

/** Back to plain navigation; the measurement goes with the tool. */
export function stopMeasure() {
  useTool.getState().set({ tool: 'none', measure: [], measureDone: false });
}

export function undoMeasurePoint() {
  const { measure } = useTool.getState();
  if (measure.length) useTool.getState().set({ measure: measure.slice(0, -1), measureDone: false });
}

export function clearMeasure() {
  useTool.getState().set({ measure: [], measureDone: false });
}

/** Ends the line / closes the polygon (Height finishes itself on its second point). False when there is nothing
 *  to finish or too few points. */
export function finishMeasure(): boolean {
  const { tool, measure, measureMode, measureDone } = useTool.getState();
  if (tool !== 'measure' || measureMode === 'height' || measureDone || measure.length < MEASURE_MIN_POINTS[measureMode]) return false;
  useTool.getState().set({ measureDone: true });
  return true;
}

// Points are pixels of one grid: another scene makes them meaningless.
useScene.subscribe((s, prev) => {
  if (s.scene?.id === prev.scene?.id) return;
  useTool.getState().clear();
  if (!s.scene) useTool.getState().set({ tool: 'none' });
});
